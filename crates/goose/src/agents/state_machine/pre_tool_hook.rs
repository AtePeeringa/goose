use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::pre_tool_execution::{PreToolCallDecision, PreToolCallHook};

use crate::agents::extension_manager::ExtensionLease;
use crate::agents::state_machine::messages_since_kickoff;
use crate::agents::state_machine::ops_toolcalling::{request_was_advertised, tool_span};
use crate::agents::tool_execution::DECLINED_RESPONSE;
use crate::config::permission::PermissionLevel;
use crate::config::GooseMode;
use crate::conversation::message::ToolRequest;
use crate::conversation::Conversation;
use crate::hooks::{HookChainOutcome, HookContext, HookDenial, HookEvent, HookManager};
use crate::permission::Permission;
use crate::session::Session;
use crate::tool_inspection::{
    get_security_finding_id_from_results, InspectionAction, ToolInspectionManager,
};

/// Observation-only record of what the `PreToolUse` chain decided. Carries
/// no veto: the decision has already been made by the time this runs.
async fn emit_pre_tool_use_result(
    hook_manager: &HookManager,
    session_id: &str,
    working_dir: &Path,
    tool_call_id: &str,
    tool_name: &str,
    tool_input: Option<&serde_json::Value>,
    outcome: &HookChainOutcome,
) {
    if !hook_manager.has_hooks(HookEvent::PreToolUseResult) {
        return;
    }
    let context = HookContext::new(HookEvent::PreToolUseResult, session_id)
        .with_tool(tool_name.to_string(), tool_input.cloned())
        .with_tool_call_id(tool_call_id)
        .with_working_dir(working_dir.to_string_lossy().to_string())
        .with_pre_tool_use_outcome(outcome);
    hook_manager.emit_pre_tool_use_result(context).await;
}

async fn pre_tool_use_denial(
    hook_manager: &HookManager,
    session_id: &str,
    working_dir: &Path,
    tool_call_id: &str,
    tool_name: &str,
    tool_input: Option<&serde_json::Value>,
) -> Option<HookDenial> {
    let outcome = if hook_manager.has_hooks(HookEvent::PreToolUse) {
        let context = HookContext::new(HookEvent::PreToolUse, session_id)
            .with_tool(tool_name.to_string(), tool_input.cloned())
            .with_tool_call_id(tool_call_id)
            .with_working_dir(working_dir.to_string_lossy().to_string());
        hook_manager
            .emit_blocking_with_outcome(HookEvent::PreToolUse, context)
            .await
    } else {
        HookChainOutcome::allow(false)
    };

    // Emitted before the denial returns, so an observer sees the denial
    // before the model receives the refusal. Best effort, like every other
    // hook emission: a subscriber that fails or is absent changes nothing.
    emit_pre_tool_use_result(
        hook_manager,
        session_id,
        working_dir,
        tool_call_id,
        tool_name,
        tool_input,
        &outcome,
    )
    .await;

    outcome.denial()
}

pub struct GoosePreToolHook<'a> {
    hook_manager: HookManager,
    tool_inspection_manager: &'a ToolInspectionManager,
    extension_lease: Arc<Mutex<Option<Arc<ExtensionLease>>>>,
}

impl<'a> GoosePreToolHook<'a> {
    pub fn new(
        hook_manager: HookManager,
        tool_inspection_manager: &'a ToolInspectionManager,
        extension_lease: Arc<Mutex<Option<Arc<ExtensionLease>>>>,
    ) -> Self {
        Self {
            hook_manager,
            tool_inspection_manager,
            extension_lease,
        }
    }

    fn working_dir(&self, session: &Session) -> PathBuf {
        self.extension_lease
            .lock()
            .expect("extension lease unavailable")
            .as_ref()
            .filter(|lease| lease.scope_id() == session.id)
            .and_then(|lease| lease.working_dir())
            .map_or_else(|| session.working_dir.clone(), Path::to_path_buf)
    }

    async fn check_approval(
        &self,
        session: &Session,
        conversation: &Conversation,
        request: &ToolRequest,
    ) -> Result<PreToolCallDecision> {
        let requests = [request.clone()];
        let inspection_results = self
            .tool_inspection_manager
            .inspect_tools(
                &session.id,
                &requests,
                conversation.messages(),
                session.goose_mode,
            )
            .await?;
        let Some(check) = self
            .tool_inspection_manager
            .process_inspection_results_with_permission_inspector(&requests, &inspection_results)
        else {
            return Ok(PreToolCallDecision::Allow);
        };
        if !check.denied.is_empty() {
            return Ok(PreToolCallDecision::Deny {
                reason: DECLINED_RESPONSE.to_string(),
            });
        }
        if check.needs_approval.is_empty() {
            return Ok(PreToolCallDecision::Allow);
        }

        let prompt = inspection_results
            .iter()
            .find(|result| result.tool_request_id == request.id)
            .and_then(|result| match &result.action {
                InspectionAction::RequireApproval(Some(message)) => Some(message.clone()),
                _ => None,
            });
        if let Some(finding_id) =
            get_security_finding_id_from_results(&request.id, &inspection_results)
        {
            tracing::info!(
                monotonic_counter.goose.prompt_injection_user_decisions = 1,
                security.event_type = "approval_request",
                security.finding_id = %finding_id,
                tool.request_id = %request.id,
                "security finding: approval requested"
            );
        }
        Ok(PreToolCallDecision::Ask { prompt })
    }
}

#[async_trait]
impl PreToolCallHook<Session> for GoosePreToolHook<'_> {
    async fn before_call(
        &self,
        session: &Session,
        conversation: &Conversation,
        request: &ToolRequest,
    ) -> Result<PreToolCallDecision> {
        if session.goose_mode == GooseMode::Chat
            || !request_was_advertised(messages_since_kickoff(conversation)?, request)
        {
            return Ok(PreToolCallDecision::Allow);
        }
        let Ok(tool_call) = &request.tool_call else {
            return Ok(PreToolCallDecision::Allow);
        };
        let tool_input = tool_call
            .arguments
            .as_ref()
            .map(|arguments| serde_json::Value::Object(arguments.clone()));
        if let Some(denial) = pre_tool_use_denial(
            &self.hook_manager,
            &session.id,
            &self.working_dir(session),
            &request.id,
            &tool_call.name,
            tool_input.as_ref(),
        )
        .await
        {
            tool_span(&tool_call.name, &request.id, &session.id)
                .record("error.type", denial.error_type);
            return Ok(PreToolCallDecision::Deny {
                reason: denial.message,
            });
        }
        self.check_approval(session, conversation, request).await
    }

    async fn on_confirmation(
        &self,
        _session: &Session,
        request: &ToolRequest,
        permission: &Permission,
    ) -> Result<()> {
        let level = match permission {
            Permission::AlwaysAllow => PermissionLevel::AlwaysAllow,
            Permission::AlwaysDeny => PermissionLevel::NeverAllow,
            _ => return Ok(()),
        };
        let Ok(tool_call) = &request.tool_call else {
            return Ok(());
        };
        self.tool_inspection_manager
            .update_permission_manager(&tool_call.name, level)
            .await;
        Ok(())
    }
}
