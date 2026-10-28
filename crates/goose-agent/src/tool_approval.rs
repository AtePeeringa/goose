use std::sync::Arc;

use anyhow::{ensure, Result};
use async_trait::async_trait;
use goose_provider_types::conversation::message::{Message, MessageContent, ToolRequest};
use goose_provider_types::conversation::Conversation;
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};
use goose_provider_types::permission::Permission;
use rmcp::model::CallToolRequestParams;

use crate::operation::{
    applied, messages_since_kickoff, not_applicable, ConversationEffect, Emitter, Operation,
    OperationResult,
};
use crate::tool_call_state::{
    add_error_response, mark_approved, mark_executable, tool_calls, ToolCallState,
};

pub const DEFAULT_DECLINED_MESSAGE: &str = "The user declined to run this tool.";

#[derive(Debug)]
pub enum ApprovalDecision {
    Allow,
    Deny { reason: String },
    Ask { prompt: Option<String> },
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ToolApproval<S>: MaybeSend + MaybeSync {
    async fn check(
        &self,
        session: &S,
        conversation: &Conversation,
        requests: &[&ToolRequest],
    ) -> Result<Vec<ApprovalDecision>>;

    async fn on_confirmation(
        &self,
        _session: &S,
        _request: &ToolRequest,
        _permission: &Permission,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn confirmation_response_message(tool_call_id: &str, permission: Permission) -> Message {
    Message::user()
        .with_visibility(false, false)
        .with_content(MessageContent::action_required_tool_confirmation_response(
            tool_call_id.to_string(),
            permission,
        ))
        .with_generated_id_if_missing()
}

pub struct ToolApprovalOperation<'a, S> {
    approval: Arc<dyn ToolApproval<S> + 'a>,
    declined_message: String,
}

impl<'a, S> ToolApprovalOperation<'a, S> {
    pub fn new(approval: Arc<dyn ToolApproval<S> + 'a>) -> Self {
        Self {
            approval,
            declined_message: DEFAULT_DECLINED_MESSAGE.to_string(),
        }
    }

    pub fn with_declined_message(mut self, message: impl Into<String>) -> Self {
        self.declined_message = message.into();
        self
    }
}

fn confirmation_request(
    request: &ToolRequest,
    call: &CallToolRequestParams,
    prompt: Option<String>,
) -> Message {
    Message::assistant()
        .with_action_required(
            request.id.clone(),
            call.name.to_string(),
            call.arguments.clone().unwrap_or_default(),
            prompt,
        )
        .user_only()
        .with_generated_id_if_missing()
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S, E> Operation<S, E> for ToolApprovalOperation<'_, S>
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<ConversationEffect> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "tool_approval"
    }

    async fn run(
        &self,
        session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let mut effects = Vec::new();
        let mut tool_responses = Message::user();
        let mut unchecked = Vec::new();
        for (request, state) in tool_calls(messages_since_kickoff(conversation)?) {
            if request.was_executed_externally() {
                continue;
            }
            let Ok(call) = &request.tool_call else {
                continue;
            };
            match state {
                ToolCallState::Unchecked => unchecked.push((request, call)),
                ToolCallState::UserAnswered(permission) => {
                    self.approval
                        .on_confirmation(session, request, permission)
                        .await?;
                    if matches!(permission, Permission::AllowOnce | Permission::AlwaysAllow) {
                        effects.push(mark_approved(&request.id));
                    } else {
                        add_error_response(&mut tool_responses, request, &self.declined_message);
                    }
                }
                ToolCallState::Responded(_)
                | ToolCallState::Approved
                | ToolCallState::Executable
                | ToolCallState::AwaitingUser
                | ToolCallState::Blocked => {}
            }
        }

        if !unchecked.is_empty() {
            let requests: Vec<_> = unchecked.iter().map(|(request, _)| *request).collect();
            let decisions = self
                .approval
                .check(session, conversation, &requests)
                .await?;
            ensure!(
                decisions.len() == unchecked.len(),
                "tool approval returned {} decisions for {} tool calls",
                decisions.len(),
                unchecked.len()
            );
            for ((request, call), decision) in unchecked.into_iter().zip(decisions) {
                match decision {
                    ApprovalDecision::Allow => effects.push(mark_approved(&request.id)),
                    ApprovalDecision::Deny { reason } => {
                        add_error_response(&mut tool_responses, request, &reason)
                    }
                    ApprovalDecision::Ask { prompt } => {
                        effects.push(mark_executable(&request.id, false));
                        effects.push(confirmation_request(request, call, prompt).into());
                    }
                }
            }
        }

        if !tool_responses.content.is_empty() {
            effects.push(emit.message(tool_responses).into());
        }
        if effects.is_empty() {
            return not_applicable();
        }
        applied(effects.into_iter().map(E::from))
    }
}
