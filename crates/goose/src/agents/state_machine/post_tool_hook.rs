use anyhow::Result;
use async_trait::async_trait;
use goose_agent::post_tool_execution::PostToolCallHook;

use crate::agents::state_machine::ops_toolcalling::{emit_extended_post_hooks, emit_post_tool_use};
use crate::conversation::message::{ToolRequest, ToolResponse};
use crate::conversation::Conversation;
use crate::hooks::{HookEvent, HookManager};
use crate::session::Session;

pub struct GoosePostToolHook {
    hook_manager: HookManager,
}

impl GoosePostToolHook {
    pub fn new(hook_manager: HookManager) -> Self {
        Self { hook_manager }
    }
}

#[async_trait]
impl PostToolCallHook<Session> for GoosePostToolHook {
    async fn after_call(
        &self,
        session: &Session,
        _conversation: &Conversation,
        request: &ToolRequest,
        response: &ToolResponse,
    ) -> Result<()> {
        let Ok(tool_call) = &request.tool_call else {
            return Ok(());
        };
        let tool_input = tool_call
            .arguments
            .as_ref()
            .map(|arguments| serde_json::Value::Object(arguments.clone()));
        let event = emit_post_tool_use(
            &self.hook_manager,
            &session.id,
            &session.working_dir.to_string_lossy(),
            &tool_call.name,
            &request.id,
            tool_input.as_ref(),
            &response.tool_result,
        )
        .await;
        if event == HookEvent::PostToolUse {
            emit_extended_post_hooks(
                &self.hook_manager,
                &tool_call.name,
                tool_input.as_ref(),
                session,
            )
            .await;
        }
        Ok(())
    }
}
