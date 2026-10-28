//! Answers tool requests that no other operation handled.

use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::{CallToolResult, ContentBlock, ErrorCode, ErrorData};
use tracing_futures::Instrument;

use crate::agents::final_output_tool::FINAL_OUTPUT_TOOL_NAME;
use crate::agents::state_machine::effects::GooseEffect;
use crate::agents::state_machine::ops_toolcalling::{
    answered_without_running, emit_extended_pre_hooks, pending_tool_requests,
    request_was_advertised, tool_span, ToolDisposition,
};
use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, Emitter, Operation, OperationResult,
};
use crate::agents::tool_execution::{CHAT_MODE_TOOL_SKIPPED_RESPONSE, DECLINED_RESPONSE};
use crate::config::GooseMode;
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::hooks::HookManager;
use crate::session::Session;

pub(super) const UNCLAIMED_TOOL_ERROR: &str = "goose.unclaimed_tool";

pub struct RemainingToolCallsOperation {
    hook_manager: HookManager,
}

impl RemainingToolCallsOperation {
    pub fn new(hook_manager: HookManager) -> Self {
        Self { hook_manager }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for RemainingToolCallsOperation {
    fn name(&self) -> &'static str {
        "remaining_tool_calls"
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let active_final_output = session
            .recipe
            .as_ref()
            .is_some_and(|recipe| recipe.response.is_some());
        let messages = messages_since_kickoff(conversation)?;
        let pending: Vec<_> = pending_tool_requests(messages)
            .into_iter()
            .filter(|(request, disposition)| {
                // Reserve for RecipeOperation only the final-output calls it will
                // actually execute. A declined one is left here so it still gets a
                // response; RecipeOperation matches on Execute alone and would
                // otherwise leave the request unanswered, which strict providers can
                // reject on the next request.
                !(active_final_output
                    && *disposition == ToolDisposition::Execute
                    && request_was_advertised(messages, request)
                    && request
                        .tool_call
                        .as_ref()
                        .is_ok_and(|tool_call| tool_call.name == FINAL_OUTPUT_TOOL_NAME))
            })
            .collect();
        if pending.is_empty() {
            return not_applicable();
        }

        let mut response = Message::user();
        for (request, disposition) in pending {
            let tool_name = request
                .tool_call
                .as_ref()
                .map(|tool_call| tool_call.name.as_ref())
                .unwrap_or("unknown");
            let span = tool_span(tool_name, &request.id, &session.id);
            let (result, unclaimed) = match disposition {
                ToolDisposition::ParseError(error) => (
                    Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "The tool call could not be parsed: {error}. Correct the arguments and try again."
                    ))])),
                    false,
                ),
                ToolDisposition::Execute if !request_was_advertised(messages, &request) => {
                    span.record("error.type", "tool_not_available");
                    (
                        Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                            "Tool '{tool_name}' is not available."
                        ))])),
                        true,
                    )
                }
                ToolDisposition::Execute if session.goose_mode == GooseMode::Chat => (
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        CHAT_MODE_TOOL_SKIPPED_RESPONSE,
                    )])),
                    false,
                ),
                ToolDisposition::Execute => {
                    match request.tool_call.as_ref() {
                        Ok(tool_call) => {
                            let tool_input = tool_call
                                .arguments
                                .as_ref()
                                .map(|arguments| serde_json::Value::Object(arguments.clone()));
                            emit_extended_pre_hooks(
                                &self.hook_manager,
                                &tool_call.name,
                                tool_input.as_ref(),
                                session,
                            )
                            .instrument(span.clone())
                            .await;
                            if tool_call.name == FINAL_OUTPUT_TOOL_NAME && !active_final_output {
                                span.record("error.type", "final_output_not_defined");
                                (
                                    Err(ErrorData::new(
                                        ErrorCode::INTERNAL_ERROR,
                                        "Final output tool not defined".to_string(),
                                        None,
                                    )),
                                    false,
                                )
                            } else {
                                span.record("error.type", "tool_not_available");
                                (
                                    Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                                        "Tool '{}' is not available.",
                                        tool_call.name
                                    ))])),
                                    true,
                                )
                            }
                        }
                        Err(error) => (
                            Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                                "The tool call could not be parsed: {error}."
                            ))])),
                            false,
                        ),
                    }
                }
                ToolDisposition::Decline => (
                    Ok(CallToolResult::error(vec![ContentBlock::text(
                        DECLINED_RESPONSE,
                    )])),
                    false,
                ),
            };
            let mut metadata = request.metadata.clone();
            if unclaimed {
                metadata
                    .get_or_insert_default()
                    .insert(UNCLAIMED_TOOL_ERROR.to_string(), true.into());
            }
            response.add_tool_response_with_metadata(request.id, result, metadata.as_ref());
        }

        applied(answered_without_running(response, emit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{Recipe, Response};
    use goose_agent::pre_tool_execution::EXECUTABLE_KEY;
    use rmcp::model::CallToolRequestParams;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    async fn run(
        session: &Session,
        messages: Vec<Message>,
    ) -> Result<OperationResult<GooseEffect>> {
        let (tx, _rx) = mpsc::unbounded_channel();
        let emit = Emitter::new(tx, CancellationToken::new());
        RemainingToolCallsOperation::new(HookManager::default())
            .run(session, &Conversation::new_unvalidated(messages), &emit)
            .await
    }

    #[tokio::test]
    async fn a_call_awaiting_approval_is_left_to_the_post_step() -> Result<()> {
        let messages = vec![
            Message::user().with_text("run it"),
            Message::assistant().with_tool_request_with_metadata(
                "call",
                Ok(CallToolRequestParams::new("shell")),
                None,
                Some(serde_json::json!({ EXECUTABLE_KEY: false })),
            ),
            Message::assistant()
                .with_action_required("call", "shell".to_string(), Default::default(), None)
                .user_only(),
        ];

        assert!(matches!(
            run(&Session::default(), messages).await?,
            OperationResult::NotApplicable
        ));
        Ok(())
    }

    #[tokio::test]
    async fn a_final_output_call_is_left_to_the_recipe_step() -> Result<()> {
        let session = Session {
            recipe: Some(
                Recipe::builder()
                    .title("recipe")
                    .description("recipe")
                    .instructions("answer")
                    .response(Response {
                        json_schema: Some(serde_json::json!({ "type": "object" })),
                    })
                    .build()
                    .unwrap(),
            ),
            ..Session::default()
        };
        let messages = vec![
            Message::user().with_text("go"),
            Message::assistant().with_tool_request(
                "final",
                Ok(CallToolRequestParams::new(FINAL_OUTPUT_TOOL_NAME)),
            ),
        ];

        assert!(matches!(
            run(&session, messages).await?,
            OperationResult::NotApplicable
        ));
        Ok(())
    }
}
