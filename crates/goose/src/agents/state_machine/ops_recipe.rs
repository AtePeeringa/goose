//! Applies recipe commands and enforces their structured final output.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use rmcp::model::{CallToolResult, ContentBlock, ErrorCode, ErrorData, Tool};
use tracing_futures::Instrument;

use crate::agents::extension_manager::ExtensionManager;
use crate::agents::final_output_tool::{
    structured_output_unsupported_message, FinalOutputTool, FINAL_OUTPUT_CONTINUATION_MESSAGE,
    FINAL_OUTPUT_NOT_ALONE_MESSAGE, FINAL_OUTPUT_TOOL_NAME,
};
use crate::agents::state_machine::ops_toolcalling::{
    answered_without_running, pending_advertised_tool_requests, tool_span, ToolDisposition,
};
use crate::agents::state_machine::{
    applied, ends_turn, messages_since_kickoff, not_applicable, yielded_with, ConversationEffect,
    Emitter, GooseEffect, Operation, OperationResult, SlashCommand,
};
use crate::agents::tool_execution::CHAT_MODE_TOOL_SKIPPED_RESPONSE;
use crate::config::extensions::is_extension_available;
use crate::config::GooseMode;
use crate::conversation::message::{Message, MessageContent};
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::Session;

pub(crate) fn final_output_tool(session: &Session) -> Result<Option<FinalOutputTool>> {
    session
        .recipe
        .as_ref()
        .and_then(|recipe| recipe.response.clone())
        .map(FinalOutputTool::try_new)
        .transpose()
        .map_err(|error| anyhow!(error))
}

fn assistant_block_bounds(messages: &[Message], message_index: usize) -> (usize, usize) {
    let start = (0..message_index)
        .rev()
        .take_while(|index| messages[*index].role == rmcp::model::Role::Assistant)
        .last()
        .unwrap_or(message_index);
    let end = (message_index + 1..messages.len())
        .take_while(|index| messages[*index].role == rmcp::model::Role::Assistant)
        .last()
        .map_or(message_index + 1, |index| index + 1);
    (start, end)
}

fn has_sibling_tool_calls(messages: &[Message], request_id: &str) -> bool {
    let Some(message_index) = messages.iter().position(|message| {
        message
            .content
            .iter()
            .filter_map(MessageContent::as_tool_request)
            .any(|request| request.id == request_id)
    }) else {
        return false;
    };
    let (start, end) = assistant_block_bounds(messages, message_index);
    messages[start..end]
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(MessageContent::as_tool_request)
        .any(|request| request.id != request_id && !request.was_executed_externally())
}

pub(crate) fn recipe_prompt_parts(session: &Session) -> Result<Vec<(String, String)>> {
    let instructions = session
        .recipe
        .as_ref()
        .and_then(|recipe| recipe.instructions.clone())
        .map(|instructions| ("recipe".to_string(), instructions));
    let final_output =
        final_output_tool(session)?.map(|tool| ("final_output".to_string(), tool.system_prompt()));
    Ok(instructions.into_iter().chain(final_output).collect())
}

pub struct RecipeOperation {
    provider: Arc<dyn Provider>,
    extension_manager: Arc<ExtensionManager>,
    response: std::sync::Mutex<Option<Message>>,
}

impl RecipeOperation {
    pub fn new(provider: Arc<dyn Provider>, extension_manager: Arc<ExtensionManager>) -> Self {
        Self {
            provider,
            extension_manager,
            response: std::sync::Mutex::default(),
        }
    }

    fn take_response(&self) -> Option<Message> {
        self.response.lock().unwrap().take()
    }

    fn command_error(
        &self,
        conversation: &Conversation,
        message: String,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let command = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("recipe command conversation has no kickoff message"))?;
        let message_id = command
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        let command = command.with_visibility(true, false);
        let response = Message::assistant()
            .with_text(message)
            .with_visibility(true, false);
        emit.message(command);
        let response = emit.message(response);
        yielded_with([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            response.into(),
        ])
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for RecipeOperation {
    fn name(&self) -> &'static str {
        "recipe"
    }

    async fn finalize_cancellation(
        &self,
        _session: &Session,
        _conversation: &Conversation,
        emit: &Emitter,
    ) -> Vec<GooseEffect> {
        match self.take_response() {
            Some(response) => vec![emit.message(response).into()],
            None => Vec::new(),
        }
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let (recipe, prompt) = match crate::slash_commands::recipe_slash_command::resolve_command(
            command.command,
            command.params_str,
        ) {
            Ok(Some(recipe)) => recipe,
            Ok(None) => return not_applicable(),
            Err(error) => return self.command_error(conversation, error, emit),
        };

        if let Some(response) = recipe.response.clone() {
            if let Err(error) = FinalOutputTool::try_new(response) {
                return self.command_error(
                    conversation,
                    format!("Recipe is not valid: {error}"),
                    emit,
                );
            }
        }

        for extension in recipe
            .extensions
            .iter()
            .flatten()
            .filter(|extension| is_extension_available(extension))
        {
            if let Err(error) = self
                .extension_manager
                .enable(&session.id, extension.clone())
                .await
            {
                return self.command_error(
                    conversation,
                    format!(
                        "Could not load recipe extension '{}': {error}",
                        extension.name()
                    ),
                    emit,
                );
            }
        }

        #[cfg(feature = "telemetry")]
        crate::posthog::emit_custom_slash_command_used();

        let command_message = messages_since_kickoff(conversation)?
            .first()
            .ok_or_else(|| anyhow!("recipe command conversation has no kickoff message"))?;
        let message_id = command_message
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        applied([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            GooseEffect::SetRecipe(Box::new(Some(recipe))),
            Message::user()
                .with_text(prompt)
                .with_visibility(false, true)
                .into(),
        ])
    }

    async fn inference_tools(&self, session: &Session) -> Result<Vec<Tool>> {
        Ok(final_output_tool(session)?
            .as_ref()
            .map(FinalOutputTool::tool)
            .into_iter()
            .collect())
    }

    async fn prompt_parts(
        &self,
        session: &Session,
        _conversation: &Conversation,
    ) -> Result<Vec<(String, String)>> {
        recipe_prompt_parts(session)
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let Some(mut final_output) = final_output_tool(session)? else {
            return not_applicable();
        };

        if !self.provider.supports_builtin_tools() {
            return self.command_error(
                conversation,
                structured_output_unsupported_message(self.provider.get_name()),
                emit,
            );
        }

        let messages = messages_since_kickoff(conversation)?;
        let pending = pending_advertised_tool_requests(messages).into_iter().find(
            |(request, disposition)| {
                *disposition == ToolDisposition::Execute
                    && request
                        .tool_call
                        .as_ref()
                        .is_ok_and(|tool_call| tool_call.name == FINAL_OUTPUT_TOOL_NAME)
            },
        );
        if let Some((request, _)) = pending {
            if session.goose_mode == GooseMode::Chat {
                let mut response = Message::user();
                response.add_tool_response_with_metadata(
                    request.id,
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        CHAT_MODE_TOOL_SKIPPED_RESPONSE,
                    )])),
                    request.metadata.as_ref(),
                );
                return applied(answered_without_running(response, emit));
            }
            if has_sibling_tool_calls(messages, &request.id) {
                let mut response = Message::user();
                response.add_tool_response_with_metadata(
                    request.id,
                    Err(ErrorData::new(
                        ErrorCode::INVALID_REQUEST,
                        FINAL_OUTPUT_NOT_ALONE_MESSAGE,
                        None,
                    )),
                    request.metadata.as_ref(),
                );
                return applied(answered_without_running(response, emit));
            }

            let tool_call = request
                .tool_call
                .map_err(|error| anyhow!("final output tool call could not be parsed: {error}"))?;
            let span = tool_span(&tool_call.name, &request.id, &session.id);
            let result = final_output
                .execute_tool_call(tool_call)
                .instrument(span.clone())
                .await;
            let output = result.result.instrument(span.clone()).await;
            match &output {
                Ok(result) if result.is_error == Some(true) => {
                    span.record("error.type", "tool_error");
                }
                Err(_) => {
                    span.record("error.type", "tool_execution_error");
                }
                _ => {}
            }
            let mut response = Message::user().with_generated_id_if_missing();
            response.add_tool_response_with_metadata(
                request.id.clone(),
                output,
                request.metadata.as_ref(),
            );
            *self.response.lock().unwrap() = Some(response);
            let response = self.take_response().expect("recipe response held");
            let response = emit.message(response);
            let Some(accepted) = final_output.final_output.take() else {
                return applied([response.into()]);
            };
            let accepted = emit.message(Message::assistant().with_text(accepted));
            return applied([response.into(), accepted.into()]);
        }

        if FinalOutputTool::successful_output(messages).is_some() {
            return not_applicable();
        }

        if ends_turn(messages) {
            let message = Message::user()
                .with_text(FINAL_OUTPUT_CONTINUATION_MESSAGE)
                .agent_only();
            let message = emit.message(message);
            return applied([message.into()]);
        }

        not_applicable()
    }
}
