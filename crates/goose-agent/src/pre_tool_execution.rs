use std::sync::Arc;

use anyhow::Result;
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
use crate::tool_call_state::{add_error_response, tool_calls, ToolCallState};
pub use crate::tool_call_state::{mark_executable, request_executable, EXECUTABLE_KEY};

pub const DEFAULT_DECLINED_MESSAGE: &str = "The user declined to run this tool.";

#[derive(Debug)]
pub enum PreToolCallDecision {
    Allow,
    Deny { reason: String },
    Ask { prompt: Option<String> },
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait PreToolCallHook<S>: MaybeSend + MaybeSync {
    async fn before_call(
        &self,
        session: &S,
        conversation: &Conversation,
        request: &ToolRequest,
    ) -> Result<PreToolCallDecision>;

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

pub struct PreToolExecutionOperation<'a, S> {
    hook: Arc<dyn PreToolCallHook<S> + 'a>,
    declined_message: String,
}

impl<'a, S> PreToolExecutionOperation<'a, S> {
    pub fn new(hook: Arc<dyn PreToolCallHook<S> + 'a>) -> Self {
        Self {
            hook,
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

fn apply_decision(
    decision: PreToolCallDecision,
    request: &ToolRequest,
    call: &CallToolRequestParams,
    effects: &mut Vec<ConversationEffect>,
    tool_responses: &mut Message,
) {
    match decision {
        PreToolCallDecision::Allow => {
            effects.push(mark_executable(&request.id, true));
        }
        PreToolCallDecision::Deny { reason } => {
            add_error_response(tool_responses, request, &reason);
        }
        PreToolCallDecision::Ask { prompt } => {
            effects.push(mark_executable(&request.id, false));
            effects.push(confirmation_request(request, call, prompt).into());
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S, E> Operation<S, E> for PreToolExecutionOperation<'_, S>
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<ConversationEffect> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "pre_tool_execution"
    }

    async fn run(
        &self,
        session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let mut effects = Vec::new();
        let mut tool_responses = Message::user();
        for (request, state) in tool_calls(messages_since_kickoff(conversation)?) {
            if request.was_executed_externally() {
                continue;
            }
            let Ok(call) = &request.tool_call else {
                continue;
            };
            match state {
                ToolCallState::Unchecked => {
                    let decision = self
                        .hook
                        .before_call(session, conversation, request)
                        .await?;
                    apply_decision(decision, request, call, &mut effects, &mut tool_responses);
                }
                ToolCallState::Confirmed(permission) => {
                    self.hook
                        .on_confirmation(session, request, permission)
                        .await?;
                    if matches!(permission, Permission::AllowOnce | Permission::AlwaysAllow) {
                        effects.push(mark_executable(&request.id, true));
                    } else {
                        add_error_response(&mut tool_responses, request, &self.declined_message);
                    }
                }
                ToolCallState::Responded(_)
                | ToolCallState::Allowed
                | ToolCallState::AwaitingConfirmation
                | ToolCallState::Denied => {}
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
