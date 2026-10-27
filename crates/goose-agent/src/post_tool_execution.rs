use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::{
    Message, MessageContent, ToolRequest, ToolResponse,
};
use goose_provider_types::conversation::Conversation;
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};

use crate::operation::{
    applied, messages_since_kickoff, not_applicable, yielded, ConversationEffect, Emitter,
    Operation, OperationResult,
};
use crate::tool_call_state::{
    add_error_response, mark_executable, open_tool_call_states, request_executable,
    tool_meta_value, ToolCallState,
};

pub const POST_HOOK_CALLED_KEY: &str = "goose.post_hook_called";
pub const NOT_HANDLED_MESSAGE: &str = "This tool call was not handled.";

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait PostToolCallHook<S>: MaybeSend + MaybeSync {
    async fn after_call(
        &self,
        session: &S,
        conversation: &Conversation,
        request: &ToolRequest,
        response: &ToolResponse,
    ) -> Result<()>;
}

pub struct PostToolExecutionOperation<'a, S> {
    hook: Option<Arc<dyn PostToolCallHook<S> + 'a>>,
}

impl<'a, S> PostToolExecutionOperation<'a, S> {
    pub fn new() -> Self {
        Self { hook: None }
    }

    pub fn with_hook(mut self, hook: Arc<dyn PostToolCallHook<S> + 'a>) -> Self {
        self.hook = Some(hook);
        self
    }
}

impl<S> Default for PostToolExecutionOperation<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

fn needs_post_hook(request: &ToolRequest) -> bool {
    !request.was_executed_externally()
        && request.tool_call.is_ok()
        && request_executable(request) != Some(false)
        && tool_meta_value(request, POST_HOOK_CALLED_KEY) != Some(true)
}

async fn call_post_hooks<S>(
    hook: &dyn PostToolCallHook<S>,
    session: &S,
    conversation: &Conversation,
    messages: &[Message],
) -> Result<Vec<ConversationEffect>> {
    let contents = || messages.iter().flat_map(|message| &message.content);
    let tool_responses_by_id: HashMap<&str, &ToolResponse> = contents()
        .filter_map(MessageContent::as_tool_response)
        .map(|response| (response.id.as_str(), response))
        .collect();
    let mut effects = Vec::new();
    for request in contents().filter_map(MessageContent::as_tool_request) {
        if !needs_post_hook(request) {
            continue;
        }
        let Some(response) = tool_responses_by_id.get(request.id.as_str()) else {
            continue;
        };
        hook.after_call(session, conversation, request, response)
            .await?;
        effects.push(ConversationEffect::PatchToolRequestMeta {
            tool_call_id: request.id.clone(),
            patch: serde_json::json!({ POST_HOOK_CALLED_KEY: true }),
        });
    }
    Ok(effects)
}

fn respond_to_unhandled_calls(
    calls: &[(ToolRequest, ToolCallState)],
    emit: &Emitter,
) -> Vec<ConversationEffect> {
    let mut effects = Vec::new();
    let mut tool_responses = Message::user();
    for (request, state) in calls {
        if *state != ToolCallState::AwaitingConfirmation {
            effects.push(mark_executable(&request.id, false));
            add_error_response(&mut tool_responses, request, NOT_HANDLED_MESSAGE);
        }
    }
    if !tool_responses.content.is_empty() {
        effects.push(ConversationEffect::AppendMessage(
            emit.message(tool_responses),
        ));
    }
    effects
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S, E> Operation<S, E> for PostToolExecutionOperation<'_, S>
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<ConversationEffect> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "post_tool_execution"
    }

    async fn run(
        &self,
        session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let messages = messages_since_kickoff(conversation)?;
        if let Some(hook) = &self.hook {
            let effects = call_post_hooks(hook.as_ref(), session, conversation, messages).await?;
            if !effects.is_empty() {
                return applied(effects.into_iter().map(E::from));
            }
        }

        let calls: Vec<_> = open_tool_call_states(messages)
            .into_iter()
            .filter(|(request, _)| !request.was_executed_externally())
            .collect();
        let effects = respond_to_unhandled_calls(&calls, emit);
        if !effects.is_empty() {
            return applied(effects.into_iter().map(E::from));
        }
        if calls
            .iter()
            .any(|(_, state)| *state == ToolCallState::AwaitingConfirmation)
        {
            return yielded();
        }
        not_applicable()
    }
}
