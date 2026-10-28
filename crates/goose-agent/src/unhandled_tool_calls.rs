use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::{Message, ToolRequest};
use goose_provider_types::conversation::Conversation;
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};

use crate::operation::{
    applied, messages_since_kickoff, not_applicable, yielded, ConversationEffect, Emitter,
    Operation, OperationResult,
};
use crate::tool_call_state::{add_error_response, tool_calls, ToolCallState};

pub const NOT_HANDLED_MESSAGE: &str = "This tool call was not handled.";

pub struct UnhandledToolCallsOperation;

impl UnhandledToolCallsOperation {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UnhandledToolCallsOperation {
    fn default() -> Self {
        Self::new()
    }
}

fn respond_to_unhandled_calls(
    calls: &[(&ToolRequest, ToolCallState<'_>)],
    emit: &Emitter,
) -> Vec<ConversationEffect> {
    let mut effects = Vec::new();
    let mut tool_responses = Message::user();
    for (request, state) in calls {
        if *state != ToolCallState::AwaitingConfirmation {
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
impl<S, E> Operation<S, E> for UnhandledToolCallsOperation
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<ConversationEffect> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "unhandled_tool_calls"
    }

    async fn run(
        &self,
        _session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let open_calls: Vec<_> = tool_calls(messages_since_kickoff(conversation)?)
            .into_iter()
            .filter(|(request, state)| state.is_open() && !request.was_executed_externally())
            .collect();
        let effects = respond_to_unhandled_calls(&open_calls, emit);
        if !effects.is_empty() {
            return applied(effects.into_iter().map(E::from));
        }
        if open_calls
            .iter()
            .any(|(_, state)| *state == ToolCallState::AwaitingConfirmation)
        {
            return yielded();
        }
        not_applicable()
    }
}
