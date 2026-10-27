use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::{
    ActionRequiredData, Message, MessageContent, ToolRequest,
};
use goose_provider_types::conversation::Conversation;
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};
use goose_provider_types::permission::Permission;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};

use crate::operation::{
    applied, messages_since_kickoff, not_applicable, ConversationEffect, Emitter, Operation,
    OperationResult,
};

pub const EXECUTABLE_KEY: &str = "goose.executable";
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

    async fn on_answer(
        &self,
        _session: &S,
        _request: &ToolRequest,
        _permission: &Permission,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn request_executable(request: &ToolRequest) -> Option<bool> {
    request
        .tool_meta
        .as_ref()
        .and_then(|meta| meta.get(EXECUTABLE_KEY))
        .and_then(|value| value.as_bool())
}

pub fn user_answer_message(tool_call_id: &str, permission: Permission) -> Message {
    Message::user()
        .with_visibility(false, false)
        .with_content(MessageContent::action_required_tool_confirmation_response(
            tool_call_id.to_string(),
            permission,
        ))
        .with_generated_id_if_missing()
}

pub(crate) fn mark_executable(tool_call_id: &str, executable: bool) -> ConversationEffect {
    ConversationEffect::PatchToolRequestMeta {
        tool_call_id: tool_call_id.to_string(),
        patch: serde_json::json!({ EXECUTABLE_KEY: executable }),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ToolCallState {
    Unchecked,
    Allowed,
    Waiting,
    UserAnswered(Permission),
    Declined,
}

pub(crate) fn call_states(messages: &[Message]) -> Vec<(ToolRequest, ToolCallState)> {
    let mut answered = HashSet::new();
    let mut questions = HashSet::new();
    let mut user_answers = HashMap::new();
    let mut requests = Vec::new();
    for content in messages.iter().flat_map(|message| &message.content) {
        match content {
            MessageContent::ToolRequest(request) => requests.push(request),
            MessageContent::ToolResponse(response) => {
                answered.insert(response.id.as_str());
            }
            MessageContent::ActionRequired(action) => match &action.data {
                ActionRequiredData::ToolConfirmation { id, .. } => {
                    questions.insert(id.as_str());
                }
                ActionRequiredData::ToolConfirmationResponse { id, permission } => {
                    user_answers.insert(id.as_str(), permission);
                }
                _ => {}
            },
            _ => {}
        }
    }

    requests
        .into_iter()
        .filter(|request| !answered.contains(request.id.as_str()))
        .map(|request| {
            let state = match request_executable(request) {
                None => ToolCallState::Unchecked,
                Some(true) => ToolCallState::Allowed,
                Some(false) if !questions.contains(request.id.as_str()) => ToolCallState::Declined,
                Some(false) => match user_answers.get(request.id.as_str()) {
                    Some(permission) => ToolCallState::UserAnswered((*permission).clone()),
                    None => ToolCallState::Waiting,
                },
            };
            (request.clone(), state)
        })
        .collect()
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

fn question(
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

pub(crate) fn add_error_answer(answers: &mut Message, request: &ToolRequest, text: &str) {
    answers.add_tool_response_with_metadata(
        request.id.clone(),
        Ok(CallToolResult::error(vec![ContentBlock::text(text)])),
        request.metadata.as_ref(),
    );
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
        let mut answers = Message::user();
        for (request, state) in &call_states(messages_since_kickoff(conversation)?) {
            if request.was_executed_externally() {
                continue;
            }
            let Ok(call) = &request.tool_call else {
                continue;
            };
            match state {
                ToolCallState::Unchecked => {
                    match self
                        .hook
                        .before_call(session, conversation, request)
                        .await?
                    {
                        PreToolCallDecision::Allow => {
                            effects.push(mark_executable(&request.id, true));
                        }
                        PreToolCallDecision::Deny { reason } => {
                            effects.push(mark_executable(&request.id, false));
                            add_error_answer(&mut answers, request, &reason);
                        }
                        PreToolCallDecision::Ask { prompt } => {
                            effects.push(mark_executable(&request.id, false));
                            effects.push(question(request, call, prompt).into());
                        }
                    }
                }
                ToolCallState::UserAnswered(permission) => {
                    self.hook.on_answer(session, request, permission).await?;
                    if matches!(permission, Permission::AllowOnce | Permission::AlwaysAllow) {
                        effects.push(mark_executable(&request.id, true));
                    } else {
                        add_error_answer(&mut answers, request, &self.declined_message);
                    }
                }
                ToolCallState::Allowed | ToolCallState::Waiting | ToolCallState::Declined => {}
            }
        }

        if !answers.content.is_empty() {
            effects.push(emit.message(answers).into());
        }
        if effects.is_empty() {
            return not_applicable();
        }
        applied(effects.into_iter().map(E::from))
    }
}
