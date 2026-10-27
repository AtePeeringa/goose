use std::collections::{HashMap, HashSet};

use goose_provider_types::conversation::message::{
    ActionRequiredData, Message, MessageContent, ToolRequest,
};
use goose_provider_types::permission::Permission;
use rmcp::model::{CallToolResult, ContentBlock};

use crate::operation::ConversationEffect;

pub const EXECUTABLE_KEY: &str = "goose.executable";

pub(crate) fn tool_meta_value(request: &ToolRequest, key: &str) -> Option<bool> {
    request
        .tool_meta
        .as_ref()
        .and_then(|meta| meta.get(key))
        .and_then(serde_json::Value::as_bool)
}

pub fn request_executable(request: &ToolRequest) -> Option<bool> {
    tool_meta_value(request, EXECUTABLE_KEY)
}

pub(crate) fn mark_executable(tool_call_id: &str, executable: bool) -> ConversationEffect {
    ConversationEffect::PatchToolRequestMeta {
        tool_call_id: tool_call_id.to_string(),
        patch: serde_json::json!({ EXECUTABLE_KEY: executable }),
    }
}

pub(crate) fn add_error_answer(answers: &mut Message, request: &ToolRequest, text: &str) {
    answers.add_tool_response_with_metadata(
        request.id.clone(),
        Ok(CallToolResult::error(vec![ContentBlock::text(text)])),
        request.metadata.as_ref(),
    );
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ToolCallState {
    Unchecked,
    Allowed,
    Waiting,
    UserAnswered(Permission),
    MarkedNotRunUnanswered,
}

pub(crate) fn open_tool_call_states(messages: &[Message]) -> Vec<(ToolRequest, ToolCallState)> {
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
                Some(false) if !questions.contains(request.id.as_str()) => {
                    ToolCallState::MarkedNotRunUnanswered
                }
                Some(false) => match user_answers.get(request.id.as_str()) {
                    Some(permission) => ToolCallState::UserAnswered((*permission).clone()),
                    None => ToolCallState::Waiting,
                },
            };
            (request.clone(), state)
        })
        .collect()
}
