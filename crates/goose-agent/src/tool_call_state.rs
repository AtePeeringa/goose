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

pub(crate) fn add_error_response(tool_responses: &mut Message, request: &ToolRequest, text: &str) {
    tool_responses.add_tool_response_with_metadata(
        request.id.clone(),
        Ok(CallToolResult::error(vec![ContentBlock::text(text)])),
        request.metadata.as_ref(),
    );
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ToolCallState {
    Unchecked,
    Allowed,
    Denied,
    AwaitingConfirmation,
    Confirmed(Permission),
}

pub(crate) fn open_tool_call_states(messages: &[Message]) -> Vec<(ToolRequest, ToolCallState)> {
    let mut tool_response_ids = HashSet::new();
    let mut confirmation_requests = HashSet::new();
    let mut confirmations = HashMap::new();
    let mut requests = Vec::new();
    for content in messages.iter().flat_map(|message| &message.content) {
        match content {
            MessageContent::ToolRequest(request) => requests.push(request),
            MessageContent::ToolResponse(response) => {
                tool_response_ids.insert(response.id.as_str());
            }
            MessageContent::ActionRequired(action) => match &action.data {
                ActionRequiredData::ToolConfirmation { id, .. } => {
                    confirmation_requests.insert(id.as_str());
                }
                ActionRequiredData::ToolConfirmationResponse { id, permission } => {
                    confirmations.insert(id.as_str(), permission);
                }
                _ => {}
            },
            _ => {}
        }
    }

    requests
        .into_iter()
        .filter(|request| !tool_response_ids.contains(request.id.as_str()))
        .map(|request| {
            let state = match request_executable(request) {
                None => ToolCallState::Unchecked,
                Some(true) => ToolCallState::Allowed,
                Some(false) if !confirmation_requests.contains(request.id.as_str()) => {
                    ToolCallState::Denied
                }
                Some(false) => match confirmations.get(request.id.as_str()) {
                    Some(permission) => ToolCallState::Confirmed((*permission).clone()),
                    None => ToolCallState::AwaitingConfirmation,
                },
            };
            (request.clone(), state)
        })
        .collect()
}
