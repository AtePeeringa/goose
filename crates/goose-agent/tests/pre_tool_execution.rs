use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::{
    events::AgentEvent,
    operation::{ConversationEffect, Emitter, Operation, OperationResult, RunStatus},
    pre_tool_execution::{
        confirmation_response_message, PreToolCallDecision, PreToolCallHook,
        PreToolExecutionOperation, DEFAULT_DECLINED_MESSAGE, EXECUTABLE_KEY,
    },
};
use goose_provider_types::conversation::{
    message::{
        ActionRequiredData, Message, MessageContent, ToolRequest, TOOL_META_EXTERNAL_DISPATCH_KEY,
    },
    Conversation,
};
use goose_provider_types::permission::Permission;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, ErrorData};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy)]
enum Reply {
    Allow,
    Deny(&'static str),
    Ask(Option<&'static str>),
    Fail,
}

#[derive(Default)]
struct FakeHook {
    replies: Vec<(&'static str, Reply)>,
    checked: Mutex<Vec<String>>,
    told: Mutex<Vec<(String, Permission)>>,
}

impl FakeHook {
    fn new(replies: &[(&'static str, Reply)]) -> Arc<Self> {
        Arc::new(Self {
            replies: replies.to_vec(),
            ..Default::default()
        })
    }

    fn checked(&self) -> Vec<String> {
        self.checked.lock().unwrap().clone()
    }

    fn told(&self) -> Vec<(String, Permission)> {
        self.told.lock().unwrap().clone()
    }
}

#[async_trait]
impl PreToolCallHook<()> for FakeHook {
    async fn before_call(
        &self,
        _session: &(),
        _conversation: &Conversation,
        request: &ToolRequest,
    ) -> Result<PreToolCallDecision> {
        self.checked.lock().unwrap().push(request.id.clone());
        let reply = self
            .replies
            .iter()
            .find(|(id, _)| *id == request.id)
            .map_or(Reply::Allow, |(_, reply)| *reply);
        match reply {
            Reply::Allow => Ok(PreToolCallDecision::Allow),
            Reply::Deny(reason) => Ok(PreToolCallDecision::Deny {
                reason: reason.to_string(),
            }),
            Reply::Ask(prompt) => Ok(PreToolCallDecision::Ask {
                prompt: prompt.map(str::to_string),
            }),
            Reply::Fail => anyhow::bail!("hook crashed"),
        }
    }

    async fn on_confirmation(
        &self,
        _session: &(),
        request: &ToolRequest,
        permission: &Permission,
    ) -> Result<()> {
        self.told
            .lock()
            .unwrap()
            .push((request.id.clone(), permission.clone()));
        Ok(())
    }
}

struct Outcome {
    status: Option<RunStatus>,
    marks: Vec<(String, bool)>,
    appended: Vec<Message>,
    shown: Vec<Message>,
}

impl Outcome {
    fn responses(&self) -> Vec<(String, String)> {
        self.appended.iter().flat_map(tool_responses).collect()
    }

    fn shown_responses(&self) -> Vec<(String, String)> {
        self.shown.iter().flat_map(tool_responses).collect()
    }

    fn confirmation_requests(&self) -> Vec<&Message> {
        self.appended
            .iter()
            .filter(|message| confirmation_request_of(message).is_some())
            .collect()
    }
}

async fn run(
    operation: &PreToolExecutionOperation<'_, ()>,
    messages: Vec<Message>,
) -> Result<Outcome> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, CancellationToken::new());
    let conversation = Conversation::new_unvalidated(messages);
    let result =
        Operation::<(), ConversationEffect>::run(operation, &(), &conversation, &emit).await?;
    let shown = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|event| match event {
            AgentEvent::Message(message) => Some(message),
            _ => None,
        })
        .collect();
    let mut outcome = Outcome {
        status: None,
        marks: Vec::new(),
        appended: Vec::new(),
        shown,
    };
    if let OperationResult::Applied(result) = result {
        outcome.status = Some(result.status);
        for effect in result.effects {
            match effect {
                ConversationEffect::AppendMessage(message) => outcome.appended.push(message),
                ConversationEffect::PatchToolRequestMeta {
                    tool_call_id,
                    patch,
                } => outcome
                    .marks
                    .push((tool_call_id, patch[EXECUTABLE_KEY].as_bool().unwrap())),
                _ => panic!("unexpected effect"),
            }
        }
    }
    Ok(outcome)
}

fn tool_responses(message: &Message) -> Vec<(String, String)> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            MessageContent::ToolResponse(response) => {
                let result = response.tool_result.as_ref().unwrap();
                assert_eq!(result.is_error, Some(true));
                let text = result.content[0].as_text().unwrap().text.clone();
                Some((response.id.clone(), text))
            }
            _ => None,
        })
        .collect()
}

fn confirmation_request_of(message: &Message) -> Option<(String, String, Option<String>)> {
    message.content.iter().find_map(|content| match content {
        MessageContent::ActionRequired(action) => match &action.data {
            ActionRequiredData::ToolConfirmation {
                id,
                tool_name,
                prompt,
                ..
            } => Some((id.clone(), tool_name.clone(), prompt.clone())),
            _ => None,
        },
        _ => None,
    })
}

fn kickoff() -> Message {
    Message::user().with_text("hi")
}

fn calls(ids: &[&str]) -> Message {
    ids.iter().fold(Message::assistant(), |message, id| {
        message.with_tool_request(*id, Ok(CallToolRequestParams::new(id.to_string())))
    })
}

fn marked_call(id: &str, executable: bool) -> Message {
    Message::assistant().with_tool_request_with_metadata(
        id,
        Ok(CallToolRequestParams::new(id.to_string())),
        None,
        Some(json!({ EXECUTABLE_KEY: executable })),
    )
}

fn confirmation_request(id: &str) -> Message {
    Message::assistant()
        .with_action_required(id, id.to_string(), Default::default(), None)
        .user_only()
}

fn click(id: &str, permission: Permission) -> Message {
    confirmation_response_message(id, permission)
}

fn tool_response(id: &str) -> Message {
    Message::user().with_tool_response(
        id,
        Ok(CallToolResult::success(vec![ContentBlock::text("done")])),
    )
}

#[tokio::test]
async fn allow_marks_the_call_true() -> Result<()> {
    let hook = FakeHook::new(&[("a", Reply::Allow)]);
    let operation = PreToolExecutionOperation::new(hook.clone());

    let outcome = run(&operation, vec![kickoff(), calls(&["a"])]).await?;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    assert_eq!(outcome.marks, vec![("a".to_string(), true)]);
    assert!(outcome.appended.is_empty());
    assert_eq!(hook.checked(), vec!["a"]);
    Ok(())
}

#[tokio::test]
async fn deny_streams_an_error_response_with_the_reason_and_leaves_no_mark() -> Result<()> {
    let operation = PreToolExecutionOperation::new(FakeHook::new(&[("a", Reply::Deny("no"))]));

    let outcome = run(&operation, vec![kickoff(), calls(&["a"])]).await?;

    let expected = vec![("a".to_string(), "no".to_string())];
    assert!(outcome.marks.is_empty());
    assert_eq!(outcome.responses(), expected);
    assert_eq!(outcome.shown_responses(), expected);
    Ok(())
}

#[tokio::test]
async fn ask_marks_false_and_saves_a_confirmation_request_without_streaming_it() -> Result<()> {
    let operation =
        PreToolExecutionOperation::new(FakeHook::new(&[("a", Reply::Ask(Some("careful")))]));

    let outcome = run(&operation, vec![kickoff(), calls(&["a"])]).await?;

    assert_eq!(outcome.marks, vec![("a".to_string(), false)]);
    let confirmation_requests = outcome.confirmation_requests();
    assert_eq!(confirmation_requests.len(), 1);
    assert_eq!(
        confirmation_request_of(confirmation_requests[0]),
        Some((
            "a".to_string(),
            "a".to_string(),
            Some("careful".to_string())
        ))
    );
    assert!(confirmation_requests[0].is_user_visible());
    assert!(!confirmation_requests[0].is_agent_visible());
    assert!(confirmation_requests[0].id.is_some());
    assert!(outcome.responses().is_empty());
    assert!(outcome.shown.is_empty());
    Ok(())
}

#[tokio::test]
async fn handles_a_mixed_batch_in_one_pass() -> Result<()> {
    let hook = FakeHook::new(&[
        ("a", Reply::Allow),
        ("b", Reply::Deny("no")),
        ("c", Reply::Ask(None)),
    ]);
    let operation = PreToolExecutionOperation::new(hook.clone());

    let outcome = run(&operation, vec![kickoff(), calls(&["a", "b", "c"])]).await?;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    assert_eq!(
        outcome.marks,
        vec![("a".to_string(), true), ("c".to_string(), false),]
    );
    assert_eq!(
        outcome.responses(),
        vec![("b".to_string(), "no".to_string())]
    );
    assert_eq!(outcome.confirmation_requests().len(), 1);
    assert_eq!(hook.checked(), vec!["a", "b", "c"]);
    Ok(())
}

#[tokio::test]
async fn never_pauses_while_a_call_is_waiting() -> Result<()> {
    let hook = FakeHook::new(&[]);
    let operation = PreToolExecutionOperation::new(hook.clone());

    let outcome = run(
        &operation,
        vec![
            kickoff(),
            marked_call("c", false),
            confirmation_request("c"),
        ],
    )
    .await?;

    assert!(outcome.status.is_none());
    assert!(hook.checked().is_empty());
    Ok(())
}

#[tokio::test]
async fn a_user_allow_tells_the_hook_once_and_marks_the_call_true() -> Result<()> {
    let hook = FakeHook::new(&[]);
    let operation = PreToolExecutionOperation::new(hook.clone());
    let choice = click("c", Permission::AlwaysAllow);
    assert!(!choice.is_user_visible());
    assert!(!choice.is_agent_visible());
    assert!(choice.id.is_some());

    let outcome = run(
        &operation,
        vec![
            kickoff(),
            marked_call("c", false),
            confirmation_request("c"),
            choice,
        ],
    )
    .await?;

    assert_eq!(outcome.marks, vec![("c".to_string(), true)]);
    assert_eq!(
        hook.told(),
        vec![("c".to_string(), Permission::AlwaysAllow)]
    );
    assert!(hook.checked().is_empty());

    let outcome = run(
        &operation,
        vec![
            kickoff(),
            marked_call("c", true),
            confirmation_request("c"),
            click("c", Permission::AlwaysAllow),
        ],
    )
    .await?;

    assert!(outcome.status.is_none());
    assert_eq!(hook.told().len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_user_deny_responds_with_the_declined_message() -> Result<()> {
    let conversation = || {
        vec![
            kickoff(),
            marked_call("c", false),
            confirmation_request("c"),
            click("c", Permission::DenyOnce),
        ]
    };

    let hook = FakeHook::new(&[]);
    let outcome = run(
        &PreToolExecutionOperation::new(hook.clone()),
        conversation(),
    )
    .await?;
    assert!(outcome.marks.is_empty());
    assert_eq!(
        outcome.responses(),
        vec![("c".to_string(), DEFAULT_DECLINED_MESSAGE.to_string())]
    );
    assert_eq!(hook.told(), vec![("c".to_string(), Permission::DenyOnce)]);

    let custom = PreToolExecutionOperation::new(FakeHook::new(&[])).with_declined_message("nope");
    let outcome = run(&custom, conversation()).await?;
    assert_eq!(
        outcome.responses(),
        vec![("c".to_string(), "nope".to_string())]
    );
    Ok(())
}

#[tokio::test]
async fn skips_unreadable_external_and_responded_calls() -> Result<()> {
    let hook = FakeHook::new(&[]);
    let operation = PreToolExecutionOperation::new(hook.clone());

    let outcome = run(
        &operation,
        vec![
            kickoff(),
            Message::assistant()
                .with_tool_request(
                    "bad",
                    Err(ErrorData::invalid_params("unreadable arguments", None)),
                )
                .with_tool_request_with_metadata(
                    "external",
                    Ok(CallToolRequestParams::new("external")),
                    None,
                    Some(json!({ TOOL_META_EXTERNAL_DISPATCH_KEY: true })),
                ),
            calls(&["done"]),
            tool_response("done"),
        ],
    )
    .await?;

    assert!(outcome.status.is_none());
    assert!(hook.checked().is_empty());
    Ok(())
}

#[tokio::test]
async fn a_hook_error_ends_the_run() {
    let operation = PreToolExecutionOperation::new(FakeHook::new(&[("a", Reply::Fail)]));

    let error = run(&operation, vec![kickoff(), calls(&["a"])])
        .await
        .err()
        .unwrap();

    assert_eq!(error.to_string(), "hook crashed");
}
