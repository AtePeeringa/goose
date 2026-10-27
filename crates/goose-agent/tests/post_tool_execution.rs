use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::{
    operation::{ConversationEffect, Emitter, Operation, OperationResult, RunStatus},
    post_tool_execution::{
        PostToolCallHook, PostToolExecutionOperation, NOT_HANDLED_MESSAGE, POST_HOOK_CALLED_KEY,
    },
    pre_tool_execution::EXECUTABLE_KEY,
};
use goose_provider_types::conversation::{
    message::{
        Message, MessageContent, ToolRequest, ToolResponse, TOOL_META_EXTERNAL_DISPATCH_KEY,
    },
    Conversation,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, ErrorData};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct FakeHook {
    fail: bool,
    called: Mutex<Vec<String>>,
}

impl FakeHook {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn failing() -> Arc<Self> {
        Arc::new(Self {
            fail: true,
            ..Self::default()
        })
    }

    fn called(&self) -> Vec<String> {
        self.called.lock().unwrap().clone()
    }
}

#[async_trait]
impl PostToolCallHook<()> for FakeHook {
    async fn after_call(
        &self,
        _session: &(),
        _conversation: &Conversation,
        request: &ToolRequest,
        response: &ToolResponse,
    ) -> Result<()> {
        assert_eq!(request.id, response.id);
        if self.fail {
            anyhow::bail!("hook failed");
        }
        self.called.lock().unwrap().push(response.id.clone());
        Ok(())
    }
}

#[derive(Default)]
struct Outcome {
    status: Option<RunStatus>,
    hook_called_marks: Vec<String>,
    executable_marks: Vec<(String, bool)>,
    responses: Vec<(String, String)>,
}

async fn run_with(
    operation: PostToolExecutionOperation<'_, ()>,
    messages: Vec<Message>,
) -> Result<Outcome> {
    let (tx, _rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, CancellationToken::new());
    let conversation = Conversation::new_unvalidated(messages);
    let result =
        Operation::<(), ConversationEffect>::run(&operation, &(), &conversation, &emit).await?;
    let OperationResult::Applied(result) = result else {
        return Ok(Outcome::default());
    };
    let mut outcome = Outcome {
        status: Some(result.status),
        ..Outcome::default()
    };
    for effect in result.effects {
        match effect {
            ConversationEffect::PatchToolRequestMeta {
                tool_call_id,
                patch,
            } => {
                if let Some(executable) = patch.get(EXECUTABLE_KEY) {
                    outcome
                        .executable_marks
                        .push((tool_call_id, executable.as_bool().unwrap()));
                } else {
                    assert_eq!(patch, json!({ POST_HOOK_CALLED_KEY: true }));
                    outcome.hook_called_marks.push(tool_call_id);
                }
            }
            ConversationEffect::AppendMessage(message) => {
                outcome
                    .responses
                    .extend(message.content.into_iter().filter_map(|content| {
                        let MessageContent::ToolResponse(response) = content else {
                            return None;
                        };
                        let result = response.tool_result.unwrap();
                        assert_eq!(result.is_error, Some(true));
                        Some((
                            response.id,
                            result.content[0].as_text().unwrap().text.clone(),
                        ))
                    }));
            }
            _ => panic!("unexpected effect"),
        }
    }
    Ok(outcome)
}

async fn run(hook: &Arc<FakeHook>, messages: Vec<Message>) -> Outcome {
    let operation = PostToolExecutionOperation::new().with_hook(hook.clone());
    run_with(operation, messages).await.unwrap()
}

async fn run_without_hook(messages: Vec<Message>) -> Outcome {
    run_with(PostToolExecutionOperation::new(), messages)
        .await
        .unwrap()
}

fn kickoff() -> Message {
    Message::user().with_text("hi")
}

fn call(id: &str, tool_meta: Option<Value>) -> Message {
    Message::assistant().with_tool_request_with_metadata(
        id,
        Ok(CallToolRequestParams::new(id.to_string())),
        None,
        tool_meta,
    )
}

fn marked_call(id: &str, executable: bool) -> Message {
    call(id, Some(json!({ EXECUTABLE_KEY: executable })))
}

fn hook_called_call(id: &str) -> Message {
    call(
        id,
        Some(json!({ EXECUTABLE_KEY: true, POST_HOOK_CALLED_KEY: true })),
    )
}

fn unreadable_call(id: &str) -> Message {
    Message::assistant().with_tool_request(id, Err(ErrorData::invalid_params("bad json", None)))
}

fn confirmation_request(id: &str) -> Message {
    Message::assistant()
        .with_action_required(id, id.to_string(), Default::default(), None)
        .user_only()
}

fn tool_responses(ids: &[&str]) -> Message {
    ids.iter().fold(Message::user(), |message, id| {
        message.with_tool_response(
            *id,
            Ok(CallToolResult::success(vec![ContentBlock::text("done")])),
        )
    })
}

fn not_handled(id: &str) -> (String, String) {
    (id.to_string(), NOT_HANDLED_MESSAGE.to_string())
}

#[tokio::test]
async fn calls_the_hook_for_each_result_in_call_order_and_marks_the_call() {
    let hook = FakeHook::new();
    let outcome = run(
        &hook,
        vec![
            kickoff(),
            call("a", None),
            marked_call("b", true),
            tool_responses(&["b", "a"]),
        ],
    )
    .await;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    assert_eq!(hook.called(), vec!["a", "b"]);
    assert_eq!(outcome.hook_called_marks, vec!["a", "b"]);
    assert!(outcome.responses.is_empty());
}

#[tokio::test]
async fn does_not_call_the_hook_twice() {
    let hook = FakeHook::new();
    let outcome = run(
        &hook,
        vec![kickoff(), hook_called_call("a"), tool_responses(&["a"])],
    )
    .await;

    assert!(outcome.status.is_none());
    assert!(hook.called().is_empty());
}

#[tokio::test]
async fn skips_calls_that_did_not_run() {
    let hook = FakeHook::new();
    let outcome = run(
        &hook,
        vec![
            kickoff(),
            marked_call("denied", false),
            unreadable_call("unreadable"),
            call(
                "external",
                Some(json!({ TOOL_META_EXTERNAL_DISPATCH_KEY: true })),
            ),
            tool_responses(&["denied", "unreadable", "external"]),
        ],
    )
    .await;

    assert!(outcome.status.is_none());
    assert!(hook.called().is_empty());
}

#[tokio::test]
async fn calls_the_hook_before_responding_to_unhandled_calls() {
    let hook = FakeHook::new();

    let outcome = run(
        &hook,
        vec![
            kickoff(),
            marked_call("a", true),
            tool_responses(&["a"]),
            marked_call("b", true),
        ],
    )
    .await;
    assert_eq!(outcome.hook_called_marks, vec!["a"]);
    assert!(outcome.responses.is_empty());

    let outcome = run(
        &hook,
        vec![
            kickoff(),
            hook_called_call("a"),
            tool_responses(&["a"]),
            marked_call("b", true),
        ],
    )
    .await;
    assert_eq!(outcome.responses, vec![not_handled("b")]);
    assert_eq!(hook.called(), vec!["a"]);
}

#[tokio::test]
async fn marks_not_handled_calls_false_so_the_hook_is_not_called() {
    let hook = FakeHook::new();
    let outcome = run(&hook, vec![kickoff(), marked_call("a", true)]).await;
    assert_eq!(outcome.executable_marks, vec![("a".to_string(), false)]);
    assert_eq!(outcome.responses, vec![not_handled("a")]);

    let outcome = run(
        &hook,
        vec![kickoff(), marked_call("a", false), tool_responses(&["a"])],
    )
    .await;
    assert!(outcome.status.is_none());
    assert!(hook.called().is_empty());
}

#[tokio::test]
async fn a_hook_error_ends_the_run() {
    let operation = PostToolExecutionOperation::new().with_hook(FakeHook::failing());
    let result = run_with(
        operation,
        vec![kickoff(), marked_call("a", true), tool_responses(&["a"])],
    )
    .await;

    assert_eq!(result.err().unwrap().to_string(), "hook failed");
}

#[tokio::test]
async fn pauses_when_every_open_call_is_waiting() {
    let outcome = run_without_hook(vec![
        kickoff(),
        marked_call("a", true),
        tool_responses(&["a"]),
        marked_call("c", false),
        confirmation_request("c"),
    ])
    .await;

    assert_eq!(outcome.status, Some(RunStatus::Yielded));
    assert!(outcome.responses.is_empty());
}

#[tokio::test]
async fn responds_to_open_calls_that_are_not_awaiting_confirmation() {
    let outcome = run_without_hook(vec![
        kickoff(),
        marked_call("allowed", true),
        call("unchecked", None),
        marked_call("declined", false),
        marked_call("waiting", false),
        confirmation_request("waiting"),
    ])
    .await;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    assert_eq!(
        outcome.responses,
        vec![
            not_handled("allowed"),
            not_handled("unchecked"),
            not_handled("declined"),
        ]
    );
    assert_eq!(
        outcome.executable_marks,
        vec![
            ("allowed".to_string(), false),
            ("unchecked".to_string(), false),
            ("declined".to_string(), false),
        ]
    );
}

#[tokio::test]
async fn does_nothing_when_no_call_is_open() {
    let outcome = run_without_hook(vec![
        kickoff(),
        marked_call("a", true),
        tool_responses(&["a"]),
        call(
            "external",
            Some(json!({ TOOL_META_EXTERNAL_DISPATCH_KEY: true })),
        ),
    ])
    .await;

    assert!(outcome.status.is_none());
}
