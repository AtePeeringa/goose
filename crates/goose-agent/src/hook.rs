use goose_provider_types::conversation::message::Message;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum HookDecision {
    Allow,
    Deny {
        reason: String,
        user_message: Option<Message>,
    },
}
