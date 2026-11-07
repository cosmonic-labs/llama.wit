use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

mod inference;
mod large;
mod live;
mod smoke;

type TestFn = fn() -> Pin<Box<dyn Future<Output = Result<(), String>>>>;

static TESTS: &[(&str, TestFn)] = &[
    ("smoke", || Box::pin(smoke::run())),
    ("live/model-swaps", || Box::pin(live::model_swaps())),
    ("live/embeddings", || Box::pin(live::embeddings())),
    ("live/prompt-reuse", || Box::pin(live::prompt_reuse())),
    ("large/qwen3-8b", || Box::pin(large::qwen3_8b())),
    ("large/qwen3-8b-stress", || Box::pin(large::qwen3_8b_stress())),
    ("large/mistral-24b-q2", || Box::pin(large::mistral_24b_q2())),
    ("large/mistral-24b-q3", || Box::pin(large::mistral_24b_q3())),
    ("large/mistral-24b-q3m", || Box::pin(large::mistral_24b_q3m())),
    ("large/mistral-24b-q4", || Box::pin(large::mistral_24b_q4())),
    ("large/mistral-24b-q8", || Box::pin(large::mistral_24b_q8())),
    ("token-bytes", || Box::pin(inference::token_bytes())),
    ("chat-template", || Box::pin(inference::chat_template())),
    ("model-lifetime", || Box::pin(inference::model_lifetime())),
    ("context-limits", || Box::pin(inference::context_limits())),
    ("sampling", || Box::pin(inference::sampling())),
    ("grammar", || Box::pin(inference::grammar())),
    #[cfg(feature = "json-schema")]
    ("json-schema", || Box::pin(inference::json_schema())),
    ("embeddings", || Box::pin(inference::embeddings())),
    ("prompt-reuse", || Box::pin(inference::prompt_reuse())),
];

pub static ALL: LazyLock<BTreeMap<&'static str, TestFn>> =
    LazyLock::new(|| TESTS.iter().copied().collect());
