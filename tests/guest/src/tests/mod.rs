use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

mod inference;
mod live;
mod smoke;

type TestFn = fn() -> Pin<Box<dyn Future<Output = Result<(), String>>>>;

static TESTS: &[(&str, TestFn)] = &[
    ("smoke", || Box::pin(smoke::run())),
    ("live/model-swaps", || Box::pin(live::model_swaps())),
    ("live/embeddings", || Box::pin(live::embeddings())),
    ("live/prompt-reuse", || Box::pin(live::prompt_reuse())),
    ("token-bytes", || Box::pin(inference::token_bytes())),
    ("context-limits", || Box::pin(inference::context_limits())),
    ("sampling", || Box::pin(inference::sampling())),
    ("grammar", || Box::pin(inference::grammar())),
    ("embeddings", || Box::pin(inference::embeddings())),
    ("prompt-reuse", || Box::pin(inference::prompt_reuse())),
];

pub static ALL: LazyLock<BTreeMap<&'static str, TestFn>> =
    LazyLock::new(|| TESTS.iter().copied().collect());
