use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

mod smoke;

type TestFn = fn() -> Pin<Box<dyn Future<Output = Result<(), String>>>>;

static TESTS: &[(&str, TestFn)] = &[("smoke", || Box::pin(smoke::run()))];

pub static ALL: LazyLock<BTreeMap<&'static str, TestFn>> =
    LazyLock::new(|| TESTS.iter().copied().collect());
