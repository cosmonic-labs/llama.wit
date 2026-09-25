use wit_bindgen::generate;

generate!({
    path: "../../wit/",
    world: "test-harness",
    generate_all,
});

mod models;
mod tests;

struct TestHarness;

impl Guest for TestHarness {
    fn list_tests() -> Vec<String> {
        tests::ALL.keys().map(|name| name.to_string()).collect()
    }

    async fn run_test(name: String) -> Result<(), String> {
        let test = tests::ALL
            .get(name.as_str())
            .ok_or_else(|| format!("no such test: {name}"))?;
        test().await
    }
}

export!(TestHarness);
