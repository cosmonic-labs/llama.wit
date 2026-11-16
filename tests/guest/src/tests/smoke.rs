//! Load a model, feed it a prompt, generate. If this fails nothing else matters.

use crate::cosmonic::llama_cpp::api::{Context, Sampler};
use crate::models;

const PROMPT: &str = "Once upon a time";

pub async fn run() -> Result<(), String> {
    let model = super::inference::load(&models::STORIES_260K).await?;

    let context = Context::create(&model, None).await?;
    let prompt = model.tokenize(PROMPT, true, true)?;
    context
        .decode(prompt)
        .await
        .map_err(|code| format!("decode failed: {code}"))?;

    let sampler = Sampler::chain_init();
    sampler.chain_add(Sampler::init_greedy())?;

    let mut generated = Vec::new();
    const MAX_NEW_TOKENS: u32 = 16;
    for _ in 0..MAX_NEW_TOKENS {
        let token = sampler.sample(&context, -1).await?;
        if model.is_eog(token) {
            break;
        }
        generated.extend(model.token_to_piece(token, 0, false)?);
        context
            .decode(vec![token])
            .await
            .map_err(|code| format!("decode failed: {code}"))?;
    }

    if generated.is_empty() {
        return Err("generated no tokens".to_string());
    }
    let generated = String::from_utf8_lossy(&generated);
    println!("smoke: {PROMPT}{generated}");
    Ok(())
}
