//! Load a model, feed it a prompt, generate. If this fails nothing else matters.

use crate::cosmonic::llama_cpp::api::{Context, Model, ModelParams, Sampler};
use crate::models;

const PROMPT: &str = "Once upon a time";
const N_GPU_LAYERS: i32 = 99;

pub async fn run() -> Result<(), String> {
    let bytes = models::load(&models::STORIES_260K)?;
    let model = Model::create(
        bytes,
        Some(ModelParams {
            n_gpu_layers: N_GPU_LAYERS,
        }),
    )
    .await?;

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
