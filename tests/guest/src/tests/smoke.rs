//! Load a model, feed it a prompt, generate. If this fails nothing else matters.

use crate::cosmonic::llama_cpp::api::{Context, Model, ModelParams, Sampler};
use crate::models;

const PROMPT: &str = "Once upon a time";
const N_GPU_LAYERS: u32 = 99;

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
    context.append(PROMPT.to_string()).await?;

    let sampler = Sampler::new(None);

    let mut generated = String::new();
    const MAX_NEW_TOKENS: u32 = 16;
    for _ in 0..MAX_NEW_TOKENS {
        let token = sampler.sample(&context).await;
        if model.is_eog(token) {
            break;
        }
        generated.push_str(&model.detokenize(&[token])?);
        context.append_tokens(vec![token]).await?;
    }

    if generated.is_empty() {
        return Err("generated no tokens".to_string());
    }
    println!("smoke: {PROMPT}{generated}");
    Ok(())
}
