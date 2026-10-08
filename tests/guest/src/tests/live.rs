use super::inference;
use crate::cosmonic::llama_cpp::api::{Context, ContextParams, Sampler};
use crate::models;

pub async fn model_swaps() -> Result<(), String> {
    for round in 0..8 {
        let source = if round % 2 == 0 {
            &models::QWEN
        } else {
            &models::LLAMA
        };
        let model = inference::load(source).await?;
        let context = Context::create(&model, None).await?;
        context.append("Hello, my name is".into()).await?;
        let sampler = Sampler::create(&model, None)?;
        if inference::generate(&model, &context, &sampler, 4)
            .await?
            .is_empty()
        {
            return Err(format!("model swap {round} produced no tokens"));
        }
    }
    Ok(())
}

pub async fn prompt_reuse() -> Result<(), String> {
    inference::reuse(&models::QWEN, 40).await
}

pub async fn embeddings() -> Result<(), String> {
    let model = inference::load(&models::BGE).await?;
    let context = Context::create(
        &model,
        Some(ContextParams {
            n_ctx: 512,
            n_batch: 512,
            embeddings: true,
        }),
    )
    .await?;
    let mut vectors = Vec::new();
    for input in [
        "A dog is playing in the park.",
        "A puppy runs outdoors.",
        "The database transaction was rolled back.",
    ] {
        let vector = context.embed(model.tokenize(input, true)?).await?;
        if vector.len() != 384 || !vector.iter().all(|x| x.is_finite()) {
            return Err("invalid BGE vector".into());
        }
        vectors.push(vector);
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
        let norm_a: f32 = a.iter().map(|a| a * a).sum();
        let norm_b: f32 = b.iter().map(|b| b * b).sum();
        dot / (norm_a * norm_b).sqrt()
    }
    let related = cosine(&vectors[0], &vectors[1]);
    let unrelated = cosine(&vectors[0], &vectors[2]);
    if !related.is_finite() || !unrelated.is_finite() || related <= unrelated {
        return Err("related sentences were not closer".into());
    }
    Ok(())
}
