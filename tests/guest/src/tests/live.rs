use super::inference;
use crate::cosmonic::llama_cpp::api::{Context, ContextParams};
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
        inference::prompt(&model, &context, "Hello, my name is").await?;
        if inference::generate(&model, &context, &inference::greedy()?, 4)
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
            n_ubatch: 512,
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
        context.memory_clear(true);
        inference::decode(&context, model.tokenize(input, true, true)?).await?;
        let vector = context
            .get_embeddings_seq(0)
            .ok_or("BGE gave no pooled embedding")?;
        if vector.len() != 384 || !vector.iter().all(|x| x.is_finite()) {
            return Err("invalid BGE vector".into());
        }
        if context.get_embeddings_ith(0).is_some() {
            return Err("per-token embedding with pooling".into());
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
