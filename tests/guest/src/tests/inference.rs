use crate::cosmonic::llama_cpp::api::{
    self, Context, ContextParams, LogitBias, Model, ModelParams, Sampler, SamplerParams,
};
use crate::models;

pub async fn load(source: &models::HfModel) -> Result<Model, String> {
    Model::create(
        models::load(source)?,
        Some(ModelParams { n_gpu_layers: 99 }),
    )
    .await
}

fn params() -> SamplerParams {
    SamplerParams {
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
        penalty_last_n: 64,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        logit_bias: Vec::new(),
        grammar: None,
    }
}

pub async fn generate(
    model: &Model,
    context: &Context,
    sampler: &Sampler,
    limit: u32,
) -> Result<Vec<u32>, String> {
    let mut tokens = Vec::new();
    for _ in 0..limit {
        let token = sampler.sample(context).await;
        if model.is_eog(token) {
            break;
        }
        tokens.push(token);
        context.append_tokens(vec![token]).await?;
    }
    Ok(tokens)
}

fn pieces(model: &Model, tokens: &[u32]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for &token in tokens {
        bytes.extend(model.token_to_piece(token, false)?);
    }
    Ok(bytes)
}

pub async fn token_bytes() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    for text in ["Once upon a time", "你好世界 🦀🎉 café"] {
        let tokens = model.tokenize(text, false)?;
        let bytes = pieces(&model, &tokens)?;
        let decoded = String::from_utf8(bytes).map_err(|e| e.to_string())?;
        if decoded.trim_start() != text {
            return Err(format!("token round trip: {decoded:?} != {text:?}"));
        }
        if model.detokenize(&tokens)? != decoded {
            return Err("detokenize changed the complete text".into());
        }
        let mut streamed = String::new();
        for token in tokens {
            streamed.push_str(&model.detokenize(&[token])?);
        }
        if text.is_ascii() && streamed != decoded {
            return Err("per-token detokenize stripped spaces".into());
        }
    }
    if model.token_to_piece(u32::MAX, false).is_ok()
        || model.detokenize(&[u32::MAX]).is_ok()
        || model.is_eog(u32::MAX)
    {
        return Err("invalid token id was accepted".into());
    }
    Ok(())
}

pub async fn context_limits() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let context = Context::create(
        &model,
        Some(ContextParams {
            n_ctx: 256,
            n_batch: 1024,
            embeddings: false,
        }),
    )
    .await?;
    let token = *model.tokenize("a", false)?.last().ok_or("no token")?;
    context.append_tokens(vec![token; 4]).await?;
    if context.append_tokens(vec![u32::MAX]).await.is_ok() || context.n_past() != 4 {
        return Err("invalid token changed the context".into());
    }
    if context.append_tokens(vec![token; 256]).await.is_ok() || context.n_past() != 4 {
        return Err("overflow changed the context".into());
    }
    context.truncate(2)?;
    if context.n_past() != 2 {
        return Err("truncate did not keep the prefix".into());
    }
    if context.truncate(999).is_ok() || context.n_past() != 0 {
        return Err("invalid truncate did not clear the context".into());
    }
    Ok(())
}

pub async fn sampling() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let token = *model.tokenize("0", false)?.last().ok_or("no token")?;
    for bias in [100.0, -100.0] {
        let context = Context::create(&model, None).await?;
        context.append("Once upon a time".into()).await?;
        let mut options = params();
        options.logit_bias = vec![LogitBias { token, bias }];
        let sampler = Sampler::create(&model, Some(&options))?;
        for _ in 0..4 {
            let sampled = sampler.sample(&context).await;
            if (sampled == token) != (bias > 0.0) {
                return Err(format!("logit bias {bias} did not apply"));
            }
            context.append_tokens(vec![sampled]).await?;
        }
    }
    Ok(())
}

pub async fn grammar() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let mut options = params();
    options.grammar = Some("this is not a grammar".into());
    if Sampler::create(&model, Some(&options)).is_ok() {
        return Err("invalid grammar was accepted".into());
    }
    if api::json_schema_to_grammar("{not json").is_ok()
        || api::json_schema_to_grammar(r#"{"type":"not-a-type"}"#).is_ok()
    {
        return Err("invalid schema was accepted".into());
    }
    options.grammar = Some(api::json_schema_to_grammar(
        r#"{"type":"object","properties":{"answer":{"const":"Paris"}},"required":["answer"],"additionalProperties":false}"#,
    )?);
    let context = Context::create(&model, None).await?;
    context.append("Tell me a story.".into()).await?;
    let sampler = Sampler::create(&model, Some(&options))?;
    let tokens = generate(&model, &context, &sampler, 64).await?;
    let value: serde_json::Value =
        serde_json::from_slice(&pieces(&model, &tokens)?).map_err(|e| e.to_string())?;
    if value != serde_json::json!({"answer":"Paris"}) {
        return Err(format!("schema was not enforced: {value}"));
    }
    Ok(())
}

pub async fn embeddings() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let context = Context::create(
        &model,
        Some(ContextParams {
            n_ctx: 256,
            n_batch: 256,
            embeddings: true,
        }),
    )
    .await?;
    let tokens = model.tokenize("Once upon a time", true)?;
    let vector = context.embed(tokens).await?;
    if vector.len() != model.n_embd() as usize
        || vector.is_empty()
        || !vector.iter().all(|x| x.is_finite())
        || !vector.iter().any(|&x| x != 0.0)
    {
        return Err("invalid embedding vector".into());
    }
    if context.n_past() != 0 {
        return Err("embedding left tokens in the context".into());
    }
    Ok(())
}

pub async fn prompt_reuse() -> Result<(), String> {
    reuse(&models::STORIES_260K, 40).await
}

pub async fn reuse(source: &models::HfModel, rounds: u32) -> Result<(), String> {
    let model = load(source).await?;
    let context = Context::create(&model, None).await?;
    let mut cached = Vec::new();
    for round in 0..rounds {
        let prompt = if round % 3 == 0 {
            "Once upon a time there was a little girl."
        } else {
            "Once upon a time there was a little boy."
        };
        let tokens = model.tokenize(prompt, true)?;
        let shared = cached
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(tokens.len().saturating_sub(1));
        context.truncate(shared as u32)?;
        context.append_tokens(tokens[shared..].to_vec()).await?;
        let sampler = Sampler::create(&model, None)?;
        let reused = generate(&model, &context, &sampler, 16).await?;
        cached = tokens.clone();
        cached.extend(&reused);
        let fresh = Context::create(&model, None).await?;
        fresh.append_tokens(tokens).await?;
        let sampler = Sampler::create(&model, None)?;
        let baseline = generate(&model, &fresh, &sampler, 16).await?;
        if reused != baseline {
            return Err(format!(
                "prompt reuse differed from fresh inference in round {round}: {reused:?} != {baseline:?}"
            ));
        }
    }
    Ok(())
}
