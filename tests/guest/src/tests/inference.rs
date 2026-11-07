use crate::cosmonic::llama_cpp::api::{
    ChatMessage, Context, ContextParams, LogitBias, Model, ModelParams, PoolingType, Sampler,
    chat_apply_template,
};
use crate::models;

pub async fn load(source: &models::HfModel) -> Result<Model, String> {
    let (data, size, errors) = models::stream(source)?;
    let (progress, model) = Model::create(data, size, Some(ModelParams { n_gpu_layers: 99 })).await;
    // Dropped unread, so the provider loads with no one reading its progress.
    drop(progress);
    model.await.map_err(|e| errors.explain(e))
}

pub fn chain(samplers: Vec<Sampler>) -> Result<Sampler, String> {
    let chain = Sampler::chain_init();
    for sampler in samplers {
        chain.chain_add(sampler)?;
    }
    Ok(chain)
}

pub fn greedy() -> Result<Sampler, String> {
    chain(vec![Sampler::init_greedy()])
}

pub async fn decode(context: &Context, tokens: Vec<u32>) -> Result<(), String> {
    context
        .decode(tokens)
        .await
        .map_err(|code| format!("decode failed: {code}"))
}

pub async fn prompt(model: &Model, context: &Context, text: &str) -> Result<(), String> {
    decode(context, model.tokenize(text, true, true)?).await
}

pub async fn generate(
    model: &Model,
    context: &Context,
    sampler: &Sampler,
    limit: u32,
) -> Result<Vec<u32>, String> {
    let mut tokens = Vec::new();
    for _ in 0..limit {
        let token = sampler.sample(context, -1).await?;
        if model.is_eog(token) {
            break;
        }
        tokens.push(token);
        decode(context, vec![token]).await?;
    }
    Ok(tokens)
}

fn pieces(model: &Model, tokens: &[u32]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for &token in tokens {
        bytes.extend(model.token_to_piece(token, 0, false)?);
    }
    Ok(bytes)
}

pub async fn token_bytes() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    for text in ["Once upon a time", "你好世界 🦀🎉 café"] {
        let tokens = model.tokenize(text, false, true)?;
        let bytes = pieces(&model, &tokens)?;
        let decoded = String::from_utf8(bytes).map_err(|e| e.to_string())?;
        if decoded.trim_start() != text {
            return Err(format!("token round trip: {decoded:?} != {text:?}"));
        }
        // llama_detokenize drops the space SentencePiece puts before the first word
        if model.detokenize(&tokens, false, false)? != text {
            return Err("detokenize changed the complete text".into());
        }
        let first = model.token_to_piece(tokens[0], 1, false)?;
        if first.first() == Some(&b' ') {
            return Err("lstrip kept the leading space".into());
        }
    }
    if model.token_to_piece(u32::MAX, 0, false).is_ok()
        || model.detokenize(&[u32::MAX], false, false).is_ok()
        || model.is_eog(u32::MAX)
    {
        return Err("invalid token id was accepted".into());
    }
    Ok(())
}

pub async fn chat_template() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    if model.chat_template(None).is_some() {
        return Err("stories260K has no chat template".into());
    }
    let chat = [ChatMessage {
        role: "user".into(),
        content: "Hi".into(),
    }];
    let prompt = chat_apply_template("chatml", &chat, true)?;
    if prompt != "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" {
        return Err(format!("unexpected chatml prompt: {prompt:?}"));
    }
    if chat_apply_template("not a template", &chat, true).is_ok() {
        return Err("unknown template was accepted".into());
    }
    Ok(())
}

pub async fn model_lifetime() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let context = Context::create(&model, None).await?;
    let sampler = chain(vec![
        Sampler::init_grammar(&model, r#"root ::= "a""#, "root")?,
        Sampler::init_greedy(),
    ])?;
    let tokens = model.tokenize("Once upon a time", true, true)?;
    drop(model);
    decode(&context, tokens).await?;
    sampler.sample(&context, -1).await?;

    let other = load(&models::STORIES_260K).await?;
    let other_context = Context::create(&other, None).await?;
    prompt(&other, &other_context, "Once upon a time").await?;
    if sampler.sample(&other_context, -1).await.is_ok() {
        return Err("sampled a context of another model".into());
    }
    if sampler
        .chain_add(Sampler::init_penalties(&other, 64, 1.1, 0.0, 0.0))
        .is_ok()
    {
        return Err("chained samplers of two models".into());
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
            n_ubatch: 0,
            embeddings: false,
        }),
    )
    .await?;
    if context.n_ctx() != 256 || context.n_batch() != 256 || context.n_ubatch() != 256 {
        return Err("n-batch was not capped at n-ctx".into());
    }
    let sampler = greedy()?;
    if sampler.sample(&context, -1).await.is_ok() {
        return Err("sampled before any decode".into());
    }
    if Sampler::init_greedy().chain_add(Sampler::init_greedy()).is_ok() {
        return Err("added to a sampler that is not a chain".into());
    }
    let token = *model.tokenize("a", false, true)?.last().ok_or("no token")?;
    decode(&context, vec![token; 4]).await?;
    for picks_nothing in [Sampler::chain_init(), chain(vec![Sampler::init_top_k(1)])?] {
        if picks_nothing.sample(&context, -1).await.is_ok() {
            return Err("a chain without greedy or dist sampled".into());
        }
    }
    for (tokens, code) in [(vec![], -1), (vec![u32::MAX], -1), (vec![token; 257], -1), (vec![token; 256], 1)] {
        if context.decode(tokens).await != Err(code) || context.memory_seq_pos_max(0) != 3 {
            return Err(format!("a decode that should fail with {code} changed the context"));
        }
    }
    if !context.memory_seq_rm(0, 2, -1) || context.memory_seq_pos_max(0) != 1 {
        return Err("memory-seq-rm did not keep the prefix".into());
    }
    context.memory_clear(true).await;
    if context.memory_seq_pos_min(0) != -1 || context.memory_seq_pos_max(0) != -1 {
        return Err("memory-clear left tokens".into());
    }
    // llama.cpp asserts on these sequence ids
    if context.memory_seq_rm(1, 0, -1) || context.memory_seq_pos_min(-1) != -1 || context.memory_seq_pos_max(1) != -1 {
        return Err("an invalid sequence id was accepted".into());
    }
    if !context.memory_seq_rm(-2, 0, -1) {
        return Err("a negative sequence id did not match every sequence".into());
    }
    Ok(())
}

pub async fn sampling() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    let token = *model.tokenize("0", false, true)?.last().ok_or("no token")?;
    for bias in [100.0, -100.0] {
        let context = Context::create(&model, None).await?;
        prompt(&model, &context, "Once upon a time").await?;
        let sampler = chain(vec![
            Sampler::init_logit_bias(&model, &[LogitBias { token, bias }]),
            Sampler::init_greedy(),
        ])?;
        for _ in 0..4 {
            let sampled = sampler.sample(&context, -1).await?;
            if (sampled == token) != (bias > 0.0) {
                return Err(format!("logit bias {bias} did not apply"));
            }
            decode(&context, vec![sampled]).await?;
        }
    }
    Ok(())
}

pub async fn grammar() -> Result<(), String> {
    let model = load(&models::STORIES_260K).await?;
    if Sampler::init_grammar(&model, "this is not a grammar", "root").is_ok() {
        return Err("invalid grammar was accepted".into());
    }
    let grammar = r#"answer ::= "{\"answer\":\"Paris\"}""#;
    let context = Context::create(&model, None).await?;
    prompt(&model, &context, "Tell me a story.").await?;
    // top-k first leaves only a token the grammar then rules out
    let late = chain(vec![
        Sampler::init_top_k(1),
        Sampler::init_grammar(&model, grammar, "answer")?,
        Sampler::init_greedy(),
    ])?;
    if late.sample(&context, -1).await.is_ok() {
        return Err("sampled a token the grammar ruled out".into());
    }
    enforces(&model, r#"answer ::= "{\"answer\":\"Paris\"}""#, "answer").await
}

#[cfg(feature = "json-schema")]
pub async fn json_schema() -> Result<(), String> {
    use crate::cosmonic::llama_cpp::json_schema;

    if json_schema::to_grammar("{not json").is_ok()
        || json_schema::to_grammar(r#"{"type":"not-a-type"}"#).is_ok()
        // The parse error quotes a cut-off UTF-8 sequence
        || json_schema::to_grammar("é").is_ok()
    {
        return Err("invalid schema was accepted".into());
    }
    let model = load(&models::STORIES_260K).await?;
    let grammar = json_schema::to_grammar(
        r#"{"type":"object","properties":{"answer":{"const":"Paris"}},"required":["answer"],"additionalProperties":false}"#,
    )?;
    enforces(&model, &grammar, "root").await
}

// Expects `grammar`, which allows only {"answer":"Paris"}, to hold a story prompt to it.
async fn enforces(model: &Model, grammar: &str, root: &str) -> Result<(), String> {
    let context = Context::create(model, None).await?;
    prompt(model, &context, "Tell me a story.").await?;
    let sampler = chain(vec![
        Sampler::init_grammar(model, grammar, root)?,
        Sampler::init_greedy(),
    ])?;
    let tokens = generate(model, &context, &sampler, 64).await?;
    let value: serde_json::Value =
        serde_json::from_slice(&pieces(model, &tokens)?).map_err(|e| e.to_string())?;
    if value != serde_json::json!({"answer":"Paris"}) {
        return Err(format!("grammar was not enforced: {value}"));
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
            n_ubatch: 256,
            embeddings: true,
        }),
    )
    .await?;
    // A generative model: no pooling, one vector per token
    if context.pooling_type() != PoolingType::None {
        return Err(format!("unexpected pooling: {:?}", context.pooling_type()));
    }
    let tokens = model.tokenize("Once upon a time", true, true)?;
    let n = tokens.len() as i32;
    decode(&context, tokens).await?;
    if context.get_embeddings_seq(0).await.is_some() {
        return Err("pooled embedding without pooling".into());
    }
    for i in [0, n - 1, -1] {
        let vector = context
            .get_embeddings_ith(i)
            .await
            .ok_or(format!("no embedding for token {i}"))?;
        if vector.len() != model.n_embd_out() as usize
            || !vector.iter().all(|x| x.is_finite())
            || !vector.iter().any(|&x| x != 0.0)
        {
            return Err("invalid embedding vector".into());
        }
    }
    if context.get_embeddings_ith(n).await.is_some() {
        return Err("embedding past the last token".into());
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
        let tokens = model.tokenize(prompt, true, true)?;
        let mut shared = cached
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(tokens.len().saturating_sub(1));
        if !context.memory_seq_rm(0, shared as i32, -1) {
            context.memory_clear(true).await;
            shared = 0;
        }
        decode(&context, tokens[shared..].to_vec()).await?;
        let reused = generate(&model, &context, &greedy()?, 16).await?;
        cached = tokens.clone();
        cached.extend(&reused);
        let fresh = Context::create(&model, None).await?;
        decode(&fresh, tokens).await?;
        let baseline = generate(&model, &fresh, &greedy()?, 16).await?;
        if reused != baseline {
            return Err(format!(
                "prompt reuse differed from fresh inference in round {round}: {reused:?} != {baseline:?}"
            ));
        }
    }
    Ok(())
}
