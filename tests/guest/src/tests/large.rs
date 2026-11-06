//! Models far bigger than the provider's 4 GiB of linear memory.
//!
//! These are the tests the streaming `model.create` exists for: the weights
//! reach the GPU a chunk at a time, so neither side ever holds the whole file.
//! Each needs its GGUF in `tests/guest-dir/models/` and enough VRAM for the
//! weights; without the file it skips rather than fails.

use super::inference;
use crate::cosmonic::llama_cpp::api::{
    ChatMessage, Context, ContextParams, Model, chat_apply_template,
};
use crate::models::{self, HfModel};

const PROMPT: &str = "In one sentence, what is a vector database?";
const MAX_NEW_TOKENS: u32 = 32;

/// Small on purpose: on a model this size the weights already crowd the GPU,
/// and the KV cache is charged on top of them.
const N_CTX: u32 = 1024;
const N_BATCH: u32 = 256;

/// Past the address-space limit, but with room to spare on the GPU. Tells a
/// streaming bug apart from simply running out of memory.
pub async fn qwen3_8b() -> Result<(), String> {
    run(&models::QWEN3_8B, "qwen3-8b", N_CTX).await
}

/// The same known-good model, but with a context big enough that its KV cache
/// alone pushes GPU memory up to what a 24B at Q8_0 needs. Weight volume and
/// total GPU footprint are separate variables and this moves only the second,
/// so a failure here is memory and a pass here is not.
///
/// Qwen3-8B spends ~144 KiB per token of context (36 layers x 8 KV heads x 128
/// x 2 for K and V, at f16), so the cache dominates the 4.2 GiB of weights well
/// before this. Positions stay small for a short prompt, so running past the
/// trained context length does not distort what comes out.
pub async fn qwen3_8b_stress() -> Result<(), String> {
    for n_ctx in [65536u32, 131072] {
        let kv_gib = (n_ctx as f64 * 147456.0) / (1u64 << 30) as f64;
        println!("qwen3-8b-stress: n_ctx {n_ctx} -> ~{kv_gib:.1} GiB of KV cache on top of the weights");
        run(&models::QWEN3_8B, "qwen3-8b-stress", n_ctx).await?;
    }
    Ok(())
}

/// Past the address-space limit *and* close to the memory of a 32 GiB machine.
pub async fn mistral_24b_q4() -> Result<(), String> {
    run(&models::MISTRAL_SMALL_24B_Q4, "mistral-24b-q4", N_CTX).await
}

pub async fn mistral_24b_q2() -> Result<(), String> {
    run(&models::MISTRAL_SMALL_24B_Q2, "mistral-24b-q2", N_CTX).await
}

pub async fn mistral_24b_q3() -> Result<(), String> {
    run(&models::MISTRAL_SMALL_24B_Q3, "mistral-24b-q3", N_CTX).await
}

pub async fn mistral_24b_q3m() -> Result<(), String> {
    run(&models::MISTRAL_SMALL_24B_Q3M, "mistral-24b-q3m", N_CTX).await
}

/// The same model at Q8_0: twice the weight, and past the point where the
/// weights stop arriving intact. Kept as its own test because the footprint,
/// not the architecture, is what distinguishes it from `mistral-24b-q4`.
pub async fn mistral_24b_q8() -> Result<(), String> {
    run(&models::MISTRAL_SMALL_24B, "mistral-24b-q8", N_CTX).await
}

async fn run(spec: &HfModel, label: &str, n_ctx: u32) -> Result<(), String> {
    if !spec.is_cached() {
        println!("{label}: skipped, {} is not in /models", spec.name());
        return Ok(());
    }

    println!("{label}: streaming {}", spec.name());
    let model = inference::load(spec).await?;
    println!("{label}: loaded {}", model.description());

    let context = Context::create(
        &model,
        Some(ContextParams {
            n_ctx,
            n_batch: N_BATCH,
            n_ubatch: 0,
            embeddings: false,
        }),
    )
    .await?;

    // Instruct models, so go through their own chat template rather than
    // feeding them a raw continuation.
    let template = model
        .chat_template(None)
        .ok_or_else(|| format!("{} has no chat template", spec.name()))?;
    let prompt = chat_apply_template(
        &template,
        &[ChatMessage {
            role: "user".to_string(),
            content: PROMPT.to_string(),
        }],
        true,
    )?;
    for chunk in model.tokenize(&prompt, true, true)?.chunks(N_BATCH as usize) {
        inference::decode(&context, chunk.to_vec()).await?;
    }

    let generated = generate(&model, &context).await?;
    println!("{label}: {PROMPT}\n{label}: {generated}");
    Ok(())
}

async fn generate(model: &Model, context: &Context) -> Result<String, String> {
    let tokens =
        inference::generate(model, context, &inference::greedy()?, MAX_NEW_TOKENS).await?;
    if tokens.is_empty() {
        return Err("generated no tokens".to_string());
    }
    let mut bytes = Vec::new();
    for &token in &tokens {
        bytes.extend(model.token_to_piece(token, 0, false)?);
    }
    let generated = String::from_utf8_lossy(&bytes).into_owned();
    // Weights that never made it onto the device read back as zeros or NaN, so
    // every logit ties and greedy sampling picks the same token every step --
    // usually token 0, the unknown token. The model still loads and generation
    // still "works", so without this the run passes while emitting gibberish.
    if tokens.len() > 1 && tokens.iter().all(|t| *t == tokens[0]) {
        return Err(format!(
            "degenerate output: token {} repeated {} times ({:?}) -- logits carry no signal, \
             which usually means the weights did not survive the trip to the device",
            tokens[0],
            tokens.len(),
            generated.chars().take(40).collect::<String>(),
        ));
    }
    Ok(generated)
}
