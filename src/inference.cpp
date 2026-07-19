// Implements the cosmonic:llama/inference world (see wit/llama.wit) on top of llama.cpp.
#include "provider_cpp.h"
#include "llama.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace api = exports::cosmonic::llama_cpp::api;

static std::unexpected<wit::string> fail(const std::string & msg) {
    return std::unexpected(wit::string::from_view(msg));
}

// --- model ---

api::Model::Model(::llama_model * m) : model(m), vocab(llama_model_get_vocab(m)) {}

api::Model::~Model() {
    llama_model_free(model);
}

std::expected<api::Model::Owned, wit::string>
api::Model::Create(wit::string path, std::optional<api::ModelParams> params) {
    static bool backends_loaded = false;
    if (!backends_loaded) {
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); }
        }, nullptr);
        ggml_backend_load_all();
        backends_loaded = true;
    }

    llama_model_params mp = llama_model_default_params();
    if (params) {
        mp.n_gpu_layers = params->n_gpu_layers;
    }
    ::llama_model * m = llama_model_load_from_file(std::string(path.data(), path.size()).c_str(), mp);
    if (!m) {
        return fail("failed to load model");
    }
    return Owned(new Model(m));
}

std::expected<wit::vector<uint32_t>, wit::string>
api::Model::Tokenize(wit::string text, bool add_special) const {
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, true);
    auto out = wit::vector<uint32_t>::allocate(n);
    if (llama_tokenize(vocab, text.data(), text.size(), (llama_token *) out.data(), n, add_special, true) < 0) {
        return fail("failed to tokenize");
    }
    return out;
}

std::expected<wit::string, wit::string>
api::Model::Detokenize(wit::vector<uint32_t> tokens) const {
    std::string buf(tokens.size() * 8 + 16, '\0');
    int n = llama_detokenize(vocab, (const llama_token *) tokens.data(), tokens.size(),
                             buf.data(), buf.size(), false, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_detokenize(vocab, (const llama_token *) tokens.data(), tokens.size(),
                             buf.data(), buf.size(), false, true);
    }
    if (n < 0) {
        return fail("failed to detokenize");
    }
    return wit::string::from_view(std::string_view(buf.data(), n));
}

std::expected<wit::string, wit::string>
api::Model::ApplyChatTemplate(wit::vector<api::ChatMessage> messages, bool add_assistant) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        return fail("model has no chat template");
    }
    // keep null-terminated copies alive for llama_chat_message
    std::vector<std::string> strs;
    std::vector<llama_chat_message> msgs;
    for (size_t i = 0; i < messages.size(); i++) {
        strs.emplace_back(messages[i].role.data(), messages[i].role.size());
        strs.emplace_back(messages[i].content.data(), messages[i].content.size());
    }
    for (size_t i = 0; i < messages.size(); i++) {
        msgs.push_back({ strs[i * 2].c_str(), strs[i * 2 + 1].c_str() });
    }
    std::string buf(1024, '\0');
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, buf.data(), buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, buf.data(), buf.size());
    }
    if (n < 0) {
        return fail("failed to apply chat template");
    }
    return wit::string::from_view(std::string_view(buf.data(), n));
}

wit::string api::Model::Description() {
    char buf[256];
    llama_model_desc(model, buf, sizeof(buf));
    return wit::string::from_view(buf);
}

bool api::Model::IsEog(uint32_t token) const {
    return llama_vocab_is_eog(vocab, token);
}

uint32_t api::Model::NCtxTrain() {
    return llama_model_n_ctx_train(model);
}

// --- context ---

api::Context::Context(::llama_context * c, const Model * m, uint32_t n_batch)
    : ctx(c), model(m), batch_size(n_batch) {}

api::Context::~Context() {
    llama_free(ctx);
}

std::expected<api::Context::Owned, wit::string>
api::Context::Create(std::reference_wrapper<const api::Model> model, std::optional<api::ContextParams> params) {
    llama_context_params cp = llama_context_default_params();
    // Treat 0 as "unset": the WIT records these as plain u32 (not option) but documents
    // per-field defaults, and n_batch == 0 would make append-tokens loop forever.
    cp.n_ctx   = (params && params->n_ctx)   ? params->n_ctx   : 4096;
    cp.n_batch = (params && params->n_batch) ? params->n_batch : 512;
    ::llama_context * c = llama_init_from_model(model.get().model, cp);
    if (!c) {
        return fail("failed to create context");
    }
    return Owned(new Context(c, &model.get(), cp.n_batch));
}

std::expected<void, wit::string>
api::Context::Append(wit::string text) {
    auto tokens = model->Tokenize(std::move(text), past == 0);
    if (!tokens) {
        return std::unexpected(std::move(tokens).error());
    }
    return AppendTokens(std::move(*tokens));
}

std::expected<void, wit::string>
api::Context::AppendTokens(wit::vector<uint32_t> tokens) {
    for (size_t i = 0; i < tokens.size(); i += batch_size) {
        size_t n = std::min((size_t) batch_size, tokens.size() - i);
        llama_batch batch = llama_batch_get_one((llama_token *) tokens.data() + i, n);
        if (llama_decode(ctx, batch)) {
            return fail("decode failed");
        }
        past += n;
    }
    return {};
}

uint32_t api::Context::NPast() {
    return past;
}

void api::Context::Clear() {
    llama_memory_clear(llama_get_memory(ctx), true);
    past = 0;
}

// --- sampler ---

api::Sampler::Sampler(std::optional<api::SamplerParams> params) {
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!params || params->temp <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (params->top_k > 0) { llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->top_k)); }
        if (params->top_p < 1) { llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params->top_p, 1)); }
        if (params->min_p > 0) { llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params->min_p, 1)); }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params->seed));
    }
}

api::Sampler::~Sampler() {
    llama_sampler_free(smpl);
}

uint32_t api::Sampler::Sample(std::reference_wrapper<const api::Context> ctx) {
    // No error channel (returns u32): an empty context has no logits at index -1,
    // so fail loudly instead of sampling garbage. Caller must append a prompt first.
    if (ctx.get().past == 0) {
        throw std::runtime_error("sampler.sample: context is empty (append a prompt first)");
    }
    return llama_sampler_sample(smpl, ctx.get().ctx, -1);
}
