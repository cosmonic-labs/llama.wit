// Implements cosmonic:llama-cpp/api (see wit/llama.wit) on top of llama.cpp,
// against the wit-bindgen C bindings (provider.h). Pure C over llama.h's C API.
#include "provider.h"
#include "llama.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// wasi-sdk wasip3 workaround: _initialize runs constructors before the
// cooperative-threading stack pointer is set, so any ctor that touches the stack
// traps. A highest-priority constructor runs the stack init first. Weak so a
// wasip2 build still links. TODO: remove once wasi-sdk fixes reactor init.
extern void __wasm_init_task(void) __attribute__((weak));
static void ensure_stack(void) { if (__wasm_init_task) { __wasm_init_task(); } }
__attribute__((constructor(101)))
static void wasip3_init_stack(void) { ensure_stack(); }

// Short aliases for the (verbose) wit-bindgen resource-rep types.
typedef exports_cosmonic_llama_cpp_api_model_t   model_t;
typedef exports_cosmonic_llama_cpp_api_context_t context_t;
typedef exports_cosmonic_llama_cpp_api_sampler_t sampler_t;

// The C header forward-declares these; we define the reps.
struct exports_cosmonic_llama_cpp_api_model_t {
    struct llama_model *       model;
    const struct llama_vocab * vocab;
};
struct exports_cosmonic_llama_cpp_api_context_t {
    struct llama_context * ctx;
    const model_t *        model;    // borrowed; the model must outlive the context
    uint32_t               batch_size;
    uint32_t               past;
};
struct exports_cosmonic_llama_cpp_api_sampler_t {
    struct llama_sampler * smpl;
};

static void set_err(provider_string_t * err, const char * msg) {
    provider_string_dup(err, msg);
}

// No-op log sink. On wasip3, writing to stderr is an async WASI call that a sync
// export can't block on, so the provider must never log; the consumer owns I/O.
static void log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) level; (void) text; (void) user;
}

// Async exports use the callback ABI: the entry point does the work (blocking on
// the component-model waitable-set is legal here, which is what the wasi:webgpu
// glue needs), publishes the result with the generated `..._return`, and exits.
// The paired `..._callback` is therefore unreachable — we always finish inline
// rather than parking on a waitable set.
#define UNREACHABLE_CALLBACK(name)                                                  \
    provider_callback_code_t name(provider_event_t * event) {                       \
        (void) event;                                                               \
        abort();                                                                    \
    }

// --- model ---

static bool model_create(
        provider_list_u8_t * data,
        exports_cosmonic_llama_cpp_api_model_params_t * maybe_params,
        exports_cosmonic_llama_cpp_api_own_model_t * ret,
        provider_string_t * err) {
    // Route llama + ggml logging to a no-op sink so the provider never writes to
    // stderr (a WASI call). The statically-linked CPU backend registers itself, so
    // we do NOT call ggml_backend_load_all() (which scans the filesystem).
    static bool inited = false;
    if (!inited) {
        llama_log_set(log_cb, NULL);
        ggml_log_set(log_cb, NULL);
        inited = true;
    }

    struct llama_model_params mp = llama_model_default_params();
    if (maybe_params) {
        mp.n_gpu_layers = maybe_params->n_gpu_layers;
    }
    // Load from the in-memory bytes via fmemopen -> no filesystem I/O, so this
    // sync export never blocks on wasip3's async wasi:filesystem.
    FILE * f = fmemopen(data->ptr, data->len, "rb");
    if (!f) {
        set_err(err, "fmemopen failed");
        return false;
    }
    struct llama_model * m = llama_model_load_from_file_ptr(f, mp);
    fclose(f);
    if (!m) {
        set_err(err, "failed to load model");
        return false;
    }
    model_t * rep = (model_t *) malloc(sizeof(model_t));
    rep->model = m;
    rep->vocab = llama_model_get_vocab(m);
    *ret = exports_cosmonic_llama_cpp_api_model_new(rep);
    return true;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_static_model_create(
        provider_list_u8_t * data,
        exports_cosmonic_llama_cpp_api_model_params_t * maybe_params) {
    exports_cosmonic_llama_cpp_api_result_own_model_string_t ret;
    ret.is_err = !model_create(data, maybe_params, &ret.val.ok, &ret.val.err);
    exports_cosmonic_llama_cpp_api_static_model_create_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

UNREACHABLE_CALLBACK(exports_cosmonic_llama_cpp_api_static_model_create_callback)

void exports_cosmonic_llama_cpp_api_model_destructor(model_t * rep) {
    ensure_stack();  // wasip3: resource-drop callbacks run with SP=0; set it first
    llama_model_free(rep->model);
    free(rep);
}

bool exports_cosmonic_llama_cpp_api_method_model_tokenize(
        model_t * self, provider_string_t * text, bool add_special,
        provider_list_u32_t * ret, provider_string_t * err) {
    const char * t = (const char *) text->ptr;
    int32_t len = (int32_t) text->len;
    int n = -llama_tokenize(self->vocab, t, len, NULL, 0, add_special, true);
    uint32_t * out = (uint32_t *) malloc(sizeof(uint32_t) * (n ? (size_t) n : 1));
    if (llama_tokenize(self->vocab, t, len, (llama_token *) out, n, add_special, true) < 0) {
        free(out);
        set_err(err, "failed to tokenize");
        return false;
    }
    ret->ptr = out;
    ret->len = (size_t) n;
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_model_detokenize(
        model_t * self, provider_list_u32_t * tokens,
        provider_string_t * ret, provider_string_t * err) {
    size_t cap = tokens->len * 8 + 16;
    char * buf = (char *) malloc(cap);
    int n = llama_detokenize(self->vocab, (const llama_token *) tokens->ptr, tokens->len,
                             buf, cap, false, true);
    if (n < 0) {
        cap = (size_t) (-n);
        buf = (char *) realloc(buf, cap);
        n = llama_detokenize(self->vocab, (const llama_token *) tokens->ptr, tokens->len,
                             buf, cap, false, true);
    }
    if (n < 0) {
        free(buf);
        set_err(err, "failed to detokenize");
        return false;
    }
    provider_string_dup_n(ret, buf, (size_t) n);
    free(buf);
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_model_is_eog(model_t * self, uint32_t token) {
    return llama_vocab_is_eog(self->vocab, token);
}

bool exports_cosmonic_llama_cpp_api_method_model_apply_chat_template(
        model_t * self, exports_cosmonic_llama_cpp_api_list_chat_message_t * messages,
        bool add_assistant, provider_string_t * ret, provider_string_t * err) {
    const char * tmpl = llama_model_chat_template(self->model, NULL);
    if (!tmpl) {
        set_err(err, "model has no chat template");
        return false;
    }

    // Null-terminated copies for llama_chat_message (role/content are const char*).
    size_t nmsg = messages->len;
    struct llama_chat_message * msgs = (struct llama_chat_message *) malloc(nmsg * sizeof(*msgs));
    for (size_t i = 0; i < nmsg; i++) {
        provider_string_t * role = &messages->ptr[i].role;
        provider_string_t * content = &messages->ptr[i].content;
        char * r = (char *) malloc(role->len + 1);
        memcpy(r, role->ptr, role->len);
        r[role->len] = '\0';
        char * c = (char *) malloc(content->len + 1);
        memcpy(c, content->ptr, content->len);
        c[content->len] = '\0';
        msgs[i].role = r;
        msgs[i].content = c;
    }

    size_t cap = 1024;
    char * buf = (char *) malloc(cap);
    int n = llama_chat_apply_template(tmpl, msgs, nmsg, add_assistant, buf, cap);
    if (n > (int) cap) {
        cap = (size_t) n;
        buf = (char *) realloc(buf, cap);
        n = llama_chat_apply_template(tmpl, msgs, nmsg, add_assistant, buf, cap);
    }

    for (size_t i = 0; i < nmsg; i++) {
        free((void *) msgs[i].role);
        free((void *) msgs[i].content);
    }
    free(msgs);

    if (n < 0) {
        free(buf);
        set_err(err, "failed to apply chat template");
        return false;
    }
    provider_string_dup_n(ret, buf, (size_t) n);
    free(buf);
    return true;
}

void exports_cosmonic_llama_cpp_api_method_model_description(model_t * self, provider_string_t * ret) {
    ensure_stack();
    // Fail closed: an unwritten buf would otherwise be strlen'd by
    // provider_string_dup and hand the caller a slice of linear memory.
    char buf[256] = {0};
    int n = llama_model_desc(self->model, buf, sizeof(buf));
    if (n < 0) {
        buf[0] = '\0';
    }
    provider_string_dup(ret, buf);
}

uint32_t exports_cosmonic_llama_cpp_api_method_model_n_ctx_train(model_t * self) {
    return llama_model_n_ctx_train(self->model);
}

// --- context ---

static bool append_tokens(context_t * self, const uint32_t * toks, size_t count, provider_string_t * err) {
    for (size_t i = 0; i < count; i += self->batch_size) {
        size_t rem = count - i;
        size_t n = self->batch_size < rem ? self->batch_size : rem;
        struct llama_batch batch = llama_batch_get_one((llama_token *) (toks + i), n);
        if (llama_decode(self->ctx, batch)) {
            set_err(err, "decode failed");
            return false;
        }
        self->past += n;
    }
    return true;
}

static bool context_create(
        model_t * model, exports_cosmonic_llama_cpp_api_context_params_t * maybe_params,
        exports_cosmonic_llama_cpp_api_own_context_t * ret, provider_string_t * err) {
    struct llama_context_params cp = llama_context_default_params();
    // Treat 0 as "unset": the WIT records these as plain u32 (not option) but documents
    // per-field defaults, and n_batch == 0 would make append-tokens loop forever.
    cp.n_ctx   = (maybe_params && maybe_params->n_ctx)   ? maybe_params->n_ctx   : 4096;
    cp.n_batch = (maybe_params && maybe_params->n_batch) ? maybe_params->n_batch : 512;
    // WASI is single-threaded (pthread stubs); force ggml to compute on one thread.
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    struct llama_context * c = llama_init_from_model(model->model, cp);
    if (!c) {
        set_err(err, "failed to create context");
        return false;
    }
    context_t * rep = (context_t *) malloc(sizeof(context_t));
    rep->ctx = c;
    rep->model = model;
    rep->batch_size = cp.n_batch;
    rep->past = 0;
    *ret = exports_cosmonic_llama_cpp_api_context_new(rep);
    return true;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_static_context_create(
        exports_cosmonic_llama_cpp_api_borrow_model_t model,
        exports_cosmonic_llama_cpp_api_context_params_t * maybe_params) {
    exports_cosmonic_llama_cpp_api_result_own_context_string_t ret;
    ret.is_err = !context_create(model, maybe_params, &ret.val.ok, &ret.val.err);
    exports_cosmonic_llama_cpp_api_static_context_create_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

UNREACHABLE_CALLBACK(exports_cosmonic_llama_cpp_api_static_context_create_callback)

void exports_cosmonic_llama_cpp_api_context_destructor(context_t * rep) {
    ensure_stack();  // wasip3: resource-drop callbacks run with SP=0; set it first
    llama_free(rep->ctx);
    free(rep);
}

static bool context_append(
        context_t * self, provider_string_t * text, provider_string_t * err) {

    const char * t = (const char *) text->ptr;
    int32_t len = (int32_t) text->len;
    bool add_special = self->past == 0;
    int n = -llama_tokenize(self->model->vocab, t, len, NULL, 0, add_special, true);
    uint32_t * toks = (uint32_t *) malloc(sizeof(uint32_t) * (n ? (size_t) n : 1));
    if (llama_tokenize(self->model->vocab, t, len, (llama_token *) toks, n, add_special, true) < 0) {
        free(toks);
        set_err(err, "failed to tokenize");
        return false;
    }
    bool ok = append_tokens(self, toks, (size_t) n, err);
    free(toks);
    return ok;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_context_append(
        context_t * self, provider_string_t * text) {
    exports_cosmonic_llama_cpp_api_result_void_string_t ret;
    ret.is_err = !context_append(self, text, &ret.val.err);
    exports_cosmonic_llama_cpp_api_method_context_append_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

UNREACHABLE_CALLBACK(exports_cosmonic_llama_cpp_api_method_context_append_callback)

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_context_append_tokens(
        context_t * self, provider_list_u32_t * tokens) {
    exports_cosmonic_llama_cpp_api_result_void_string_t ret;
    ret.is_err = !append_tokens(self, tokens->ptr, tokens->len, &ret.val.err);
    exports_cosmonic_llama_cpp_api_method_context_append_tokens_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

UNREACHABLE_CALLBACK(exports_cosmonic_llama_cpp_api_method_context_append_tokens_callback)

uint32_t exports_cosmonic_llama_cpp_api_method_context_n_past(context_t * self) {
    return self->past;
}

void exports_cosmonic_llama_cpp_api_method_context_clear(context_t * self) {
    llama_memory_clear(llama_get_memory(self->ctx), true);
    self->past = 0;
}

// --- sampler ---

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_constructor_sampler(
        exports_cosmonic_llama_cpp_api_sampler_params_t * maybe_params) {
    sampler_t * rep = (sampler_t *) malloc(sizeof(sampler_t));
    rep->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!maybe_params || maybe_params->temp <= 0) {
        llama_sampler_chain_add(rep->smpl, llama_sampler_init_greedy());
    } else {
        if (maybe_params->top_k > 0) { llama_sampler_chain_add(rep->smpl, llama_sampler_init_top_k(maybe_params->top_k)); }
        if (maybe_params->top_p < 1) { llama_sampler_chain_add(rep->smpl, llama_sampler_init_top_p(maybe_params->top_p, 1)); }
        if (maybe_params->min_p > 0) { llama_sampler_chain_add(rep->smpl, llama_sampler_init_min_p(maybe_params->min_p, 1)); }
        llama_sampler_chain_add(rep->smpl, llama_sampler_init_temp(maybe_params->temp));
        llama_sampler_chain_add(rep->smpl, llama_sampler_init_dist(maybe_params->seed));
    }
    return exports_cosmonic_llama_cpp_api_sampler_new(rep);
}

void exports_cosmonic_llama_cpp_api_sampler_destructor(sampler_t * rep) {
    ensure_stack();  // wasip3: resource-drop callbacks run with SP=0; set it first
    llama_sampler_free(rep->smpl);
    free(rep);
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_sampler_sample(
        sampler_t * self, context_t * ctx) {
    // No error channel (returns u32): an empty context has no logits at index -1,
    // so fail loudly instead of sampling garbage. Caller must append a prompt first.
    if (ctx->past == 0) {
        fputs("sampler.sample: context is empty (append a prompt first)\n", stderr);
        abort();
    }
    exports_cosmonic_llama_cpp_api_method_sampler_sample_return(
        llama_sampler_sample(self->smpl, ctx->ctx, -1));
    return PROVIDER_CALLBACK_CODE_EXIT;
}

UNREACHABLE_CALLBACK(exports_cosmonic_llama_cpp_api_method_sampler_sample_callback)
