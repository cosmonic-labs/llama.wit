// cosmonic:llama-cpp/api (wit/llama.wit) over llama.cpp's C API
// Exports own their arguments, so each one frees what it is handed (borrows aside)
#include "provider.h"
#include "llama.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef exports_cosmonic_llama_cpp_api_model_t   model_t;
typedef exports_cosmonic_llama_cpp_api_context_t context_t;
typedef exports_cosmonic_llama_cpp_api_sampler_t sampler_t;

// The C header forward-declares these; we define the reps.
struct exports_cosmonic_llama_cpp_api_model_t {
    struct llama_model *       model;
    const struct llama_vocab * vocab;
    uint32_t                   refs;     // the handle, plus each context and sampler using it
};
struct exports_cosmonic_llama_cpp_api_context_t {
    struct llama_context * ctx;
    model_t *              model;    // retained, so dropping the model handle first is safe
};
struct exports_cosmonic_llama_cpp_api_sampler_t {
    struct llama_sampler * smpl;
    model_t *              model;    // retained; NULL unless built from (or holding) a model's sampler
    llama_token_data *     cur;      // candidates for sample, reused across calls
    size_t                 cur_cap;
};

static model_t * model_retain(model_t * model) {
    if (model) {
        model->refs++;
    }
    return model;
}

static void model_release(model_t * model) {
    if (model && --model->refs == 0) {
        llama_model_free(model->model);
        free(model);
    }
}

static void set_err(provider_string_t * err, const char * msg) {
    provider_string_dup(err, msg);
}

static char * cstr(const provider_string_t * s) {
    char * out = (char *) malloc(s->len + 1);
    memcpy(out, s->ptr, s->len);
    out[s->len] = '\0';
    return out;
}

// llama.cpp throws on an out-of-range token id, which traps the component.
static bool tokens_in_range(const model_t * model, const uint32_t * toks, size_t count) {
    uint32_t n_vocab = (uint32_t) llama_vocab_n_tokens(model->vocab);
    for (size_t i = 0; i < count; i++) {
        if (toks[i] >= n_vocab) {
            return false;
        }
    }
    return true;
}

// Lifting a string that is not UTF-8 traps the caller, so replace bad bytes with U+FFFD
// Not static: grammar.cpp uses it too
void string_from_bytes(const char * in, size_t len, provider_string_t * out) {
    const unsigned char * s = (const unsigned char *) in;
    // Each input byte becomes at most one 3-byte U+FFFD.
    uint8_t * buf = (uint8_t *) malloc(len * 3 + 1);
    size_t o = 0;
    size_t i = 0;
    while (i < len) {
        unsigned char c = s[i];
        if (c < 0x80) {
            buf[o++] = c;
            i++;
            continue;
        }
        size_t need = 0;
        uint32_t cp = 0;
        uint32_t min = 0;
        if ((c & 0xE0) == 0xC0) {
            need = 1; cp = c & 0x1F; min = 0x80;
        } else if ((c & 0xF0) == 0xE0) {
            need = 2; cp = c & 0x0F; min = 0x800;
        } else if ((c & 0xF8) == 0xF0) {
            need = 3; cp = c & 0x07; min = 0x10000;
        }
        size_t j = 1;
        while (need && j <= need && i + j < len && (s[i + j] & 0xC0) == 0x80) {
            cp = (cp << 6) | (s[i + j] & 0x3F);
            j++;
        }
        bool valid = need && j > need && cp >= min && cp <= 0x10FFFF &&
                     !(cp >= 0xD800 && cp <= 0xDFFF);
        if (valid) {
            memcpy(buf + o, s + i, j);
            o += j;
            i += j;
        } else {
            buf[o++] = 0xEF; buf[o++] = 0xBF; buf[o++] = 0xBD;
            // A cut-off sequence is one replacement; any other bad byte is its own.
            i += (need && j <= need) ? j : 1;
        }
    }
    out->ptr = buf;
    out->len = o;
}

// Sync exports can't block on wasip3's async stderr, so never log
static void log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) level; (void) text; (void) user;
}

// --- model ---

// The Canonical ABI limits one stream read to Buffer.MAX_LENGTH items
// (CanonicalABI.md: `class Buffer: MAX_LENGTH = 2**28 - 1`), and wasmtime traps
// on a larger one rather than reading less.
#define STREAM_MAX_READ ((1u << 28) - 1u)

// Reads the whole stream into `buf`, which holds `size` bytes. llama.cpp's loader
// seeks backwards, so the GGUF has to be in memory before it starts.
static bool stream_read_all(
        exports_cosmonic_llama_cpp_api_stream_u8_t reader, uint8_t * buf, size_t size,
        provider_string_t * err) {
    provider_waitable_set_t set = provider_waitable_set_new();
    provider_waitable_join(reader, set);
    size_t got = 0;
    bool ended = false;
    while (got < size && !ended) {
        size_t amt = size - got;
        if (amt > STREAM_MAX_READ) {
            amt = STREAM_MAX_READ;
        }
        provider_waitable_status_t st =
            exports_cosmonic_llama_cpp_api_stream_u8_read(reader, buf + got, amt);
        if (st == PROVIDER_WAITABLE_STATUS_BLOCKED) {
            provider_event_t ev;  // the reader is the set's only waitable
            provider_waitable_set_wait(set, &ev);
            st = (provider_waitable_status_t) ev.code;
        }
        got += PROVIDER_WAITABLE_COUNT(st);
        ended = PROVIDER_WAITABLE_STATE(st) != PROVIDER_WAITABLE_COMPLETED;
    }
    provider_waitable_join(reader, 0);  // leave the set before dropping it
    provider_waitable_set_drop(set);
    if (got < size) {
        set_err(err, "stream ended before `size` bytes");
        return false;
    }
    return true;
}

static bool model_create(
        exports_cosmonic_llama_cpp_api_stream_u8_t data,
        uint64_t size,
        exports_cosmonic_llama_cpp_api_model_params_t * maybe_params,
        exports_cosmonic_llama_cpp_api_own_model_t * ret,
        provider_string_t * err) {
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

    uint8_t * buf = size <= SIZE_MAX ? (uint8_t *) malloc(size ? (size_t) size : 1) : NULL;
    if (!buf) {
        exports_cosmonic_llama_cpp_api_stream_u8_drop_readable(data);
        set_err(err, "model does not fit in memory");
        return false;
    }
    bool read = stream_read_all(data, buf, (size_t) size, err);
    exports_cosmonic_llama_cpp_api_stream_u8_drop_readable(data);
    if (!read) {
        free(buf);
        return false;
    }

    // fmemopen avoids wasip3's async wasi:filesystem
    FILE * f = fmemopen(buf, (size_t) size, "rb");
    if (!f) {
        free(buf);
        set_err(err, "fmemopen failed");
        return false;
    }
    // Loading copies the tensors out, so the GGUF bytes can go once the file is closed
    struct llama_model * m = llama_model_load_from_file_ptr(f, mp);
    fclose(f);
    free(buf);
    if (!m) {
        set_err(err, "failed to load model");
        return false;
    }
    model_t * rep = (model_t *) malloc(sizeof(model_t));
    rep->model = m;
    rep->vocab = llama_model_get_vocab(m);
    rep->refs = 1;
    *ret = exports_cosmonic_llama_cpp_api_model_new(rep);
    return true;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_static_model_create(
        exports_cosmonic_llama_cpp_api_stream_u8_t data,
        uint64_t size,
        exports_cosmonic_llama_cpp_api_model_params_t * maybe_params) {
    exports_cosmonic_llama_cpp_api_result_own_model_string_t ret;
    ret.is_err = !model_create(data, size, maybe_params, &ret.val.ok, &ret.val.err);
    exports_cosmonic_llama_cpp_api_static_model_create_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_static_model_create_callback(provider_event_t * event) {
    (void) event;
    // exports_cosmonic_llama_cpp_api_static_model_create never returns yield/wait
    abort();
}

void exports_cosmonic_llama_cpp_api_model_destructor(model_t * rep) {
    model_release(rep);
}

static bool model_tokenize(
        model_t * self, provider_string_t * text, bool add_special, bool parse_special,
        provider_list_u32_t * ret, provider_string_t * err) {
    const char * t = (const char *) text->ptr;
    int32_t len = (int32_t) text->len;
    int n = -llama_tokenize(self->vocab, t, len, NULL, 0, add_special, parse_special);
    uint32_t * out = (uint32_t *) malloc(sizeof(uint32_t) * (n ? (size_t) n : 1));
    if (llama_tokenize(self->vocab, t, len, (llama_token *) out, n, add_special, parse_special) < 0) {
        free(out);
        set_err(err, "failed to tokenize");
        return false;
    }
    ret->ptr = out;
    ret->len = (size_t) n;
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_model_tokenize(
        model_t * self, provider_string_t * text, bool add_special, bool parse_special,
        provider_list_u32_t * ret, provider_string_t * err) {
    bool ok = model_tokenize(self, text, add_special, parse_special, ret, err);
    provider_string_free(text);
    return ok;
}

static bool model_detokenize(
        model_t * self, provider_list_u32_t * tokens, bool remove_special, bool unparse_special,
        provider_string_t * ret, provider_string_t * err) {
    if (!tokens_in_range(self, tokens->ptr, tokens->len)) {
        set_err(err, "token out of range");
        return false;
    }
    if (tokens->len > (INT32_MAX - 16) / 8) {
        set_err(err, "too many tokens");
        return false;
    }
    const llama_token * toks = (const llama_token *) tokens->ptr;
    int32_t n_tokens = (int32_t) tokens->len;
    int32_t cap = n_tokens * 8 + 16;
    char * buf = (char *) malloc((size_t) cap);
    int32_t n = llama_detokenize(self->vocab, toks, n_tokens, buf, cap, remove_special, unparse_special);
    if (n < 0) {
        cap = -n;
        free(buf);
        buf = (char *) malloc((size_t) cap);
        n = llama_detokenize(self->vocab, toks, n_tokens, buf, cap, remove_special, unparse_special);
    }
    if (n < 0) {
        free(buf);
        set_err(err, "failed to detokenize");
        return false;
    }
    string_from_bytes(buf, (size_t) n, ret);
    free(buf);
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_model_detokenize(
        model_t * self, provider_list_u32_t * tokens, bool remove_special, bool unparse_special,
        provider_string_t * ret, provider_string_t * err) {
    bool ok = model_detokenize(self, tokens, remove_special, unparse_special, ret, err);
    provider_list_u32_free(tokens);
    return ok;
}

bool exports_cosmonic_llama_cpp_api_method_model_token_to_piece(
        model_t * self, uint32_t token, int32_t lstrip, bool special,
        provider_list_u8_t * ret, provider_string_t * err) {
    if (!tokens_in_range(self, &token, 1)) {
        set_err(err, "token out of range");
        return false;
    }
    char small[64];
    char * buf = small;
    int n = llama_token_to_piece(self->vocab, (llama_token) token, buf, sizeof(small), lstrip, special);
    if (n < 0) {
        buf = (char *) malloc((size_t) (-n));
        n = llama_token_to_piece(self->vocab, (llama_token) token, buf, -n, lstrip, special);
    }
    if (n < 0) {
        if (buf != small) {
            free(buf);
        }
        set_err(err, "failed to convert token to piece");
        return false;
    }
    uint8_t * out = (uint8_t *) malloc(n ? (size_t) n : 1);
    memcpy(out, buf, (size_t) n);
    if (buf != small) {
        free(buf);
    }
    ret->ptr = out;
    ret->len = (size_t) n;
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_model_is_eog(model_t * self, uint32_t token) {
    return tokens_in_range(self, &token, 1) && llama_vocab_is_eog(self->vocab, token);
}

bool exports_cosmonic_llama_cpp_api_method_model_chat_template(
        model_t * self, provider_string_t * maybe_name, provider_string_t * ret) {
    char * name = maybe_name ? cstr(maybe_name) : NULL;
    const char * tmpl = llama_model_chat_template(self->model, name);
    free(name);
    if (maybe_name) {
        provider_string_free(maybe_name);
    }
    if (!tmpl) {
        return false;
    }
    string_from_bytes(tmpl, strlen(tmpl), ret);
    return true;
}

static bool chat_apply_template(
        provider_string_t * tmpl, exports_cosmonic_llama_cpp_api_list_chat_message_t * chat,
        bool add_ass, provider_string_t * ret, provider_string_t * err) {
    // llama_chat_message needs null-terminated strings
    size_t nmsg = chat->len;
    struct llama_chat_message * msgs = (struct llama_chat_message *) malloc((nmsg ? nmsg : 1) * sizeof(*msgs));
    for (size_t i = 0; i < nmsg; i++) {
        msgs[i].role = cstr(&chat->ptr[i].role);
        msgs[i].content = cstr(&chat->ptr[i].content);
    }
    char * t = cstr(tmpl);

    int32_t cap = 1024;
    char * buf = (char *) malloc((size_t) cap);
    int32_t n = llama_chat_apply_template(t, msgs, nmsg, add_ass, buf, cap);
    if (n > cap) {
        cap = n;
        buf = (char *) realloc(buf, (size_t) cap);
        n = llama_chat_apply_template(t, msgs, nmsg, add_ass, buf, cap);
    }

    free(t);
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
    string_from_bytes(buf, (size_t) n, ret);
    free(buf);
    return true;
}

bool exports_cosmonic_llama_cpp_api_chat_apply_template(
        provider_string_t * tmpl, exports_cosmonic_llama_cpp_api_list_chat_message_t * chat,
        bool add_ass, provider_string_t * ret, provider_string_t * err) {
    bool ok = chat_apply_template(tmpl, chat, add_ass, ret, err);
    provider_string_free(tmpl);
    exports_cosmonic_llama_cpp_api_list_chat_message_free(chat);
    return ok;
}

void exports_cosmonic_llama_cpp_api_method_model_description(model_t * self, provider_string_t * ret) {
    // Zeroed so a failed llama_model_desc can't leak linear memory
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

uint32_t exports_cosmonic_llama_cpp_api_method_model_n_embd_out(model_t * self) {
    return (uint32_t) llama_model_n_embd_out(self->model);
}

uint32_t exports_cosmonic_llama_cpp_api_method_model_n_cls_out(model_t * self) {
    return llama_model_n_cls_out(self->model);
}

uint32_t exports_cosmonic_llama_cpp_api_method_model_n_vocab(model_t * self) {
    return (uint32_t) llama_vocab_n_tokens(self->vocab);
}

uint32_t exports_cosmonic_llama_cpp_api_method_model_n_swa(model_t * self) {
    return (uint32_t) llama_model_n_swa(self->model);
}

// --- context ---

static bool context_create(
        model_t * model, exports_cosmonic_llama_cpp_api_context_params_t * maybe_params,
        exports_cosmonic_llama_cpp_api_own_context_t * ret, provider_string_t * err) {
    struct llama_context_params cp = llama_context_default_params();
    if (maybe_params) {
        // llama.cpp only rejects n_batch == 0 together with n_ubatch == 0
        if (maybe_params->n_batch == 0) {
            set_err(err, "n-batch must not be 0");
            return false;
        }
        cp.n_ctx      = maybe_params->n_ctx;
        cp.n_batch    = maybe_params->n_batch;
        cp.n_ubatch   = maybe_params->n_ubatch;
        cp.embeddings = maybe_params->embeddings;
    }
    struct llama_context * c = llama_init_from_model(model->model, cp);
    if (!c) {
        set_err(err, "failed to create context");
        return false;
    }
    context_t * rep = (context_t *) malloc(sizeof(context_t));
    rep->ctx = c;
    rep->model = model_retain(model);
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

provider_callback_code_t exports_cosmonic_llama_cpp_api_static_context_create_callback(provider_event_t * event) {
    (void) event;
    // exports_cosmonic_llama_cpp_api_static_context_create never returns yield/wait
    abort();
}

void exports_cosmonic_llama_cpp_api_context_destructor(context_t * rep) {
    llama_free(rep->ctx);
    model_release(rep->model);
    free(rep);
}

uint32_t exports_cosmonic_llama_cpp_api_method_context_n_ctx(context_t * self) {
    return llama_n_ctx(self->ctx);
}

uint32_t exports_cosmonic_llama_cpp_api_method_context_n_batch(context_t * self) {
    return llama_n_batch(self->ctx);
}

uint32_t exports_cosmonic_llama_cpp_api_method_context_n_ubatch(context_t * self) {
    return llama_n_ubatch(self->ctx);
}

exports_cosmonic_llama_cpp_api_pooling_type_t exports_cosmonic_llama_cpp_api_method_context_pooling_type(context_t * self) {
    // The WIT enum is llama_pooling_type shifted up by one, from UNSPECIFIED = -1
    return (exports_cosmonic_llama_cpp_api_pooling_type_t) (llama_pooling_type(self->ctx) + 1);
}

static int32_t context_decode(context_t * self, provider_list_u32_t * tokens) {
    // These are asserts in llama_decode, which would trap the component
    uint32_t n = (uint32_t) tokens->len;
    bool whole_ubatch = !llama_get_causal_attn(self->ctx) || !llama_get_memory(self->ctx);
    if (n > llama_n_batch(self->ctx) || (whole_ubatch && n > llama_n_ubatch(self->ctx))) {
        return -1;
    }
    return llama_decode(self->ctx, llama_batch_get_one((llama_token *) tokens->ptr, (int32_t) n));
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_context_decode(
        context_t * self, provider_list_u32_t * tokens) {
    exports_cosmonic_llama_cpp_api_result_void_s32_t ret;
    int32_t code = context_decode(self, tokens);
    ret.is_err = code != 0;
    ret.val.err = code;
    provider_list_u32_free(tokens);
    exports_cosmonic_llama_cpp_api_method_context_decode_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_context_decode_callback(provider_event_t * event) {
    (void) event;
    // exports_cosmonic_llama_cpp_api_method_context_decode never returns yield/wait
    abort();
}

void exports_cosmonic_llama_cpp_api_method_context_memory_clear(context_t * self, bool data) {
    llama_memory_clear(llama_get_memory(self->ctx), data);
}

bool exports_cosmonic_llama_cpp_api_method_context_memory_seq_rm(
        context_t * self, int32_t seq_id, int32_t p0, int32_t p1) {
    // llama.cpp asserts on a seq_id outside [-1, n_seq_max)
    if (seq_id < 0) {
        seq_id = -1;
    } else if ((uint32_t) seq_id >= llama_n_seq_max(self->ctx)) {
        return false;
    }
    return llama_memory_seq_rm(llama_get_memory(self->ctx), seq_id, p0, p1);
}

int32_t exports_cosmonic_llama_cpp_api_method_context_memory_seq_pos_min(context_t * self, int32_t seq_id) {
    if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(self->ctx)) {
        return -1;
    }
    return llama_memory_seq_pos_min(llama_get_memory(self->ctx), seq_id);
}

int32_t exports_cosmonic_llama_cpp_api_method_context_memory_seq_pos_max(context_t * self, int32_t seq_id) {
    if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(self->ctx)) {
        return -1;
    }
    return llama_memory_seq_pos_max(llama_get_memory(self->ctx), seq_id);
}

static bool floats_dup(const float * src, size_t len, provider_list_f32_t * ret) {
    if (!src) {
        return false;
    }
    ret->ptr = (float *) malloc(len ? len * sizeof(float) : 1);
    memcpy(ret->ptr, src, len * sizeof(float));
    ret->len = len;
    return true;
}

bool exports_cosmonic_llama_cpp_api_method_context_get_embeddings_ith(
        context_t * self, int32_t i, provider_list_f32_t * ret) {
    // llama.cpp only fills per-token embeddings without pooling
    if (llama_pooling_type(self->ctx) != LLAMA_POOLING_TYPE_NONE) {
        return false;
    }
    size_t n_embd = (size_t) llama_model_n_embd_out(self->model->model);
    return floats_dup(llama_get_embeddings_ith(self->ctx, i), n_embd, ret);
}

bool exports_cosmonic_llama_cpp_api_method_context_get_embeddings_seq(
        context_t * self, int32_t seq_id, provider_list_f32_t * ret) {
    size_t len = llama_pooling_type(self->ctx) == LLAMA_POOLING_TYPE_RANK
        ? (size_t) llama_model_n_cls_out(self->model->model)
        : (size_t) llama_model_n_embd_out(self->model->model);
    return floats_dup(llama_get_embeddings_seq(self->ctx, seq_id), len, ret);
}

// --- sampler ---

static exports_cosmonic_llama_cpp_api_own_sampler_t sampler_new(struct llama_sampler * smpl, model_t * model) {
    sampler_t * rep = (sampler_t *) malloc(sizeof(sampler_t));
    rep->smpl = smpl;
    rep->model = model_retain(model);
    rep->cur = NULL;
    rep->cur_cap = 0;
    return exports_cosmonic_llama_cpp_api_sampler_new(rep);
}

void exports_cosmonic_llama_cpp_api_sampler_destructor(sampler_t * rep) {
    // NULL once a chain owns it
    if (rep->smpl) {
        llama_sampler_free(rep->smpl);
    }
    model_release(rep->model);
    free(rep->cur);
    free(rep);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_chain_init(void) {
    return sampler_new(llama_sampler_chain_init(llama_sampler_chain_default_params()), NULL);
}

bool exports_cosmonic_llama_cpp_api_method_sampler_chain_add(
        sampler_t * self, exports_cosmonic_llama_cpp_api_own_sampler_t smpl, provider_string_t * err) {
    sampler_t * added = exports_cosmonic_llama_cpp_api_sampler_rep(smpl);
    // chain_get(-1) returns the sampler itself only for a chain
    bool ok = false;
    if (llama_sampler_chain_get(self->smpl, -1) == NULL) {
        set_err(err, "the sampler is not a chain");
    } else if (self->model && added->model && self->model != added->model) {
        set_err(err, "the samplers were made for different models");
    } else {
        llama_sampler_chain_add(self->smpl, added->smpl);
        added->smpl = NULL;
        // The chain now holds the added sampler's model reference
        if (!self->model) {
            self->model = added->model;
            added->model = NULL;
        }
        ok = true;
    }
    exports_cosmonic_llama_cpp_api_sampler_drop_own(smpl);
    return ok;
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_greedy(void) {
    return sampler_new(llama_sampler_init_greedy(), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_dist(uint32_t seed) {
    return sampler_new(llama_sampler_init_dist(seed), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_top_k(int32_t k) {
    return sampler_new(llama_sampler_init_top_k(k), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_top_p(float p, uint32_t min_keep) {
    return sampler_new(llama_sampler_init_top_p(p, min_keep), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_min_p(float p, uint32_t min_keep) {
    return sampler_new(llama_sampler_init_min_p(p, min_keep), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_temp(float t) {
    return sampler_new(llama_sampler_init_temp(t), NULL);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_penalties(
        model_t * model, int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present) {
    return sampler_new(llama_sampler_init_penalties(
        llama_vocab_n_tokens(model->vocab), penalty_last_n, penalty_repeat, penalty_freq, penalty_present), model);
}

exports_cosmonic_llama_cpp_api_own_sampler_t exports_cosmonic_llama_cpp_api_static_sampler_init_logit_bias(
        model_t * model, exports_cosmonic_llama_cpp_api_list_logit_bias_t * logit_bias) {
    int32_t n_vocab = llama_vocab_n_tokens(model->vocab);
    llama_logit_bias * valid = (llama_logit_bias *) malloc(sizeof(*valid) * (logit_bias->len ? logit_bias->len : 1));
    int32_t n = 0;
    for (size_t i = 0; i < logit_bias->len; i++) {
        if (logit_bias->ptr[i].token < (uint32_t) n_vocab) {
            valid[n].token = (llama_token) logit_bias->ptr[i].token;
            valid[n].bias = logit_bias->ptr[i].bias;
            n++;
        }
    }
    struct llama_sampler * smpl = llama_sampler_init_logit_bias(n_vocab, n, valid);
    free(valid);
    exports_cosmonic_llama_cpp_api_list_logit_bias_free(logit_bias);
    return sampler_new(smpl, model);
}

bool exports_cosmonic_llama_cpp_api_static_sampler_init_grammar(
        model_t * model, provider_string_t * grammar_str, provider_string_t * grammar_root,
        exports_cosmonic_llama_cpp_api_own_sampler_t * ret, provider_string_t * err) {
    char * str = cstr(grammar_str);
    char * root = cstr(grammar_root);
    struct llama_sampler * smpl = llama_sampler_init_grammar(model->vocab, str, root);
    free(str);
    free(root);
    provider_string_free(grammar_str);
    provider_string_free(grammar_root);
    if (!smpl) {
        set_err(err, "the grammar does not parse");
        return false;
    }
    *ret = sampler_new(smpl, model);
    return true;
}

// llama_sampler_sample, spelled out as llama.h documents it, so a chain that picks no
// token or picks one the grammar ruled out fails here instead of asserting in llama.cpp
static bool sampler_sample(sampler_t * self, context_t * ctx, int32_t idx, uint32_t * ret, provider_string_t * err) {
    // The grammar sampler indexes the vocabulary it was built with
    if (self->model && self->model != ctx->model) {
        set_err(err, "the sampler was made for a different model than the context's");
        return false;
    }
    const float * logits = llama_get_logits_ith(ctx->ctx, idx);
    if (!logits) {
        set_err(err, "no logits for that output (decode tokens first)");
        return false;
    }
    size_t n_vocab = (size_t) llama_vocab_n_tokens(ctx->model->vocab);
    if (self->cur_cap < n_vocab) {
        free(self->cur);
        self->cur = (llama_token_data *) malloc(n_vocab * sizeof(llama_token_data));
        self->cur_cap = n_vocab;
    }
    for (size_t i = 0; i < n_vocab; i++) {
        self->cur[i] = (llama_token_data) { (llama_token) i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = { self->cur, n_vocab, -1, false };
    llama_sampler_apply(self->smpl, &cur_p);
    if (cur_p.selected < 0 || (size_t) cur_p.selected >= cur_p.size) {
        set_err(err, "the sampler picked no token (end the chain with greedy or dist)");
        return false;
    }
    llama_token_data picked = cur_p.data[cur_p.selected];
    if (isinf(picked.logit) && picked.logit < 0) {
        set_err(err, "the sampler picked a token that was ruled out (put the grammar earlier in the chain)");
        return false;
    }
    llama_sampler_accept(self->smpl, picked.id);
    *ret = (uint32_t) picked.id;
    return true;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_sampler_sample(
        sampler_t * self, context_t * ctx, int32_t idx) {
    exports_cosmonic_llama_cpp_api_result_u32_string_t ret;
    ret.is_err = !sampler_sample(self, ctx, idx, &ret.val.ok, &ret.val.err);
    exports_cosmonic_llama_cpp_api_method_sampler_sample_return(ret);
    return PROVIDER_CALLBACK_CODE_EXIT;
}

provider_callback_code_t exports_cosmonic_llama_cpp_api_method_sampler_sample_callback(provider_event_t * event) {
    (void) event;
    // exports_cosmonic_llama_cpp_api_method_sampler_sample never returns yield/wait
    abort();
}
