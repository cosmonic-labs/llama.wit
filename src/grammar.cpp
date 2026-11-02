// cosmonic:llama-cpp/json-schema, through llama.cpp's own converter
// (common/json-schema-to-grammar.cpp). Only built for provider-with-json-schema.
#include "json-schema-to-grammar.h"
#include "provider.h"

#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

// The converter uses these three from llama.cpp's common library, which the
// component does not build: it pulls in threads and filesystem code WASI lacks.
// Same signatures and behavior as common/common.cpp.
// TODO: remove these and link common/common.cpp once it builds for WASI.
// From inference.c: the error can quote a cut-off UTF-8 sequence from the input
extern "C" void string_from_bytes(const char * in, size_t len, provider_string_t * out);

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    std::string result;
    result.reserve(str.length() * n);
    for (size_t i = 0; i < n; ++i) {
        result += str;
    }
    return result;
}

bool exports_cosmonic_llama_cpp_json_schema_to_grammar(
        provider_string_t * schema, provider_string_t * ret, provider_string_t * err) {
    bool ok;
    try {
        const auto parsed = common_json::parse(std::string((const char *) schema->ptr, schema->len));
        provider_string_dup(ret, json_schema_to_grammar(parsed).c_str());
        ok = true;
    } catch (const std::exception & e) {
        string_from_bytes(e.what(), strlen(e.what()), err);
        ok = false;
    }
    provider_string_free(schema);
    return ok;
}
