// JSON Schema to GBNF for inference.c, through llama.cpp's own converter
// (common/json-schema-to-grammar.cpp, compiled into the component).
#include "json-schema-to-grammar.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

// The converter uses these three from llama.cpp's common library, which the
// component does not build: it pulls in threads and filesystem code WASI lacks.
// Same signatures and behavior as common/common.cpp.
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

// Returns a malloc'd grammar in *out, or a malloc'd message in *err.
extern "C" bool llama_wit_json_schema_to_grammar(const char * schema, size_t len, char ** out, char ** err) {
    try {
        const auto parsed = nlohmann::ordered_json::parse(schema, schema + len);
        *out = strdup(json_schema_to_grammar(parsed).c_str());
        return true;
    } catch (const std::exception & e) {
        *err = strdup(e.what());
        return false;
    }
}
