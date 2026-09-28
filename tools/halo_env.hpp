// Which route the measured process actually took, recorded from the environment.
//
// Route selection on this engine is environment driven and read once per process, so an on/off
// comparison is necessarily cross-process. A run that does not record which flags it saw cannot be
// paired safely afterwards: two "on" runs compare clean and look like a passing identity check.
// Recording the whole HALO_* set instead of one field per flag means a new route needs no edit
// here, and an older analysis still sees flags that did not exist when it was written.
#pragma once
#include <cstring>
#include <string>
#include "../vendor/nlohmann/json.hpp"

extern char ** environ;

inline nlohmann::json halo_env_snapshot() {
    nlohmann::json env = nlohmann::json::object();
    for (char ** e = environ; e && *e; ++e) {
        if (std::strncmp(*e, "HALO_", 5) != 0) continue;
        const char * eq = std::strchr(*e, '=');
        if (!eq) continue;
        env[std::string(*e, (size_t) (eq - *e))] = std::string(eq + 1);
    }
    return env;
}
