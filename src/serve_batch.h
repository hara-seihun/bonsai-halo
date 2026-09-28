#pragma once
#include "engine.h"
#include <functional>
#include <memory>
#include <vector>

namespace halo {

// Continuous batching for the resident server.
//
// The engine is a single-threaded object driven by one HIP stream, so every request used to take a
// mutex and run `Engine::generate` alone: N concurrent clients shared one single-stream decode and
// each got 1/N of it, while the same box runs eight sequences through one weight stream for about
// the cost of one. `ServeBatch` owns the engine on its own thread instead. Requests are submitted
// from HTTP handler threads, admitted into sequence state slots, and decoded together: one pass per
// step carrying one row per active request.
//
// What a request's arithmetic depends on does not include which rows share its pass. A row carries
// its own token, state slot and position; the recurrent state is per slot, the K/V cache is per
// slot, and the weights are shared and read-only. A batched decode step is therefore the same
// eight-row persistent-kernel pass the speculative verify path has always run, and the tokens it
// produces are the tokens the serial route produces.
//
// One request alone still gets the drafted route it has today: `ServeBatch` runs DFlash2
// speculation whenever a request is the only one in flight. A request that has shared a step with
// another request stays on plain decode for the rest of its life, because the drafter's cache is
// fed by features captured during that sequence's own passes and a batched step does not feed it.
struct ServeBatch {
    // `max_active` requests decode together; 1 reproduces the serialized server exactly.
    // `e.nslots` is the hard cap, since each active request needs its own state slot.
    ServeBatch(Engine & e, int max_active);
    ~ServeBatch();

    // Runs one request to completion, calling `on_token` on the caller's thread for every token,
    // in order. `on_token` returning false stops generation, which is how the server stops at an
    // end-of-generation token or a disconnected client. Throws what the engine throws.
    void run(const std::vector<int> & prompt, const Engine::GenParams & params,
             const std::function<bool(int)> & on_token, Engine::SpecStats & stats);

    int max_active() const;
    // Requests decoding right now, for logs and /health.
    int active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace halo
