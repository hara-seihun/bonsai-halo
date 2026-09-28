// Compare the replayable target's features for every DFlash2 layer against the
// ordinary eight-row target pass. No drafter weights are needed to exercise capture.
#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace halo;

int main(int argc, char ** argv) {
    try {
        const char * model = argc > 1 ? argv[1] : "../../data/bonsai2/PTQ1_0.gguf";
        Engine e;
        e.load(model, 16, 2, 256);
        e.prepare_batch(16, Engine::BATCH_WORKSPACE_ONLY);
        e.df.loaded = true; // feature capture is gated on this, not on drafter weights
        Engine::Seq plain, batch;
        plain.slot = 0; batch.slot = 1;
        e.reset_seq(plain); e.reset_seq(batch);
        std::vector<int> prompt { 144, 521, 912, 1207, 788, 329, 601, 114 };
        auto pass = [&](bool batched, const std::vector<int> & tokens) {
            Engine::Seq & s = batched ? batch : plain;
            std::vector<Engine::Seq *> sv { &s };
            std::vector<std::vector<int>> tv { tokens };
            std::vector<int> am;
            HIP_CHECK_H(hipMemsetAsync(e.hcap, 0, (size_t) tokens.size() * NCAP * D * sizeof(float), e.stream));
            if (batched) { e.batch_mode = 4; e.forward_batch(sv, tv, true, &am); }
            else { e.fwd.hcap = e.hcap; e.forward(sv, tv, true, &am); e.fwd.hcap = nullptr; }
            return std::pair{am, e.read(e.hcap, (size_t) tokens.size() * NCAP * D)};
        };
        auto compare = [&](const char * phase, const auto & reference, const auto & candidate, int nrows) {
            int argmax_diff = 0;
            for (int row = 0; row < nrows; ++row) argmax_diff += reference.first[row] != candidate.first[row];
            if (argmax_diff) throw std::runtime_error(std::string(phase) + ": target argmax differs");
            for (int layer = 0; layer < NCAP; ++layer) {
                double rms = 0, delta = 0;
                size_t count = (size_t) nrows * D;
                for (int row = 0; row < nrows; ++row)
                    for (int d = 0; d < D; ++d) {
                        const size_t at = ((size_t) row * NCAP + layer) * D + d;
                        double a = reference.second[at], b = candidate.second[at];
                        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite capture");
                        rms += a * a;
                        delta += (a - b) * (a - b);
                    }
                rms = std::sqrt(rms / count); delta = std::sqrt(delta / count);
                std::printf("%s feature %d: reference RMS %.6f, difference RMS %.6f\n", phase, layer, rms, delta);
                if (rms < 0.01 || delta > rms * 0.05)
                    throw std::runtime_error(std::string(phase) + ": capture mismatch at layer " + std::to_string(layer));
            }
        };
        auto ref = pass(false, prompt), got = pass(true, prompt);
        compare("initial", ref, got, prompt.size());
        Engine::rollback(plain, 3); Engine::rollback(batch, 3);
        if (plain.len != batch.len || plain.keep != 3 || batch.keep != 3 || plain.parity != batch.parity)
            throw std::runtime_error("rollback bookkeeping mismatch");
        std::vector<int> continuation { 881, 323, 41, 102 };
        ref = pass(false, continuation); got = pass(true, continuation);
        compare("replay", ref, got, continuation.size());
        std::puts("mode-4 feature capture and replay passed bounded comparison");
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "serve capture check: %s\n", ex.what());
        return 1;
    }
}
