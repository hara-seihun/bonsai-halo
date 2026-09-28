// Isolate the 31-row SET_ROWS scatter with Qwen's 512-wide FP16 KV cell
// and exact host index permutation, independently of the complete graph.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdint>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        std::fprintf(stderr, "device[%zu]=%s type=%d\n", i, ggml_backend_dev_name(candidate), int(ggml_backend_dev_type(candidate)));
        if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = candidate;
    }
    if (!dev) return 2;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 3;
    std::vector<unsigned char> arena(ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 65536);
    ggml_init_params params{arena.size(), arena.data(), true};
    auto * ctx = ggml_init(params);
    constexpr int width = 512, streams = 32, stride = 256, count = 31;
    auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, width, streams * stride);
    auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, count);
    auto * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, count);
    auto * result = ggml_set_rows(ctx, dst, src, idx);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 4;
    std::vector<uint16_t> blank(width * streams * stride, 0);
    std::vector<float> values(width * count);
    std::vector<int64_t> indices(count);
    for (int r = 0; r < count; ++r) {
        indices[r] = (r + 1) * stride + 9;
        for (int c = 0; c < width; ++c) values[r * width + c] = (r + 1) + c / 1024.0f;
    }
    ggml_backend_tensor_set(dst, blank.data(), 0, blank.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(src, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(idx, indices.data(), 0, indices.size() * sizeof(int64_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
    std::vector<uint16_t> row(width);
    int wrong = 0;
    for (int r = 0; r < streams; ++r) {
        ggml_backend_tensor_get(result, row.data(), (r * stride + 9) * width * sizeof(uint16_t), width * sizeof(uint16_t));
        uint16_t expected = r == 0 ? 0 : ggml_fp32_to_fp16(float(r));
        int matches = 0;
        for (int c = 0; c < width; ++c) {
            const uint16_t target = r == 0 ? 0 : ggml_fp32_to_fp16(float(r) + c / 1024.0f);
            matches += row[c] == target;
        }
        wrong += matches != width;
        std::printf("{\"stream\":%d,\"expected_first_half\":%u,\"actual_first_half\":%u,\"matched_first_half\":%d,\"expected_word_count\":512,\"matched_words\":%d}\n",
                    r, expected, row[0], row[0] == expected, matches);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return wrong ? 6 : 0;
}
