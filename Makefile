LLAMA ?= ../bonsai-hip
HIPCC := hipcc
ARCH  := gfx1151
# Compile-time axes an engineer is pricing, passed through to every translation unit:
#   make DEFS='-DHALO_ROWS_TRIM=1'      a build that fits a bounded tool call, sliced route only
#   make DEFS='-DHALO_MV_ACC=2'         any other -D the kernels read
# kernels/halo_rows.hip documents HALO_ROWS_TRIM; it changes which instantiations exist, so a
# trimmed object and a full one must not be mixed in one link.
#
# Which is why the define set is a dependency of every object. `-MMD` tracks headers and nothing
# tracked this: `make DEFS=-DX` followed by `make` rebuilt only the sources that had changed since,
# and linked them against objects compiled under the other define set. When the defines change a
# struct's layout - a params field behind an #if, a different instantiation list - the result is an
# executable that launches kernels reading the host's fields at the wrong offsets, which arrives as
# a device page fault in whatever kernel runs first and points at nothing. docs/rows-grid-occupancy.md
# tells engineers to `rm kernels/halo_rows.o` by hand for this reason, and an engineer who forgot
# lost a measured result to it (docs/drafter-window-units.md). `.build-defs` holds the define set
# the objects in this tree were built under and is rewritten only when it changes, so a changed -D
# set rebuilds everything and an unchanged one costs nothing.
DEFS ?=
CXXFLAGS := -MMD -O3 -std=c++17 -Wall -Wno-unused-function -Isrc -Ikernels -I$(LLAMA)/include -I$(LLAMA)/ggml/include $(DEFS)
LDFLAGS := -L$(LLAMA)/build-hip/bin -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA)/build-hip/bin -lpthread

SRCS := src/gguf.cpp src/repack.cpp src/q8.cpp src/engine.cpp src/batch.cpp src/tokenizer.cpp src/chat.cpp src/serve_batch.cpp src/server.cpp src/main.cpp vendor/cpp-httplib/httplib.cpp
KSRC := kernels/halo_kernels.hip kernels/halo_rows.hip kernels/halo_draft.hip kernels/ffn_batch.hip kernels/sequence_batch.hip kernels/head_batch.hip kernels/prep_batch.hip kernels/attn_batch.hip
OBJS := $(SRCS:.cpp=.o) $(KSRC:.hip=.o)

all: bonsai-halo

tools/perplexity: tools/perplexity.o $(filter-out src/main.o,$(OBJS))
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

tools/perplexity.o: .build-defs

tools/speculative_compare: tools/speculative_compare.o tools/speculative_recycle.o $(filter-out src/main.o,$(OBJS))
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

tools/speculative_compare.o tools/speculative_recycle.o: .build-defs

.build-defs: FORCE
	@[ -f $@ ] && [ "$$(cat $@)" = '$(DEFS)' ] || { printf '%s' '$(DEFS)' > $@; echo "build defines changed to '$(DEFS)': rebuilding every object"; }
FORCE:
$(OBJS) tools/batch_compare.o tools/batch_profile.o tools/serve_capture_check.o: .build-defs

bonsai-halo: $(OBJS)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -c $< -o $@

%.o: %.hip
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -c $< -o $@

tools/ffn_pack_check: tools/ffn_pack_check.cpp
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

tools/ffn_a4_pack_check: kernels/ffn_a4_pack_check.cpp kernels/ffn_a4_operands.hpp kernels/ffn_operands.hpp src/halo_format.h
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the shared ternary operand maps: the deployed peel, the perm gather the
# matvec runs, and the spread two-bit coordinate, all against halo::decode_block. No GPU.
kernels/head_op_check: kernels/head_op_check.cpp kernels/halo_expand.hpp src/halo_format.h
	g++ -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the sequence projections' three stored code orders, against the weights
# a deployed word names rather than against each other. No GPU.
kernels/seq_op_check: kernels/seq_op_check.cpp kernels/sequence_operands.hpp kernels/ffn_a4_operands.hpp kernels/ffn_operands.hpp
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the windowed attention unit list: that a chunk the window excludes is
# empty in the score body's own visibility arithmetic, and that folding its (-inf, 0, 0) partial is
# the identity bit for bit. The drafter cannot show this on the device - its K-split matvecs drain
# with atomicAdd and its float state is not reproducible run to run. No GPU.
kernels/attn_window_check: kernels/attn_window_check.cpp kernels/phases.hpp kernels/halo_kernels.h
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host model of the block loop's token maps: that each is a bijection onto the same tokens, and
# how many cache lines one fragment instruction's sixteen lanes straddle. No GPU.
kernels/seq_tmap_check: kernels/seq_tmap_check.cpp
	g++ -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the IU4 ternary operand map; runs the kernel's own source, no GPU
kernels/mv_nibble_check: kernels/mv_nibble_check.cpp kernels/mv_nibble.hpp src/halo_format.h
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the eight-row matrix body's incremental operand expansion; no GPU
kernels/mv8_expand_check: kernels/mv8_expand_check.cpp kernels/halo_expand.hpp src/halo_format.h
	g++ -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the single-token palette operand map; needs no GPU
kernels/single_map_check: kernels/single_map_check.cpp src/halo_format.h
	g++ -O2 -std=c++17 -Isrc -Ikernels -o $@ $<

# host exactness probe for the Q4 drafter coordinate's placement, operand map and bias; no GPU
kernels/q4_pack_check: kernels/q4_pack_check.cpp kernels/q4_format.h src/q8.cpp src/halo_format.h
	g++ -O2 -std=c++17 -Isrc -Ikernels -o $@ kernels/q4_pack_check.cpp src/q8.cpp -lpthread

# build a drafter weight cache without a GPU, so a measured process starts warm
# what grid the runtime will actually give the persistent kernel, per register budget
# what the device really co-schedules, against what the two predictions in coop_grid claim.
# Neither of these needs a model, a lock or more than two seconds of build: occ_surface launches
# nothing at all and occ_resident launches a kernel with no grid sync, so it cannot deadlock.
bench/occ_resident: bench/occ_resident.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

tools/occ_surface: tools/occ_surface.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

# HALO_ROWS_PROBE cuts the host launchers below `k_forward_rows`, but not the resident-state one
# above it, which reads the sequence stage's activation coordinate - so the probe links the real
# `sequence_quant` rather than a stub that could drift from it. `-x none` is what stops hipcc
# compiling the object as HIP source, since a .hip input sets the language for the whole line.
tools/rows_occ_probe: tools/rows_occ_probe.hip kernels/halo_rows.hip kernels/phases.hpp kernels/halo_kernels.h kernels/prep_batch.o
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -Isrc -Ikernels -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $< -x none kernels/prep_batch.o

# what the runtime co-schedules for each recurrent state instantiation
tools/gdn_occ_probe: tools/gdn_occ_probe.hip kernels/halo_rows.hip kernels/phases.hpp kernels/halo_kernels.h kernels/gdn_state_codec.hpp
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -Isrc -Ikernels -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $<

# which way row_ror rotates, and therefore which of warp_sum's two roundings a lane holds
tools/gdn_warpsum_probe: tools/gdn_warpsum_probe.hip kernels/device.hpp kernels/halo_kernels.h
	$(HIPCC) --offload-arch=$(ARCH) -O2 -std=c++17 -Isrc -Ikernels -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $<

tools/draft_quant: tools/draft_quant.cpp src/q8.cpp kernels/q4_format.h kernels/halo_kernels.h
	$(HIPCC) -O2 -std=c++17 -Isrc -Ikernels -o $@ tools/draft_quant.cpp src/q8.cpp -lpthread

# does one lane reproduce a 32-lane warp_sum, bit for bit? needs a GPU, no model, no lock
kernels/attn_tree_check: kernels/attn_tree_check.hip kernels/attn_leaf.hpp kernels/device.hpp kernels/halo_kernels.h
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -Isrc -Ikernels -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $<

tools/profile_read_check: tools/profile_read_check.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -o $@ $<

tools/batch_profile: tools/batch_profile.o $(filter-out src/main.o,$(OBJS))
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

tools/batch_compare: tools/batch_compare.o $(filter-out src/main.o,$(OBJS))
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

tools/serve_capture_check: tools/serve_capture_check.o $(filter-out src/main.o,$(OBJS))
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $^ $(LDFLAGS)

bench/coop_cost: bench/coop_cost.hip kernels/phases.hpp kernels/device.hpp
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -Isrc -Ikernels -o $@ $<

bench/bw: bench/bw.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -o $@ $<

# which counter the multi-row matvec waits on, and what an issue slot in that loop is worth:
# achieved GB/s per activation path, per expansion, per resident waves. `HALO_MV_EXPAND_PROBE`
# compiles the expansion-free ablation, which computes a wrong answer on purpose and exists only
# here - no runtime translation unit defines it.
bench/mvsched: bench/mvsched.hip kernels/device.hpp kernels/halo_expand.hpp kernels/halo_kernels.h
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -DHALO_MV_EXPAND_PROBE -Isrc -Ikernels -o $@ $<

# weight-stream order: achieved GB/s against concurrent stream count and run length
bench/wstream: bench/wstream.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

# what a matrix instruction costs a SIMD32, so a block census can be converted into time
bench/wmma_cost: bench/wmma_cost.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

# what a spinning host thread costs the shader clock, on a package the CPU shares
bench/power_probe: bench/power_probe.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

# what a DEPENDENT matrix chain costs, which is the half of that constant a census cannot see
bench/wmma_chain: bench/wmma_chain.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

# the same instruction in the shape the FFN slice issues it, with the shader clock read inside the
# kernel so a busy socket cannot move the constant
bench/mvblock: bench/mvblock.hip
	$(HIPCC) --offload-arch=$(ARCH) -O3 -std=c++17 -o $@ $<

-include $(OBJS:.o=.d) tools/batch_compare.d tools/batch_profile.d tools/serve_capture_check.d tools/speculative_compare.d tools/speculative_recycle.d

clean:
	# rocprofv3 writes .rocprofv3/ into the working directory of any counter run, and it is
	# ignored rather than tracked, so agent-workspace release classifies a checkout that has
	# taken a counter panel as repair-required until it goes.
	rm -rf .rocprofv3
	rm -f .build-defs tools/speculative_compare tools/speculative_compare.o tools/speculative_compare.d tools/speculative_recycle.o tools/speculative_recycle.d
	rm -f bench/mvsched bench/occ_resident tools/occ_surface bonsai-halo-arm0 tools/profile_read_check tools/batch_profile tools/batch_profile.o tools/batch_profile.d $(OBJS) $(OBJS:.o=.d) bonsai-halo tools/batch_compare tools/batch_compare.o tools/batch_compare.d tools/serve_capture_check tools/serve_capture_check.o tools/serve_capture_check.d tools/ffn_pack_check tools/ffn_a4_pack_check kernels/single_map_check kernels/mv_nibble_check kernels/mv8_expand_check kernels/head_op_check kernels/seq_op_check kernels/seq_tmap_check kernels/attn_window_check kernels/q4_pack_check kernels/attn_tree_check tools/draft_quant tools/rows_occ_probe tools/gdn_occ_probe tools/gdn_warpsum_probe
	# Same rule, same reason, for the bench probes: they are ignored rather than tracked, none
	# of them is in OBJS, and a checkout that has run one is repair-required until it goes.
	# `kernels/halo_forward.{o,d}` outlives a source file this tree no longer has - the object
	# survives every build because nothing references it, and it travels with any copy of a
	# built tree.
	rm -f bench/bw bench/mv bench/mvrows bench/wstream bench/coop_cost bench/wmma_cost bench/wmma_chain bench/mvblock bench/power_probe kernels/halo_forward.o kernels/halo_forward.d
	# The state acceptance binary lives in its own directory with its own rule, and
	# `agent-workspace release` reports a checkout holding it as repair-required, so the
	# clean the lane README tells everyone to run before releasing has to reach it.
	$(MAKE) -C tools/direct-commit clean

.PHONY: all clean
