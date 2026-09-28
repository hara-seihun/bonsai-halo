# Host-only checks for the direct-commit GDN contract, plus the native acceptance tool that needs
# the engine. Kept out of the root Makefile, which main owns; nothing here changes the inference
# build. The engine objects come from the root Makefile's own pattern rules, so this file adds no
# second set of compiler flags to keep in step.
#
#   make -C tools/direct-commit            host probe only, no GPU, no HIP
#   make -C tools/direct-commit check      build and run the host probe
#   make -C tools/direct-commit sequence_state_check     the native mixed-state acceptance tool
#
# sequence_state_check loads the real model and runs on the GPU; building it links the whole
# engine, so the first build is as slow as a root build and later ones reuse those objects.

ROOT := ../..
LLAMA := ../bonsai-hip
HIPCC := hipcc
ARCH := gfx1151
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

# Both lists are READ FROM THE ROOT MAKEFILE rather than copied, because a copy drifts and the
# thing it breaks is the state acceptance of every change to the recurrence. It drifted twice:
# `attn_batch.hip` arrived in KSRC and not here (eight undefined `halo::attn_batch_*`), then
# `src/serve_batch.cpp` arrived in SRCS and not here (five undefined `halo::ServeBatch::*`), both
# times leaving `make sequence_state_check` unlinkable for an unrelated reason. The root Makefile
# keeps each list on one line; if that ever changes, this reads nothing and the link says so
# immediately rather than quietly omitting a translation unit.
ENGINE_SRCS := $(filter-out src/main.cpp,$(shell sed -n 's/^SRCS := //p' $(ROOT)/Makefile))
KERNEL_SRCS := $(shell sed -n 's/^KSRC := //p' $(ROOT)/Makefile)
ENGINE_OBJS := $(ENGINE_SRCS:.cpp=.o) $(KERNEL_SRCS:.hip=.o)

HIPFLAGS := -MMD -O3 -std=c++17 -Wall -Wno-unused-function -I$(ROOT)/src -I$(ROOT)/kernels \
            -I$(LLAMA)/include -I$(LLAMA)/ggml/include
LDFLAGS := -L$(LLAMA)/build-hip/bin -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA)/build-hip/bin -lpthread

all: commit_equiv

commit_equiv: commit_equiv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

check: commit_equiv
	./commit_equiv

sequence_state_check: sequence_state_check.o
	$(MAKE) -C $(ROOT) $(ENGINE_OBJS)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $< $(addprefix $(ROOT)/,$(ENGINE_OBJS)) $(LDFLAGS)

sequence_state_check.o: $(ROOT)/tools/sequence_state_check.cpp
	$(HIPCC) --offload-arch=$(ARCH) $(HIPFLAGS) -c $< -o $@

clean:
	rm -f commit_equiv sequence_state_check sequence_state_check.o sequence_state_check.d

-include sequence_state_check.d

.PHONY: all check clean
