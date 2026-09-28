#!/usr/bin/env bash
# Build the full-model batch comparison driver. The top-level Makefile owns the target; this just
# runs it from anywhere in the tree.
#
#   tools/batch-compare/build.sh
#   LLAMA=/path/to/llama.cpp tools/batch-compare/build.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
make tools/batch_compare "$@"
echo "built tools/batch_compare"
