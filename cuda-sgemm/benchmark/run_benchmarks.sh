#!/usr/bin/env bash
# Run every implemented kernel + cuBLAS at a fixed size and collect GFLOPs.
# Writes benchmark/results.csv. Kernels that are still stubs (exit 3) are
# skipped and reported, so partial progress still produces a valid chart.
#
#   ./benchmark/run_benchmarks.sh [size] [repeats]
set -u

SIZE="${1:-4096}"
REPEATS="${2:-20}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$HERE/sgemm"
OUT="$HERE/benchmark/results.csv"

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found. Run 'make' first (needs an NVIDIA GPU)." >&2
  exit 1
fi

echo "kernel_id,name,size,gflops,max_rel_err,status" > "$OUT"

# Kernel ids come straight from the binary's registry, so this loop never needs
# editing when you add a kernel.
for id in $("$BIN" list); do
  row="$("$BIN" "$id" "$SIZE" "$REPEATS" 2>/dev/null)"
  if [ -n "$row" ]; then
    echo "$row" | tee -a "$OUT"
  else
    echo "kernel $id: skipped (not implemented)" >&2
  fi
done

echo "" >&2
echo "wrote $OUT" >&2
