#!/usr/bin/env bash
set -euo pipefail

ROOT=~/citadel
TOOLS=$ROOT/foundry/tools
ART_DIR=$ROOT/foundry/artifacts/transformer_layer_v1
RESULTS_DIR=$ROOT/foundry/results

mkdir -p "$RESULTS_DIR"

H=32
B=1
SEQS=(8 16 32 64 128 256 512)

echo "===== TRANSFORMER LAYER V1 SWEEP START ====="

for S in "${SEQS[@]}"; do
  echo ""
  echo "========== SHAPE ${B}x${S}x${H} =========="

  # ---- Build TRT Engine ----
  source $ROOT/foundry/profiles/trtllm/bin/activate
  python $TOOLS/build_trtllm_transformer_layer_v1.py --B $B --S $S --H $H
  deactivate

  ART=$(ls -td $ART_DIR/* | head -1)
  echo "Using artifact: $ART"

  # ---- TRT Inference ----
  source $ROOT/foundry/profiles/trtllm/bin/activate
  python $TOOLS/trtllm_infer_dump_transformer_v1.py --artifact_dir "$ART"
  deactivate

  # ---- Vision ONNX Export ----
  source $ROOT/foundry/profiles/vision/bin/activate
  python $TOOLS/vision_build_and_validate_transformer_v1.py --artifact_dir "$ART"
  deactivate

  # ---- Numerical Validation ----
  source $ROOT/foundry/profiles/modeling/bin/activate
  python $TOOLS/compare_modeling_transformer_v1.py --artifact_dir "$ART" --tol 0.2
  deactivate

  # ---- Benchmark ----
  source $ROOT/foundry/profiles/trtllm/bin/activate
  python $TOOLS/benchmark_transformer_layer_v1.py --artifact_dir "$ART" \
      > "$RESULTS_DIR/bench_${B}x${S}x${H}.json"
  deactivate

done

echo ""
echo "===== SWEEP COMPLETE ====="