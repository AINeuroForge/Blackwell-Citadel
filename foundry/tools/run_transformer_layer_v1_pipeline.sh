#!/usr/bin/env bash
set -euo pipefail

echo "=== BUILD (TRT-LLM PROFILE) ==="
source ~/citadel/foundry/profiles/trtllm/bin/activate
python ~/citadel/foundry/tools/build_trtllm_transformer_layer_v1.py
deactivate

# Select newest artifact after build
ART=$(ls -1dt ~/citadel/foundry/artifacts/transformer_layer_v1/* | head -n 1)
echo "Using artifact: $ART"

# Enforce artifact contract
for f in transformer_layer_v1.plan weights.npz x.npy meta.json; do
  if [[ ! -f "$ART/$f" ]]; then
    echo "ERROR: Missing required artifact file: $ART/$f"
    exit 2
  fi
done

echo "=== TRT INFERENCE ==="
source ~/citadel/foundry/profiles/trtllm/bin/activate
python ~/citadel/foundry/tools/trtllm_infer_dump_transformer_v1.py --artifact_dir "$ART"
deactivate

echo "=== VISION ONNX BUILD ==="
source ~/citadel/foundry/profiles/vision/bin/activate
python ~/citadel/foundry/tools/vision_build_and_validate_transformer_v1.py --artifact_dir "$ART"
deactivate

echo "=== MODELING COMPARE ==="
source ~/citadel/foundry/profiles/modeling/bin/activate
python ~/citadel/foundry/tools/compare_modeling_transformer_v1.py --artifact_dir "$ART" --tol 0.20
deactivate

echo "=== PIPELINE COMPLETE ==="