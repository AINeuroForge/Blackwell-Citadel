#!/usr/bin/env python3
import os, json, hashlib, time
from pathlib import Path
import numpy as np
import tensorrt as trt
from tensorrt_llm.builder import Builder

ROOT = Path.home() / "citadel" / "foundry"
CONTRACT_PATH = ROOT / "CONTRACT_v1.json"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    contract = json.loads(CONTRACT_PATH.read_text())

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    outdir = ROOT / "artifacts" / "minimal_attention_v1" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # --- deterministic weights
    rng = np.random.default_rng(72)

    # Contract dims
    B, S, H = 1, 8, 32
    n_heads = 4
    head_dim = H // n_heads

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = Builder()
    config = builder.create_builder_config(precision="float16")
    network = builder.create_network()
    trtnet = network.trt_network

    # Input
    x = trtnet.add_input("x", trt.DataType.HALF, (B, S, H))

    # Linear helper: [B,S,H] -> [B*S,H] @ [H,H] -> [B,S,H]
    def linear(inp, w):
        sh1 = trtnet.add_shuffle(inp)
        sh1.reshape_dims = (B*S, H)
        w_const = trtnet.add_constant((H, H), trt.Weights(w))
        mm = trtnet.add_matrix_multiply(sh1.get_output(0), trt.MatrixOperation.NONE,
                                        w_const.get_output(0), trt.MatrixOperation.NONE)
        sh2 = trtnet.add_shuffle(mm.get_output(0))
        sh2.reshape_dims = (B, S, H)
        return sh2.get_output(0)

    Wq = rng.standard_normal((H, H)).astype(np.float16)
    Wk = rng.standard_normal((H, H)).astype(np.float16)
    Wv = rng.standard_normal((H, H)).astype(np.float16)
    Wo = rng.standard_normal((H, H)).astype(np.float16)

    q = linear(x, Wq)
    k = linear(x, Wk)
    v = linear(x, Wv)

    # Split heads: [B,S,H] -> [B,Heads,S,HeadDim]
    def split_heads(t):
        sh_a = trtnet.add_shuffle(t)
        sh_a.reshape_dims = (B, S, n_heads, head_dim)
        sh_b = trtnet.add_shuffle(sh_a.get_output(0))
        sh_b.second_transpose = (0, 2, 1, 3)
        return sh_b.get_output(0)

    qh = split_heads(q)
    kh = split_heads(k)
    vh = split_heads(v)

    scores = trtnet.add_matrix_multiply(qh, trt.MatrixOperation.NONE,
                                        kh, trt.MatrixOperation.TRANSPOSE)

    scale = np.array([1.0/np.sqrt(head_dim)], dtype=np.float16).reshape(1,1,1,1)
    scale_const = trtnet.add_constant((1,1,1,1), trt.Weights(scale))
    scaled = trtnet.add_elementwise(scores.get_output(0),
                                    scale_const.get_output(0),
                                    trt.ElementWiseOperation.PROD)

    sm = trtnet.add_softmax(scaled.get_output(0))
    sm.axes = 1 << 3

    ctx = trtnet.add_matrix_multiply(sm.get_output(0), trt.MatrixOperation.NONE,
                                     vh, trt.MatrixOperation.NONE)

    # Merge heads robustly: transpose -> 2D -> 3D
    sh_t = trtnet.add_shuffle(ctx.get_output(0))
    sh_t.second_transpose = (0, 2, 1, 3)

    sh_2d = trtnet.add_shuffle(sh_t.get_output(0))
    sh_2d.reshape_dims = (B*S, H)

    sh_3d = trtnet.add_shuffle(sh_2d.get_output(0))
    sh_3d.reshape_dims = (B, S, H)
    attn_out = sh_3d.get_output(0)

    # Output projection + residual: y = x + (attn_out @ Wo)
    proj = linear(attn_out, Wo)
    add = trtnet.add_elementwise(x, proj, trt.ElementWiseOperation.SUM)
    y = add.get_output(0)
    y.name = "y"
    trtnet.mark_output(y)
    y.dtype = trt.DataType.HALF

    # Build engine
    engine_bytes = builder.build_engine(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    plan_path = outdir / "engine.plan"
    plan_path.write_bytes(engine_bytes)

    # Save weights for ONNX build/compare parity downstream
    np.savez(outdir / "weights.npz", Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)

    # Export ONNX (minimal: write a “contract ONNX” placeholder)
    # NOTE: True TRT->ONNX export is not native; we will create an ONNX reference model in vision rail using weights.
    # The ONNX will be built from weights in vision rail to enforce interoperability.
    (outdir / "model.onnx").write_text("ONNX_BUILT_IN_VISION_RAIL_FROM_weights.npz\n")

    # Emit run manifest + hashes
    artifacts = {
        "timestamp_utc": now_utc(),
        "contract": contract["version"],
        "artifact_dir": str(outdir),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "weights": {"path": str(outdir / "weights.npz"), "sha256": sha256_file(outdir / "weights.npz")},
        "onnx": {"path": str(outdir / "model.onnx"), "sha256": sha256_file(outdir / "model.onnx")},
        "shape": {"B": B, "S": S, "H": H, "heads": n_heads, "head_dim": head_dim},
        "tensorrt": trt.__version__,
    }
    (outdir / "contract.json").write_text(json.dumps(contract, indent=2))
    (outdir / "run_manifest.json").write_text(json.dumps(artifacts, indent=2))

    print("Wrote artifacts to:", outdir)

if __name__ == "__main__":
    main()
