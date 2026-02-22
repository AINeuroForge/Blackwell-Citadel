import os
import json
import time
import hashlib
import argparse
import numpy as np

import tensorrt as trt
from tensorrt_llm.builder import Builder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def utc_ts():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def add_const(trtnet, name, arr):
    c = trtnet.add_constant(arr.shape, trt.Weights(arr))
    c.name = name
    return c.get_output(0)

def linear_bsh(trtnet, x_bsh, W_hh, B_h, B, S, H, name):
    outH = W_hh.shape[1]

    sh1 = trtnet.add_shuffle(x_bsh)
    sh1.reshape_dims = (B * S, H)
    sh1.name = f"{name}_flatten"

    w = add_const(trtnet, f"{name}_W", W_hh)
    b = add_const(trtnet, f"{name}_b", B_h.reshape(1, -1))

    mm = trtnet.add_matrix_multiply(
        sh1.get_output(0), trt.MatrixOperation.NONE,
        w, trt.MatrixOperation.NONE
    )
    mm.name = f"{name}_mm"

    addb = trtnet.add_elementwise(mm.get_output(0), b, trt.ElementWiseOperation.SUM)
    addb.name = f"{name}_bias"

    sh2 = trtnet.add_shuffle(addb.get_output(0))
    sh2.reshape_dims = (B, S, outH)
    sh2.name = f"{name}_unflatten"

    return sh2.get_output(0)

def split_heads(trtnet, x_bsh, B, S, H, n_heads, head_dim, name):
    sh = trtnet.add_shuffle(x_bsh)
    sh.reshape_dims = (B, S, n_heads, head_dim)
    sh.second_transpose = (0, 2, 1, 3)
    sh.name = name
    return sh.get_output(0)

def merge_heads(trtnet, x_bhsd, B, S, H, name):
    sh1 = trtnet.add_shuffle(x_bhsd)
    sh1.second_transpose = (0, 2, 1, 3)
    sh1.name = f"{name}_transpose"

    sh2 = trtnet.add_shuffle(sh1.get_output(0))
    sh2.reshape_dims = (B, S, H)
    sh2.name = f"{name}_reshape"

    return sh2.get_output(0)

def mul_const_broadcast(trtnet, x, scalar, name, broadcast_ndims):
    arr = np.array([scalar], dtype=np.float16).reshape((1,) * broadcast_ndims)
    c = add_const(trtnet, f"{name}_c", arr)
    ew = trtnet.add_elementwise(x, c, trt.ElementWiseOperation.PROD)
    ew.name = name
    return ew.get_output(0)

def add_residual(trtnet, a, b, name):
    ew = trtnet.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    ew.name = name
    return ew.get_output(0)

def layernorm_lastdim(trtnet, x_bsh, gamma_h, beta_h, eps, B, S, H, name):
    mean = trtnet.add_reduce(x_bsh, trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    mean.name = f"{name}_mean"

    xm = trtnet.add_elementwise(x_bsh, mean.get_output(0), trt.ElementWiseOperation.SUB)
    xm.name = f"{name}_xm"

    sq = trtnet.add_elementwise(xm.get_output(0), xm.get_output(0), trt.ElementWiseOperation.PROD)
    sq.name = f"{name}_sq"

    var = trtnet.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    var.name = f"{name}_var"

    eps_c = add_const(trtnet, f"{name}_eps", np.array([eps], dtype=np.float16).reshape(1, 1, 1))
    var_eps = trtnet.add_elementwise(var.get_output(0), eps_c, trt.ElementWiseOperation.SUM)
    var_eps.name = f"{name}_var_eps"

    sqrt = trtnet.add_unary(var_eps.get_output(0), trt.UnaryOperation.SQRT)
    sqrt.name = f"{name}_sqrt"

    one = add_const(trtnet, f"{name}_one", np.array([1.0], dtype=np.float16).reshape(1, 1, 1))
    rsqrt = trtnet.add_elementwise(one, sqrt.get_output(0), trt.ElementWiseOperation.DIV)
    rsqrt.name = f"{name}_rsqrt"

    norm = trtnet.add_elementwise(xm.get_output(0), rsqrt.get_output(0), trt.ElementWiseOperation.PROD)
    norm.name = f"{name}_norm"

    gamma = add_const(trtnet, f"{name}_gamma", gamma_h.astype(np.float16).reshape(1, 1, H))
    beta  = add_const(trtnet, f"{name}_beta",  beta_h.astype(np.float16).reshape(1, 1, H))

    scaled = trtnet.add_elementwise(norm.get_output(0), gamma, trt.ElementWiseOperation.PROD)
    scaled.name = f"{name}_scale"

    shifted = trtnet.add_elementwise(scaled.get_output(0), beta, trt.ElementWiseOperation.SUM)
    shifted.name = f"{name}_shift"

    return shifted.get_output(0)

def gelu(trtnet, x, name):
    if hasattr(trt.ActivationType, "GELU"):
        act = trtnet.add_activation(x, trt.ActivationType.GELU)
        act.name = name
        return act.get_output(0)

    x3 = trtnet.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    x3b = trtnet.add_elementwise(x3.get_output(0), x, trt.ElementWiseOperation.PROD)

    c0 = add_const(trtnet, f"{name}_c0", np.array([0.044715], dtype=np.float16).reshape(1, 1, 1))
    t0 = trtnet.add_elementwise(x3b.get_output(0), c0, trt.ElementWiseOperation.PROD)

    t1 = trtnet.add_elementwise(x, t0.get_output(0), trt.ElementWiseOperation.SUM)

    c1 = add_const(trtnet, f"{name}_c1", np.array([0.79788456], dtype=np.float16).reshape(1, 1, 1))
    t2 = trtnet.add_elementwise(t1.get_output(0), c1, trt.ElementWiseOperation.PROD)

    th = trtnet.add_activation(t2.get_output(0), trt.ActivationType.TANH)

    one = add_const(trtnet, f"{name}_one", np.array([1.0], dtype=np.float16).reshape(1, 1, 1))
    t3 = trtnet.add_elementwise(one, th.get_output(0), trt.ElementWiseOperation.SUM)

    half = add_const(trtnet, f"{name}_half", np.array([0.5], dtype=np.float16).reshape(1, 1, 1))
    t4 = trtnet.add_elementwise(x, half, trt.ElementWiseOperation.PROD)

    out = trtnet.add_elementwise(t4.get_output(0), t3.get_output(0), trt.ElementWiseOperation.PROD)
    out.name = name
    return out.get_output(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_mult", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--out_root", type=str, default=os.path.expanduser("~/citadel/foundry/artifacts/transformer_layer_v1"))
    args = ap.parse_args()

    B, S, H = args.B, args.S, args.H
    n_heads = args.heads
    head_dim = H // n_heads
    Hff = H * args.mlp_mult

    np.random.seed(args.seed)

    out_dir = os.path.join(args.out_root, utc_ts())
    ensure_dir(out_dir)

    def w(shape):
        return (np.random.randn(*shape) * 0.02).astype(np.float16)

    weights = {
        "Wq": w((H, H)), "bq": np.zeros((H,), np.float16),
        "Wk": w((H, H)), "bk": np.zeros((H,), np.float16),
        "Wv": w((H, H)), "bv": np.zeros((H,), np.float16),
        "Wo": w((H, H)), "bo": np.zeros((H,), np.float16),
        "ln1_g": np.ones((H,), np.float16), "ln1_b": np.zeros((H,), np.float16),
        "ln2_g": np.ones((H,), np.float16), "ln2_b": np.zeros((H,), np.float16),
        "W1": w((H, Hff)), "b1": np.zeros((Hff,), np.float16),
        "W2": w((Hff, H)), "b2": np.zeros((H,), np.float16),
    }
    weights_path = os.path.join(out_dir, "weights.npz")
    np.savez(weights_path, **weights)

    # Deterministic input sample (artifact contract: x.npy must exist)
    rng = np.random.RandomState(args.seed + 12345)
    x_sample = rng.randn(B, S, H).astype(np.float16)
    x_path = os.path.join(out_dir, "x.npy")
    np.save(x_path, x_sample)

    builder = Builder()
    config = builder.create_builder_config(precision="float16")
    network = builder.create_network()
    trtnet = network.trt_network

    print("Builder + config + network: OK")

    x = trtnet.add_input("x", trt.DataType.HALF, (B, S, H))

    q = linear_bsh(trtnet, x, weights["Wq"], weights["bq"], B, S, H, "q_proj")
    k = linear_bsh(trtnet, x, weights["Wk"], weights["bk"], B, S, H, "k_proj")
    v = linear_bsh(trtnet, x, weights["Wv"], weights["bv"], B, S, H, "v_proj")

    qh = split_heads(trtnet, q, B, S, H, n_heads, head_dim, "q_split")
    kh = split_heads(trtnet, k, B, S, H, n_heads, head_dim, "k_split")
    vh = split_heads(trtnet, v, B, S, H, n_heads, head_dim, "v_split")

    scores = trtnet.add_matrix_multiply(qh, trt.MatrixOperation.NONE, kh, trt.MatrixOperation.TRANSPOSE)
    scores_scaled = mul_const_broadcast(trtnet, scores.get_output(0), 1.0 / np.sqrt(head_dim), "attn_scale", 4)

    sm = trtnet.add_softmax(scores_scaled)
    sm.axes = 1 << 3

    ctx = trtnet.add_matrix_multiply(sm.get_output(0), trt.MatrixOperation.NONE, vh, trt.MatrixOperation.NONE)
    merged = merge_heads(trtnet, ctx.get_output(0), B, S, H, "attn_merge")
    attn_out = linear_bsh(trtnet, merged, weights["Wo"], weights["bo"], B, S, H, "o_proj")

    res1 = add_residual(trtnet, x, attn_out, "residual_1")
    ln1 = layernorm_lastdim(trtnet, res1, weights["ln1_g"], weights["ln1_b"], args.eps, B, S, H, "ln1")

    h1 = linear_bsh(trtnet, ln1, weights["W1"], weights["b1"], B, S, H, "mlp_fc1")
    h1_g = gelu(trtnet, h1, "mlp_gelu")
    h2 = linear_bsh(trtnet, h1_g, weights["W2"], weights["b2"], B, S, Hff, "mlp_fc2")

    res2 = add_residual(trtnet, ln1, h2, "residual_2")
    y = layernorm_lastdim(trtnet, res2, weights["ln2_g"], weights["ln2_b"], args.eps, B, S, H, "ln2")

    trtnet.mark_output(y)
    y.name = "y"
    y.dtype = trt.DataType.HALF

    print("Graph constructed + output marked: OK")

    plan = builder.build_engine(network, config)
    if plan is None:
        raise RuntimeError("Engine build failed")

    plan_path = os.path.join(out_dir, "transformer_layer_v1.plan")
    with open(plan_path, "wb") as f:
        f.write(plan)

    # Deterministic meta + hashes (artifact contract: plan + weights + x.npy + meta)
    meta_path = os.path.join(out_dir, "meta.json")
    try:
        import tensorrt_llm as trtllm  # noqa: F401
        trtllm_ver = getattr(trtllm, "__version__", None)
    except Exception:
        trtllm_ver = None

    meta = {
        "artifact": "transformer_layer_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "B": B,
            "S": S,
            "H": H,
            "heads": n_heads,
            "head_dim": head_dim,
            "mlp_mult": args.mlp_mult,
            "Hff": Hff,
            "eps": float(args.eps),
            "seed": int(args.seed),
        },
        "io": {
            "input_name": "x",
            "input_dtype": "fp16",
            "input_shape": [B, S, H],
            "output_name": "y",
            "output_dtype": "fp16",
            "output_shape": [B, S, H],
        },
        "versions": {
            "tensorrt": getattr(trt, "__version__", None),
            "tensorrt_llm": trtllm_ver,
        },
        "files": {
            "transformer_layer_v1.plan": {"sha256": None, "bytes": None},
            "weights.npz": {"sha256": None, "bytes": None},
            "x.npy": {"sha256": None, "bytes": None},
            "meta.json": {"sha256": None, "bytes": None},
        },
    }

    # Populate hashes/sizes
    for fn in ("transformer_layer_v1.plan", "weights.npz", "x.npy"):
        p = os.path.join(out_dir, fn)
        meta["files"][fn]["sha256"] = sha256_file(p)
        meta["files"][fn]["bytes"] = int(os.stat(p).st_size)

    # Write meta, then hash meta itself
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    meta["files"]["meta.json"]["sha256"] = sha256_file(meta_path)
    meta["files"]["meta.json"]["bytes"] = int(os.stat(meta_path).st_size)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("Wrote artifacts to:", out_dir)

if __name__ == "__main__":
    main()