import os
import time
import json
import argparse
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort
import subprocess

def cuda_sync():
    cuda.Context.synchronize()

def torch_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def percentile(a, p):
    return float(np.percentile(a, p))

def summarize(times_ms):
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": percentile(arr, 50),
        "p90_ms": percentile(arr, 90),
        "p99_ms": percentile(arr, 99),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "iters": int(arr.size),
    }

def try_nvidia_smi():
    # Best-effort GPU telemetry snapshot (no hard fail if unavailable)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return out
    except Exception as e:
        return f"<nvidia-smi unavailable: {e}>"

def trt_load_engine(plan_path):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")
    return engine

def trt_get_io_names(engine):
    inp = out = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inp = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out = name
    if inp is None or out is None:
        raise RuntimeError(f"Could not resolve IO tensors (inp={inp}, out={out})")
    return inp, out

def trt_engine_expected_shape(engine, inp_name):
    # Returns a tuple of ints (can include -1 for dynamic). If TRT can't provide, return None.
    try:
        shp = engine.get_tensor_shape(inp_name)
        return tuple(int(d) for d in shp)
    except Exception:
        return None

def trt_shape_compatible(expected, requested):
    # expected can contain -1 (dynamic). If expected is None, assume compatible.
    if expected is None:
        return True
    if len(expected) != len(requested):
        return False
    for e, r in zip(expected, requested):
        if e != -1 and e != r:
            return False
    return True

def bench_trt(plan_path, x_fp16, iters=500, warmup=100):
    """
    TRT 10+ safe benchmark:
      - Uses IO tensor API (set_input_shape, set_tensor_address, execute_async_v3)
      - Times end-to-end with CUDA stream synchronize for correctness
    """
    engine = trt_load_engine(plan_path)
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("Failed to create execution context")

    inp_name, out_name = trt_get_io_names(engine)

    B, S, H = x_fp16.shape
    ctx.set_input_shape(inp_name, (B, S, H))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    if any(d < 0 for d in out_shape):
        raise RuntimeError(f"Unresolved output shape: {out_shape}")

    # Allocate device buffers
    y_fp16 = np.empty(out_shape, dtype=np.float16)

    d_in = cuda.mem_alloc(x_fp16.nbytes)
    d_out = cuda.mem_alloc(y_fp16.nbytes)
    stream = cuda.Stream()

    # Copy input once
    cuda.memcpy_htod_async(d_in, x_fp16, stream)
    stream.synchronize()

    ctx.set_tensor_address(inp_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))

    # Warmup
    for _ in range(warmup):
        ok = ctx.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False during warmup")
    stream.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ok = ctx.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False during timed run")
        stream.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    # Copy output once (sanity)
    cuda.memcpy_dtoh_async(y_fp16, d_out, stream)
    stream.synchronize()

    return times, y_fp16

def ort_expected_shape(onnx_path):
    # Returns expected static dims if available; dims can be None/'None'/0/-1 for dynamic.
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shp = []
        for d in inp.shape:
            if d is None:
                shp.append(-1)
            elif isinstance(d, str):
                shp.append(-1)
            else:
                try:
                    shp.append(int(d))
                except Exception:
                    shp.append(-1)
        return tuple(shp)
    except Exception:
        return None

def ort_shape_compatible(expected, requested):
    if expected is None:
        return True
    if len(expected) != len(requested):
        return False
    for e, r in zip(expected, requested):
        if e != -1 and e != r:
            return False
    return True

def bench_onnx(onnx_path, x_fp32, provider, iters=300, warmup=50):
    # Provide fallback to CPUExecutionProvider if provider fails
    sess = ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        sess.run(None, {inp_name: x_fp32})

    # Sync only if we’re actually on GPU provider
    if "CUDA" in provider or "Tensorrt" in provider:
        torch_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: x_fp32})
        if "CUDA" in provider or "Tensorrt" in provider:
            torch_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times

def estimate_ops(B, S, H, heads=4, mlp_mult=4):
    """
    Rough FLOP model for your layer:
      QKV: 3 * (B*S*H*H*2)
      Attn scores: (B*heads*S*S*head_dim*2)
      Attn ctx:    (B*heads*S*S*head_dim*2)
      Out proj:    (B*S*H*H*2)
      MLP fc1:     (B*S*H*(H*mlp_mult)*2)
      MLP fc2:     (B*S*(H*mlp_mult)*H*2)
    """
    Hd = H // heads
    Hff = H * mlp_mult

    qkv = 3.0 * (B * S * H * H * 2.0)
    scores = (B * heads * S * S * Hd * 2.0)
    ctx = (B * heads * S * S * Hd * 2.0)
    outp = (B * S * H * H * 2.0)
    fc1 = (B * S * H * Hff * 2.0)
    fc2 = (B * S * Hff * H * 2.0)

    return qkv + scores + ctx + outp + fc1 + fc2

def add_perf_metrics(summary, B, S, H, ops):
    mean_s = summary["mean_ms"] / 1000.0
    if mean_s <= 0:
        return summary
    summary = dict(summary)
    summary["effective_GFLOPs"] = float((ops / mean_s) / 1e9)
    summary["elements_per_s"] = float((B * S * H) / mean_s)
    return summary

def resolve_artifact_file(artifact_dir, candidates, kind):
    for name in candidates:
        p = os.path.join(artifact_dir, name)
        if os.path.isfile(p):
            return p
    # Fallback: search by extension in artifact_dir
    ext = ".onnx" if kind == "onnx" else ".plan"
    matches = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith(ext)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer a file that contains the layer name if present
        for m in matches:
            base = os.path.basename(m)
            if "transformer_layer_v1" in base:
                return m
        return matches[0]
    raise FileNotFoundError(f"Could not find {kind} file in artifact_dir={artifact_dir}. Tried: {candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--iters_trt", type=int, default=500)
    ap.add_argument("--warmup_trt", type=int, default=100)
    ap.add_argument("--iters_ort", type=int, default=300)
    ap.add_argument("--warmup_ort", type=int, default=50)
    ap.add_argument("--sweep", default="1x8x32,1x64x32,1x256x32,4x256x32,8x256x32,1x512x32")
    args = ap.parse_args()

    art = args.artifact_dir

    plan = resolve_artifact_file(
        art,
        candidates=["transformer_layer_v1.plan", "transformer_layer_v1.engine", "engine.plan", "model.plan"],
        kind="plan"
    )
    onnx_path = resolve_artifact_file(
        art,
        candidates=["transformer_layer_v1.onnx", "model.onnx", "export.onnx"],
        kind="onnx"
    )

    report = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_dir": art,
        "plan": plan,
        "onnx": onnx_path,
        "trt_version": trt.__version__,
        "ort_available_providers": ort.get_available_providers(),
        "gpu_snapshot": try_nvidia_smi(),
        "runs": []
    }

    # Preflight: determine what shapes the artifact/model actually supports
    # (If static, we will SKIP mismatched shapes instead of crashing.)
    trt_engine = trt_load_engine(plan)
    trt_inp_name, _ = trt_get_io_names(trt_engine)
    trt_expected = trt_engine_expected_shape(trt_engine, trt_inp_name)

    ort_expected = ort_expected_shape(onnx_path)

    report["trt_expected_input_shape"] = list(trt_expected) if trt_expected is not None else None
    report["ort_expected_input_shape"] = list(ort_expected) if ort_expected is not None else None

    shapes = []
    for token in args.sweep.split(","):
        token = token.strip()
        if not token:
            continue
        b, s, h = token.split("x")
        shapes.append((int(b), int(s), int(h)))

    for (B, S, H) in shapes:
        print(f"\n=== SHAPE {B}x{S}x{H} ===")

        requested = (B, S, H)

        trt_ok = trt_shape_compatible(trt_expected, requested)
        ort_ok = ort_shape_compatible(ort_expected, requested)

        # deterministic input per shape
        rng = np.random.default_rng(72)
        x = rng.standard_normal((B, S, H), dtype=np.float32)
        x_fp16 = x.astype(np.float16)

        ops = estimate_ops(B, S, H)

        shape_result = {
            "shape": [B, S, H],
            "ops_estimate": float(ops),
            "compatibility": {
                "trt_expected": list(trt_expected) if trt_expected is not None else None,
                "ort_expected": list(ort_expected) if ort_expected is not None else None,
                "trt_compatible": bool(trt_ok),
                "ort_compatible": bool(ort_ok),
            },
            "results": {}
        }

        if not trt_ok:
            msg = "SKIPPED: TensorRT engine has static shape incompatible with requested shape"
            print(msg)
            shape_result["results"]["TensorRT"] = {"status": "SKIPPED", "reason": msg}
        else:
            print("Benchmarking TRT...")
            trt_times, _ = bench_trt(plan, x_fp16, iters=args.iters_trt, warmup=args.warmup_trt)
            trt_sum = add_perf_metrics(summarize(trt_times), B, S, H, ops)
            trt_sum["status"] = "OK"
            shape_result["results"]["TensorRT"] = trt_sum

        if not ort_ok:
            msg = "SKIPPED: ONNX model has static shape incompatible with requested shape"
            print(msg)
            shape_result["results"]["ORT_CUDA"] = {"status": "SKIPPED", "reason": msg}
            shape_result["results"]["ORT_TRT_EP"] = {"status": "SKIPPED", "reason": msg}
        else:
            print("Benchmarking ORT CUDA...")
            ort_cuda_times = bench_onnx(
                onnx_path,
                x.astype(np.float32),
                "CUDAExecutionProvider",
                iters=args.iters_ort,
                warmup=args.warmup_ort
            )
            ort_cuda_sum = add_perf_metrics(summarize(ort_cuda_times), B, S, H, ops)
            ort_cuda_sum["status"] = "OK"
            shape_result["results"]["ORT_CUDA"] = ort_cuda_sum

            print("Benchmarking ORT TensorRT EP...")
            ort_trt_times = bench_onnx(
                onnx_path,
                x.astype(np.float32),
                "TensorrtExecutionProvider",
                iters=args.iters_ort,
                warmup=args.warmup_ort
            )
            ort_trt_sum = add_perf_metrics(summarize(ort_trt_times), B, S, H, ops)
            ort_trt_sum["status"] = "OK"
            shape_result["results"]["ORT_TRT_EP"] = ort_trt_sum

        report["runs"].append(shape_result)

        print(json.dumps(shape_result["results"], indent=2))

    out_json = os.path.join(art, f"bench_report_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote report:", out_json)

if __name__ == "__main__":
    main()