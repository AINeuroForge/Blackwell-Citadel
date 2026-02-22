#!/usr/bin/env python3
# One-shape-per-process TensorRT benchmark worker (WSL2 stability doctrine)
# - No torch
# - No pycuda.autoinit
# - Explicit CUDA context lifecycle
# - Writes JSON artifact to results/

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


def now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def discover_io(engine: trt.ICudaEngine):
    inp = out = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inp = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out = name
    if inp is None or out is None:
        raise RuntimeError(f"Could not discover I/O tensors (inp={inp}, out={out})")
    return inp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to .plan engine")
    ap.add_argument("--shape", required=True, help="Shape like 1024x1024 or 1x3x224x224")
    ap.add_argument("--profile", type=int, default=0, help="Optimization profile index")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--out", default=None, help="Output JSON path (default: results/auto)")
    ap.add_argument("--log_level", default="ERROR", choices=["ERROR", "WARNING", "INFO", "VERBOSE"])
    args = ap.parse_args()

    # Parse shape
    dims = tuple(int(x) for x in args.shape.lower().replace("x", ",").split(","))
    if len(dims) < 1:
        raise ValueError("Bad --shape")

    # Deterministic host input
    rng = np.random.default_rng(args.seed)

    # Explicit CUDA driver init + explicit context (no autoinit)
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    artifact = {
        "ts_utc": now_utc_iso(),
        "engine": str(Path(args.engine).resolve()),
        "shape": dims,
        "profile": args.profile,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "tensorrt_version": trt.__version__,
        "status": "INIT",
        "errors": [],
    }

    try:
        # TensorRT logger
        level_map = {
            "ERROR": trt.Logger.ERROR,
            "WARNING": trt.Logger.WARNING,
            "INFO": trt.Logger.INFO,
            "VERBOSE": trt.Logger.VERBOSE,
        }
        TRT_LOGGER = trt.Logger(level_map[args.log_level])
        runtime = trt.Runtime(TRT_LOGGER)

        with open(args.engine, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")

        inp_name, out_name = discover_io(engine)

        artifact["io"] = {
            "input": {
                "name": inp_name,
                "dtype": str(engine.get_tensor_dtype(inp_name)),
                "shape": tuple(int(x) for x in engine.get_tensor_shape(inp_name)),
            },
            "output": {
                "name": out_name,
                "dtype": str(engine.get_tensor_dtype(out_name)),
                "shape": tuple(int(x) for x in engine.get_tensor_shape(out_name)),
            },
        }

        exec_ctx = engine.create_execution_context()
        if exec_ctx is None:
            raise RuntimeError("Failed to create execution context")

        # Stream for all work in this process
        stream = cuda.Stream()

        # Select optimization profile (async)
        exec_ctx.set_optimization_profile_async(args.profile, stream.handle)

        # Set dynamic input shape
        exec_ctx.set_input_shape(inp_name, dims)
        out_shape = tuple(int(x) for x in exec_ctx.get_tensor_shape(out_name))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(f"Output shape unresolved after set_input_shape: {out_shape}")

        # Allocate
        inp_dtype = trt.nptype(engine.get_tensor_dtype(inp_name))
        out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))

        x_host = rng.standard_normal(size=dims).astype(inp_dtype, copy=False)
        y_host = np.empty(out_shape, dtype=out_dtype)

        d_in = cuda.mem_alloc(x_host.nbytes)
        d_out = cuda.mem_alloc(y_host.nbytes)

        # Bind addresses
        exec_ctx.set_tensor_address(inp_name, int(d_in))
        exec_ctx.set_tensor_address(out_name, int(d_out))

        # Warmup
        for _ in range(args.warmup):
            cuda.memcpy_htod_async(d_in, x_host, stream)
            ok = exec_ctx.execute_async_v3(stream_handle=stream.handle)
            if not ok:
                raise RuntimeError("execute_async_v3 returned False during warmup")
        stream.synchronize()

        # Timed loop with CUDA events
        start_evt = cuda.Event()
        end_evt = cuda.Event()

        lat_ms = []
        for _ in range(args.iters):
            cuda.memcpy_htod_async(d_in, x_host, stream)
            start_evt.record(stream)
            ok = exec_ctx.execute_async_v3(stream_handle=stream.handle)
            if not ok:
                raise RuntimeError("execute_async_v3 returned False during benchmark")
            end_evt.record(stream)
            end_evt.synchronize()
            lat_ms.append(start_evt.time_till(end_evt))

        # Pull one output once
        cuda.memcpy_dtoh_async(y_host, d_out, stream)
        stream.synchronize()

        lat_ms = np.array(lat_ms, dtype=np.float64)
        artifact["latency_ms"] = {
            "p50": float(np.percentile(lat_ms, 50)),
            "p90": float(np.percentile(lat_ms, 90)),
            "p99": float(np.percentile(lat_ms, 99)),
            "mean": float(lat_ms.mean()),
            "min": float(lat_ms.min()),
            "max": float(lat_ms.max()),
        }
        artifact["output_stats"] = {
            "min": float(np.min(y_host)),
            "mean": float(np.mean(y_host)),
            "max": float(np.max(y_host)),
            "any_nan": bool(np.isnan(y_host).any()),
        }
        artifact["status"] = "OK"

    except Exception as e:
        artifact["status"] = "FAIL"
        artifact["errors"].append(str(e))

    finally:
        try:
            ctx.pop()
        except Exception:
            pass

    # Write artifact
    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_path = f"{Path(__file__).resolve().parents[1]}/results/bench_{stamp}_{args.shape.replace('x','x')}.json"
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=to_jsonable)

    print(f"[bench_one_shape] {artifact['status']} -> {out_path}")


if __name__ == "__main__":
    main()