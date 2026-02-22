#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def find_io(engine):
    inp = out = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inp = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out = name
    if inp is None or out is None:
        raise RuntimeError("Failed to locate input/output tensors")
    return inp, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--seed", type=int, default=72)
    args = ap.parse_args()

    d = Path(args.artifact_dir)
    plan_path = d / "engine.plan"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    rt = trt.Runtime(TRT_LOGGER)
    engine = rt.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")

    ctx = engine.create_execution_context()
    inp, out = find_io(engine)

    # Fixed contract shape for v1
    shape = (1, 8, 32)
    ctx.set_input_shape(inp, shape)
    out_shape = tuple(ctx.get_tensor_shape(out))

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(shape).astype(np.float16)
    y = np.empty(out_shape, dtype=np.float16)

    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(y.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, x, stream)
    ctx.set_tensor_address(inp, int(d_in))
    ctx.set_tensor_address(out, int(d_out))

    ok = ctx.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    if not ok:
        raise RuntimeError("execute_async_v3 returned False")

    cuda.memcpy_dtoh_async(y, d_out, stream)
    stream.synchronize()

    np.save(d / "x.npy", x)
    np.save(d / "y_trt.npy", y)

    (d / "trt_infer_manifest.json").write_text(json.dumps({
        "ts_utc": now_utc(),
        "tensorrt": trt.__version__,
        "input_name": inp,
        "output_name": out,
        "input_shape": list(shape),
        "output_shape": list(out_shape),
        "seed": args.seed,
    }, indent=2))

    print("Wrote:", d / "x.npy")
    print("Wrote:", d / "y_trt.npy")
    print("OK")

if __name__ == "__main__":
    main()
