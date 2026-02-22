import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    rt = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError("Failed to deserialize engine")
    return eng

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args()

    art = args.artifact_dir
    plan_path = os.path.join(art, "transformer_layer_v1.plan")
    x_path = os.path.join(art, "x.npy")

    engine = load_engine(plan_path)
    ctx = engine.create_execution_context()

    inp = engine.get_tensor_name(0)
    out = engine.get_tensor_name(1)

    x = np.load(x_path)
    ctx.set_input_shape(inp, x.shape)

    out_shape = tuple(ctx.get_tensor_shape(out))

    inp_dtype = trt.nptype(engine.get_tensor_dtype(inp))
    out_dtype = trt.nptype(engine.get_tensor_dtype(out))

    x = x.astype(inp_dtype)
    y = np.empty(out_shape, dtype=out_dtype)

    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(y.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, x, stream)
    ctx.set_tensor_address(inp, int(d_in))
    ctx.set_tensor_address(out, int(d_out))

    ok = ctx.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TRT execution failed")

    cuda.memcpy_dtoh_async(y, d_out, stream)
    stream.synchronize()

    out_path = os.path.join(art, "y_trt.npy")
    np.save(out_path, y)
    print("Wrote:", out_path)
    print("TRT OK")

if __name__ == "__main__":
    main()
