#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--opset", type=int, default=19)
    args = ap.parse_args()

    d = Path(args.artifact_dir)
    wpath = d / "weights.npz"
    xpath = d / "x.npy"
    if not wpath.exists():
        raise FileNotFoundError(wpath)
    if not xpath.exists():
        raise FileNotFoundError(xpath)

    W = np.load(wpath)
    Wq, Wk, Wv, Wo = W["Wq"], W["Wk"], W["Wv"], W["Wo"]

    x = np.load(xpath).astype(np.float16)  # (1,8,32)

    B, S, H = 1, 8, 32
    n_heads = 4
    head_dim = 8

    def const(name, arr: np.ndarray):
        return numpy_helper.from_array(arr, name=name)

    in_x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [B, S, H])
    out_y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [B, S, H])

    inits = [
        const("Wq", Wq.astype(np.float16)),
        const("Wk", Wk.astype(np.float16)),
        const("Wv", Wv.astype(np.float16)),
        const("Wo", Wo.astype(np.float16)),
        const("shape_BS_H", np.array([B*S, H], dtype=np.int64)),
        const("shape_B_S_H", np.array([B, S, H], dtype=np.int64)),
        const("shape_B_S_nh_hd", np.array([B, S, n_heads, head_dim], dtype=np.int64)),
        const("shape_BS_H_out", np.array([B*S, H], dtype=np.int64)),
        const("scale", np.array([1.0/np.sqrt(head_dim)], dtype=np.float16).reshape(1,1,1,1)),
    ]

    nodes = []

    # flatten x: [1,8,32] -> [8,32]
    nodes.append(helper.make_node("Reshape", ["x", "shape_BS_H"], ["x2d"]))

    # QKV 2D: [8,32] @ [32,32] -> [8,32]
    nodes.append(helper.make_node("MatMul", ["x2d", "Wq"], ["q2d"]))
    nodes.append(helper.make_node("MatMul", ["x2d", "Wk"], ["k2d"]))
    nodes.append(helper.make_node("MatMul", ["x2d", "Wv"], ["v2d"]))

    # back to 3D: [1,8,32]
    nodes.append(helper.make_node("Reshape", ["q2d", "shape_B_S_H"], ["q"]))
    nodes.append(helper.make_node("Reshape", ["k2d", "shape_B_S_H"], ["k"]))
    nodes.append(helper.make_node("Reshape", ["v2d", "shape_B_S_H"], ["v"]))

    # split heads: [1,8,32] -> [1,8,4,8] -> transpose -> [1,4,8,8]
    nodes.append(helper.make_node("Reshape", ["q", "shape_B_S_nh_hd"], ["q_rs"]))
    nodes.append(helper.make_node("Reshape", ["k", "shape_B_S_nh_hd"], ["k_rs"]))
    nodes.append(helper.make_node("Reshape", ["v", "shape_B_S_nh_hd"], ["v_rs"]))

    nodes.append(helper.make_node("Transpose", ["q_rs"], ["qh"], perm=[0,2,1,3]))
    nodes.append(helper.make_node("Transpose", ["k_rs"], ["kh"], perm=[0,2,1,3]))
    nodes.append(helper.make_node("Transpose", ["v_rs"], ["vh"], perm=[0,2,1,3]))

    # scores = qh @ kh^T over last 2 dims: khT perm [0,1,3,2]
    nodes.append(helper.make_node("Transpose", ["kh"], ["khT"], perm=[0,1,3,2]))
    nodes.append(helper.make_node("MatMul", ["qh", "khT"], ["scores"]))

    # scaled scores
    nodes.append(helper.make_node("Mul", ["scores", "scale"], ["scaled"]))

    # softmax over last dim
    nodes.append(helper.make_node("Softmax", ["scaled"], ["p"], axis=3))

    # context = p @ vh -> [1,4,8,8]
    nodes.append(helper.make_node("MatMul", ["p", "vh"], ["ctx"]))

    # merge: transpose [0,2,1,3] -> [1,8,4,8]
    nodes.append(helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0,2,1,3]))

    # reshape -> [8,32]
    nodes.append(helper.make_node("Reshape", ["ctx_t", "shape_BS_H_out"], ["attn2d"]))

    # output proj -> [8,32]
    nodes.append(helper.make_node("MatMul", ["attn2d", "Wo"], ["proj2d"]))

    # back to [1,8,32]
    nodes.append(helper.make_node("Reshape", ["proj2d", "shape_B_S_H"], ["proj"]))

    # residual add: y = x + proj
    nodes.append(helper.make_node("Add", ["x", "proj"], ["y"]))

    graph = helper.make_graph(
        nodes=nodes,
        name="micro_transformer_v1",
        inputs=[in_x],
        outputs=[out_y],
        initializer=inits
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", args.opset)],
        producer_name="CitadelFoundry"
    )
    onnx.checker.check_model(model)

    onnx_path = d / "model.onnx"
    onnx.save(model, onnx_path)

    # ONNXRuntime inference
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    y = sess.run(["y"], {"x": x})[0]

    np.save(d / "y_onnx.npy", y)

    (d / "vision_onnx_manifest.json").write_text(json.dumps({
        "ts_utc": now_utc(),
        "opset": args.opset,
        "providers": sess.get_providers(),
        "onnx_path": str(onnx_path),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "onnxruntime": ort.__version__,
    }, indent=2))

    print("Wrote:", onnx_path)
    print("Wrote:", d / "y_onnx.npy")
    print("Providers:", sess.get_providers())
    print("OK")

if __name__ == "__main__":
    main()
