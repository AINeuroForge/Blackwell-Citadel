import os
import argparse
import numpy as np
import json
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.20)
    args = ap.parse_args()

    art = args.artifact_dir

    y_trt = np.load(os.path.join(art,"y_trt.npy"))
    y_onnx = np.load(os.path.join(art,"y_onnx.npy"))

    diff = np.abs(y_trt.astype(np.float32) - y_onnx.astype(np.float32))

    result = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape": list(y_trt.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p99_abs_diff": float(np.percentile(diff,99)),
        "tolerance": args.tol,
        "pass": bool(diff.max() <= args.tol)
    }

    print(json.dumps(result, indent=2))

    if not result["pass"]:
        exit(1)

if __name__ == "__main__":
    main()
