#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.10)
    args = ap.parse_args()

    d = Path(args.artifact_dir)
    y_trt = np.load(d / "y_trt.npy").astype(np.float32)
    y_onnx = np.load(d / "y_onnx.npy").astype(np.float32)

    if y_trt.shape != y_onnx.shape:
        raise RuntimeError(f"Shape mismatch: trt={y_trt.shape} onnx={y_onnx.shape}")

    diff = np.abs(y_trt - y_onnx)
    report = {
        "ts_utc": now_utc(),
        "shape": list(y_trt.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p99_abs_diff": float(np.quantile(diff, 0.99)),
        "tolerance": args.tol,
        "pass": bool(diff.max() <= args.tol),
    }

    (d / "compare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
