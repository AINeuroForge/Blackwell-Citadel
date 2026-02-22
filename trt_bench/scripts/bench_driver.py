#!/usr/bin/env python3
# Multi-shape driver: spawns bench_one_shape.py as a subprocess per shape.

import argparse
import json
import subprocess
from pathlib import Path
import time


def run_one(python_bin, worker, engine, shape, profile, warmup, iters, log_level):
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = Path(worker).resolve().parents[1] / "results" / f"bench_{stamp}_{shape}.json"

    cmd = [
        python_bin, str(worker),
        "--engine", str(engine),
        "--shape", shape,
        "--profile", str(profile),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--log_level", log_level,
        "--out", str(out),
    ]
    print("\n[bench_driver] launching:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[bench_driver] worker nonzero exit:", r.returncode)
        print(r.stderr.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python", help="Python interpreter to use (isolated env)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--shapes", required=True, help="Comma list: 1024x1024,2048x2048")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--log_level", default="ERROR", choices=["ERROR", "WARNING", "INFO", "VERBOSE"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts" / "bench_one_shape.py"
    engine = Path(args.engine).resolve()

    outs = []
    for shape in [s.strip() for s in args.shapes.split(",") if s.strip()]:
        outs.append(run_one(args.python, worker, engine, shape, args.profile, args.warmup, args.iters, args.log_level))

    # Summarize
    summary = {"engine": str(engine), "runs": []}
    for p in outs:
        with open(p, "r") as f:
            summary["runs"].append(json.load(f))

    sum_path = root / "results" / f"summary_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[bench_driver] summary -> {sum_path}")


if __name__ == "__main__":
    main()