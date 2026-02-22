#!/usr/bin/env python3
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

FOUNDRY_ROOT = Path.home() / "citadel" / "foundry"
PROFILES = FOUNDRY_ROOT / "profiles"
RESULTS = FOUNDRY_ROOT / "manifests"

RESULTS.mkdir(parents=True, exist_ok=True)

# Define required modules per profile
PROFILE_MODULES = {
    "trtllm": [
        "tensorrt",
        "tensorrt_llm",
        "torch",
        "numpy",
        "pycuda"
    ],
    "vision": [
        "torch",
        "torchvision",
        "numpy",
        "onnx",
        "onnxruntime",
        "cv2",
        "PIL"
    ],
    "modeling": [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "plotly",
        "polars",
        "pyarrow",
        "dask",
        "optuna",
        "mlflow",
        "ray"
    ]
}

def run_import_test(python_path, modules):
    results = {}
    for mod in modules:
        try:
            cmd = [
                str(python_path),
                "-c",
                f"import {mod}; print({mod}.__version__ if hasattr({mod}, '__version__') else 'OK')"
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            results[mod] = out.strip()
        except subprocess.CalledProcessError as e:
            results[mod] = f"FAIL: {e.output.strip()}"
    return results

def main():
    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "profiles": {}
    }

    for profile_name, modules in PROFILE_MODULES.items():
        profile_path = PROFILES / profile_name
        python_path = profile_path / "bin" / "python"

        if not python_path.exists():
            report["profiles"][profile_name] = {
                "status": "MISSING_PROFILE",
                "details": None
            }
            continue

        print(f"\n[CHECKING] {profile_name}")
        results = run_import_test(python_path, modules)

        failures = [k for k, v in results.items() if v.startswith("FAIL")]
        status = "OK" if not failures else "FAILED_IMPORTS"

        report["profiles"][profile_name] = {
            "status": status,
            "details": results
        }

        for mod, result in results.items():
            print(f"  {mod}: {result}")

    out_file = RESULTS / f"foundry_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report written to: {out_file}")

    # Fail hard if any profile failed
    any_failed = any(
        p["status"] != "OK" for p in report["profiles"].values()
    )

    if any_failed:
        print("\n[FOUNDATION WARNING] One or more profiles failed validation.")
        sys.exit(1)

    print("\n[FOUNDATION STABLE] All profiles validated successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
