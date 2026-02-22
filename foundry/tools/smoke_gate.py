import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "citadel" / "foundry"
PROFILES = {
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
        "onnx",
        "onnxruntime",
        "cv2",
        "PIL",
        "numpy"
    ],
}

def run_profile(profile):
    venv = ROOT / "profiles" / profile / "bin" / "python"
    if not venv.exists():
        print(f"[FAIL] Missing profile: {profile}")
        return False

    for mod in PROFILES[profile]:
        code = f"import {mod}"
        result = subprocess.run([str(venv), "-c", code])
        if result.returncode != 0:
            print(f"[FAIL] {profile} failed importing {mod}")
            return False

    print(f"[OK] {profile} imports clean")
    return True

def main():
    ok = True
    for p in PROFILES:
        if not run_profile(p):
            ok = False
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
