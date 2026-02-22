#!/usr/bin/env bash
set -euo pipefail

echo "=== GATE 0: GPU identity ==="
nvidia-smi --query-gpu=name,pci.bus_id,driver_version --format=csv
ls -l /dev/dxg

echo
echo "=== GATE 1: Toolkit identity ==="
command -v nvcc
nvcc --version
ls -l /usr/local | grep -E 'cuda'

echo
echo "=== GATE 2: Runtime/Driver integer check ==="
cat > /tmp/citadel_versions.cu <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int drv=0, rt=0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);
  printf("cudaDriverGetVersion=%d cudaRuntimeGetVersion=%d\n", drv, rt);
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  printf("device=%s cc=%d.%d\n", p.name, p.major, p.minor);
  return 0;
}
CU

nvcc -O2 /tmp/citadel_versions.cu -o /tmp/citadel_versions
/tmp/citadel_versions

echo
echo "=== GATE 3: PTX JIT forced execution ==="
export CUDA_FORCE_PTX_JIT=1
/tmp/citadel_versions
unset CUDA_FORCE_PTX_JIT

echo
echo "PASS: All gates succeeded."
