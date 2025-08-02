# turboinfra-agent :: single-file minimal demo
"""
A **toy end-to-end prototype** that follows the TurboInfra Agent flow in **one Python
file**.  Each stage prints what it would normally do; heavy-weight work (real FFTs,
Nvprof, GPT calls) is stubbed with placeholders so the script can run anywhere
that has Python-3 and an NVIDIA toolchain.

Pipeline:
1. parser      - ingest a tiny JSON workload description
2. profiler    - (placeholder) create a fake trace / hot-op list
3. planner     - (placeholder) choose an optimisation plan
4. kernelgen   - write a dummy CUDA kernel file
5. runner      - compile & run the kernel (or mock if nvcc unavailable)
6. scorer      - print a made-up efficiency score

Run with:
    python demo.py examples/turbo_fno_minimal.json
"""

import json
import subprocess
import os
import sys
from pathlib import Path

# -------------------------------------------------------------
# 1) Parser - read NL / code description → JSON workflow IR
# -------------------------------------------------------------

def parse_workload(path: str):
    print("[parser] reading workload description …")
    with open(path, "r") as f:
        data = json.load(f)
    ops       = data.get("model", {}).get("ops", [])
    hardware  = data.get("hardware", "unknown")
    print(f"[parser] ops: {len(ops)} | target hw: {hardware}\n")
    return ops, hardware

# -------------------------------------------------------------
# 2) Profiler - pretend to profile and output hot-op list
# -------------------------------------------------------------

def profile_workload(ops):
    print("[profiler] generating fake trace …")
    trace = [{"op": op, "latency_ms": 1.0 + i * 0.1} for i, op in enumerate(ops)]
    hot   = sorted(trace, key=lambda x: -x["latency_ms"])[:3]
    for h in hot:
        print(f"  hot-op: {h['op']}  {h['latency_ms']:.2f}-ms")
    print()
    return trace, hot

# -------------------------------------------------------------
# 3) Planner - naive: pick first hot-op and say "fuse" it
# -------------------------------------------------------------

def plan_optimisations(hot_ops):
    target = hot_ops[0]["op"] if hot_ops else "unknown_op"
    plan = {"target": target, "strategy": "fuse_with_next"}
    print(f"[planner] plan → optimise '{target}' via {plan['strategy']}\n")
    return plan

# -------------------------------------------------------------
# 4) KernelGen - write a dummy CUDA kernel implementing Op*Op
# -------------------------------------------------------------

def generate_kernel(plan):
    kernel_path = Path("generated_kernel.cu")
    print(f"[kernelgen] writing placeholder kernel → {kernel_path}")
    kernel_src = f"""
#include <cuda_runtime.h>
__global__ void fused_kernel(float* a, float* b, float* out) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = a[idx] * b[idx];  // pretend fused '{plan['target']}'
}}
"""
    kernel_path.write_text(kernel_src)
    return kernel_path

# -------------------------------------------------------------
# 5) Runner - compile + run dummy kernel (or mock if nvcc absent)
# -------------------------------------------------------------

def compile_and_run(kernel_path: Path):
    bin_path = kernel_path.with_suffix(".out")
    nvcc = "nvcc"
    compile_cmd = f"{nvcc} {kernel_path} -o {bin_path}"
    print(f"[runner] compiling with: {compile_cmd}")
    try:
        subprocess.run(compile_cmd.split(), check=True)
        print("[runner] running compiled kernel … (fake)\n")
        # Real run would launch kernel via cuda runtime; we mock here.
    except FileNotFoundError:
        print("[runner] nvcc not found - skipping compile (mock run)\n")
    return {"latency_ms": 0.42, "gflops": 123.4}

# -------------------------------------------------------------
# 6) Scorer - compute toy efficiency score
# -------------------------------------------------------------

def score(metrics):
    util = metrics["gflops"] / 312.0  # pretend peak = 312-GFLOPS
    print(f"[scorer] achieved {metrics['gflops']}-GFLOPS → util {util*100:.1f}%\n")
    return util

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(cfg_path: str):
    ops, hw          = parse_workload(cfg_path)
    trace, hot_ops   = profile_workload(ops)
    plan             = plan_optimisations(hot_ops)
    kernel_path      = generate_kernel(plan)
    metrics          = compile_and_run(kernel_path)
    final_score      = score(metrics)
    print(f"[agent] DONE - demo finished with score: {final_score:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo.py <workload_json>")
        sys.exit(1)
    main(sys.argv[1])
