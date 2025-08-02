# 🚀 TurboInfra Agent

**LLM-powered end-to-end optimizer for AI core computations** — from kernel rewriting to operator fusion — pushing your code toward hardware peak performance.

---

## 🧠 What is TurboInfra Agent?

`turboinfra-agent` is an autonomous system that understands your AI workloads and rewrites them for speed. It leverages large language models (LLMs) to:

- Parse user-defined compute pipelines or model architectures
- Analyze bottlenecks using real profiling data
- Plan optimizations across kernels and operator boundaries
- Generate and fuse custom CUDA kernels
- Score and refine implementations toward optimal performance

---

## 📦 Key Features

- ✨ **LLM-Powered Optimization Loop**: Parse → Plan → Generate → Run → Score → Refine
- 🧩 **Cross-layer optimization**: from intra-kernel tricks to inter-operator fusion
- 🔍 **Hardware-aware scoring**: GFLOPS, memory BW, utilization
- 📐 **Supports structured or natural language inputs**: e.g., YAML / Python / prompts
- 🛠️ **Easily extendable**: Modular system by design

---

## 🧱 Project Structure

```

turboinfra-agent/
│
├── modules/
│   ├── parser.py        # Extracts semantic + structural description from code/prompt
│   ├── profiler.py      # Calls Nsight or similar to extract perf profile
│   ├── planner.py       # Plans optimization pass: fusion, reordering, etc.
│   ├── kernelgen.py     # Generates customized CUDA kernels
│   ├── runner.py        # Executes and tests generated kernels
│   └── scorer.py        # Evaluates latency, efficiency, FLOPS
│
├── examples/
│   └── turbo\_fno.yaml   # Example workload (FFT -> GEMM -> iFFT)
│
├── scripts/
│   └── run\_agent.py     # End-to-end entry script
│
├── assets/              # Architecture diagrams, plots
├── README.md
├── LICENSE              # Apache 2.0
└── requirements.txt     # Python dependencies (e.g., openai, torch, etc.)

````

---

## 🔧 Quick Start

```bash
# Clone the repository
git clone https://github.com/TurboInfra/turboinfra-agent.git
cd turboinfra-agent

# Install dependencies
pip install -r requirements.txt

# Run on an example workload
python scripts/run_agent.py --input examples/turbo_fno.yaml
````

---

## 🧪 Example Use Case: TurboFNO

From this simple prompt:

```yaml
model:
  name: TurboFNO
  ops:
    - FFT2D(input_shape=[bs, dx, dy, k])
    - Truncation(shape=[64, 64])
    - ComplexGEMM(in_shape=[bs*64*64, k], weight_shape=[k, n])
    - iFFT2D(output_shape=[bs, dx, dy, n])
hardware: A100
```

TurboInfra Agent automatically:

* Detects opportunities for built-in truncation + zero-padding
* Generates fused CUDA kernel with memory swizzling
* Scores performance using real profiling
* Iteratively rewrites until achieving \~80% of theoretical FP32 peak

---

## 📃 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## ✨ Vision

TurboInfra aims to **automate the co-design loop between algorithms and hardware**, using GenAI agents to replace tedious, manual performance engineering.
No more hand-tuning. Just describe what you want — and get optimized code back.

---

## 🤝 Contributing

We welcome contributions! Please file issues or open PRs for feature ideas, optimizations, or workloads you'd like to see integrated.

---

## 📫 Contact

You can reach the TurboInfra team at: `turboinfra404@gmail.com`
