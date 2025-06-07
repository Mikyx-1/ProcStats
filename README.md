# 📊 ProcStats

A powerful Python package for **monitoring CPU, RAM, and GPU resources** with adaptive sampling and noise reduction. ProcStats provides robust tools to track system resource usage for any given process or function — with smart stabilization and optional NVIDIA GPU monitoring via `pynvml`.

---

## 🚀 Features

* ✅ **Tracks All Child Processes**: Monitors resource usage across parent and child processes — ideal for multi-processing applications.
* 🧠 **Smart Stabilization**: Adaptive sampling and noise filtering (moving averages + outlier rejection) yield stable, accurate metrics.
* 💻 **Cross-Platform**: Works on macOS, Windows, Linux, and Unix-like systems.
* ⏱️ **Timeout Support**: Enforce max runtime limits with graceful shutdowns.
* 🎮 **Optional NVIDIA GPU Monitoring**: Auto-detects and includes GPU stats if `pynvml` is available.
* 🧪 **Non-Intrusive Monitoring**: Uses multiprocessing to isolate the monitor from the target workload.
* 📈 **High Data Quality**: Reports max, average, 95th percentile, and a custom data quality score.
* 🧪 **Code Coverage**: >90% test coverage with full support for CPU, RAM, and GPU features (see [codecov](https://codecov.io/gh/Mikyx-1/ProcStats))

---

## 📦 Installation

Install with pip:

```bash
pip install procstats
```

To enable **GPU monitoring**, also install:

```bash
pip install pynvml
```

---

## 🧰 Usage

### ▶️ Monitor a Python Function

```python
from procstats import monitor_function_resources
from procstats.scripts.cpu_test_lib import burn_cpu_accurate

result = monitor_function_resources(
    burn_cpu_accurate,
    kwargs={"cpu_percent": 150, "duration": 5},
    base_interval=0.05,
    timeout=10.0
)
print(result)
```

### 💻 CLI Support

```python
import argparse
import os
import multiprocessing as mp
import torch

from procstats.scripts.cpu_test_lib import burn_cpu_accurate

def gpu_workload(gpu_id: int = 1):
    print(f"[Child] PID: {os.getpid()} using GPU {gpu_id}")
    try:
        device = f"cuda:{gpu_id}"
        a = torch.randn(5000, 5000, device=device)
        for _ in range(2000):
            b = torch.matmul(a, a.T)
    except RuntimeError as e:
        print(f"[Child] GPU task on cuda:{gpu_id} failed: {e}")


def heavy_gpu_task():
    print(f"[Parent] PID: {os.getpid()}")

    # Pick GPU 1 for parent, GPU 0 for child (customize as needed)
    parent_gpu = 1
    child_gpu = 1

    # Start child process
    p = mp.Process(target=gpu_workload, args=(child_gpu,))
    p.start()

    # Run the same logic in the parent process
    gpu_workload(parent_gpu)

    p.join()

    return 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPU burn workload with optional GPU usage")
    parser.add_argument("--cpu_percent", type=int, default=350, required=True, help="Total CPU percent to consume (e.g., 350)")
    parser.add_argument("--duration", type=int, default=10, required=True, help="Duration of the workload in seconds")

    args = parser.parse_args()

    burn_cpu_accurate(cpu_percent=args.cpu_percent, duration=args.duration)
```

```bash
# Syntax to monitor test_lib.py with procstats
procstats test_lib.py --cpu_percent 350 --duration 5
```

Example output:

```bash
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ procstats test_lib.py --cpu_percent 350 --duration 5
2025-06-07 22:29:18,175 - INFO - Monitoring script: test_lib.py
2025-06-07 22:29:18,175 - INFO - Script arguments: --cpu_percent 350 --duration 5
2025-06-07 22:29:18,175 - INFO - Procstats config - Interval: 0.05s, Timeout: 12.0s
2025-06-07 22:29:18,177 - INFO - Started subprocess with PID: 25511
2025-06-07 22:29:18,285 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:29:18,286 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:29:18,286 - INFO - Starting GPU monitoring for PID 25511 (include_children=True) on 2 GPU(s) with interval 0.05s and timeout 12.0s
[Monitor] Parent process 25511 terminated
2025-06-07 22:29:24,450 - INFO - No processes to monitor (original PID: 25511)
2025-06-07 22:29:24,452 - INFO - Monitoring completed. Tracked 5 PIDs: [25511, 25539, 25540, 25541, 25542]
2025-06-07 22:29:25,430 - ERROR - Failed to load function output: No module named 'test_burn_cpu'
============================================================
PROCSTATS MONITORING RESULTS
============================================================

📊 EXECUTION SUMMARY
Duration: 6.18 seconds
Timeout reached: No
Measurements taken: 35
Data quality score: 50.00

🔄 PROCESS INFORMATION
Max processes: 5
Tracked PIDs: 5

🖥️  CPU USAGE
Max CPU: 372.4%
Average CPU: 227.2%
95th percentile CPU: 372.4%
CPU cores: 12

💾 MEMORY USAGE
Max RAM: 1277.9 MB
Average RAM: 812.2 MB
95th percentile RAM: 1277.9 MB

🎮 GPU USAGE
GPU 0 - Max utilization: 0.0%
GPU 0 - Mean utilization: 0.0%
GPU 0 - Max VRAM: 0.0 MB
GPU 0 - Mean VRAM: 0.0 MB
GPU 1 - Max utilization: 0.0%
GPU 1 - Mean utilization: 0.0%
GPU 1 - Max VRAM: 0.0 MB
GPU 1 - Mean VRAM: 0.0 MB

📤 STDOUT
[Subprocess] Running with PID: 25511
pid: 25511
System: 12 CPU cores (theoretical max 1200%)
Target: 350% CPU for 5s
Strategy: 4 processes
  Process 0: 100%
  Process 1: 100%
  Process 2: 100%
  Process 3: 50%
Process 0: Starting 100% CPU burn for 5s
Process 1: Starting 100% CPU burn for 5s
Process 2: Starting 100% CPU burn for 5s
Process 3: Starting 50% CPU burn for 5s

============================================================
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ 
...
============================================================
```

---

## 📋 Requirements

* **Python**: >= 3.8
* **Required**:

  * `psutil>=5.9.0`
* **Optional**:

  * `pynvml>=11.0.0` (for GPU support)
  * `torch>=2.0.0` (for demo workloads)

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request:
👉 [github.com/Mikyx-1/ProcStats](https://github.com/Mikyx-1/ProcStats)

---

## 📜 License

Licensed under the **MIT License**. See the [LICENSE](LICENSE) file.

---

## 👤 Author

* **Name**: Le Hoang Viet
* **Email**: [lehoangviet2k@gmail.com](mailto:lehoangviet2k@gmail.com)
* **GitHub**: [Mikyx-1](https://github.com/Mikyx-1/ProcStats)

---
