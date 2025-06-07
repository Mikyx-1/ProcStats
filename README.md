# ProcStats

A Python package for monitoring CPU, RAM, and GPU resources with adaptive sampling and noise reduction. ProcStats provides a robust tool to track system resource usage for a given process or function, with a smart stabilisation algorithm to ensure accurate measurements. It automatically detects and includes NVIDIA GPU monitoring if `pynvml` is installed, without requiring a separate package.

## Advantages

- **Tracks All Child Processes**: Monitors resource usage across parent and child processes, ensuring comprehensive coverage of multi-process applications.
- **Smart Stabilisation Algorithm**: Uses adaptive sampling intervals and noise reduction (outlier filtering and moving average smoothing) for reliable and accurate measurements.
- **Multi-Platform Support**: Compatible with macOS, Windows, Linux, and other Unix-like systems, providing cross-platform system information and resource monitoring.
- **Timeout Support**: Allows setting a maximum monitoring duration to prevent runaway processes, with graceful termination handling.
- **Automatic GPU Detection**: Seamlessly includes NVIDIA GPU monitoring (utilisation and VRAM) when `pynvml` is available, without additional package installation.
- **Non-Intrusive Monitoring**: Leverages multiprocessing to monitor resources without interfering with the target process.
- **High Data Quality**: Provides detailed statistics (max, average, 95th percentile) and a data quality score to assess measurement reliability.

## Installation

Install the package with core dependencies (CPU and RAM monitoring):

```bash
pip install procstats
```

For NVIDIA GPU monitoring, install the optional dependency:

```bash
pip install pynvml
```

## Usage (In code & CLI Support)

### Monitor a Function
Use `monitor_function_resources` to track resource usage of a function:

```python
from procstats import monitor_function_resources
from procstats.scripts.cpu_test_lib import burn_cpu_accurate

# Monitor a CPU-intensive function
result = monitor_function_resources(burn_cpu_accurate, kwargs={"cpu_percent": 150, "duration": 5}, base_interval=0.05, timeout=10.0)
print(result)
```

```bash
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ python demo.py 
2025-06-07 22:10:34,627 - INFO - Started subprocess with PID: 22578
2025-06-07 22:10:34,740 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:10:34,745 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:10:34,745 - INFO - Starting GPU monitoring for PID 22578 (include_children=True) on 2 GPU(s) with interval 0.05s and timeout 10.0s
2025-06-07 22:10:39,821 - INFO - No processes to monitor (original PID: 22578)
2025-06-07 22:10:39,822 - INFO - Monitoring completed. Tracked 3 PIDs: [22578, 22596, 22597]
[Monitor] Parent process 22578 terminated
{'cpu_max': 153.23333333333335, 'cpu_avg': 148.66764705882355, 'cpu_p95': 153.23333333333335, 'ram_max': 70.1875, 'ram_avg': 70.1875, 'ram_p95': 70.1875, 'num_cores': 12, 'measurements_taken': 17, 'data_quality_score': 97.20216817221342, 'gpu_max_util': {0: 0, 1: 0}, 'gpu_mean_util': {0: 0, 1: 0}, 'vram_max_mb': {0: 0.0, 1: 0.0}, 'vram_mean_mb': {0: 0.0, 1: 0.0}, 'start_time': 1749309034.7407196, 'duration': 5.133028745651245, 'timeout_reached': False, 'system_info': {'cpu': {'name': '12th Gen Intel(R) Core(TM) i5-12400F', 'cores': 12, 'architecture': 'x86_64', 'frequency': 4400.0}, 'system': {'platform': 'linux', 'os_name': 'Linux', 'os_version': '6.11.0-26-generic', 'total_ram_mb': 15829.0859375, 'python_version': '3.11.11'}}, 'num_processes': 3, 'tracked_pids': [22578, 22596, 22597], 'function_result': None, 'function_error': None, 'stdout': '[Subprocess] Running with PID: 22578\npid: 22578\nSystem: 12 CPU cores (theoretical max 1200%)\nTarget: 150% CPU for 5s\nStrategy: 2 processes\n  Process 0: 100%\n  Process 1: 50%\nProcess 1: Starting 50% CPU burn for 5s\nProcess 0: Starting 100% CPU burn for 5s\n', 'stderr': ''}
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ 
```

Or you can use **CLI command**. 

For example,

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
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ procstats test_lib.py --cpu_percent 350 --duration 5
2025-06-07 22:14:51,499 - INFO - Monitoring script: test_lib.py
2025-06-07 22:14:51,499 - INFO - Script arguments: --cpu_percent 350 --duration 5
2025-06-07 22:14:51,499 - INFO - Procstats config - Interval: 0.05s, Timeout: 12.0s
2025-06-07 22:14:51,501 - INFO - Started subprocess with PID: 23507
2025-06-07 22:14:51,616 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:14:51,624 - INFO - NVIDIA ML initialized successfully
2025-06-07 22:14:51,624 - INFO - Starting GPU monitoring for PID 23507 (include_children=True) on 2 GPU(s) with interval 0.05s and timeout 12.0s
[Monitor] Parent process 23507 terminated
2025-06-07 22:14:57,751 - INFO - No processes to monitor (original PID: 23507)
2025-06-07 22:14:57,753 - INFO - Monitoring completed. Tracked 5 PIDs: [23507, 23533, 23534, 23535, 23536]
2025-06-07 22:14:58,755 - ERROR - Failed to load function output: No module named 'test_burn_cpu'
============================================================
PROCSTATS MONITORING RESULTS
============================================================

📊 EXECUTION SUMMARY
Duration: 6.15 seconds
Timeout reached: No
Measurements taken: 32
Data quality score: 95.85

🔄 PROCESS INFORMATION
Max processes: 5
Tracked PIDs: 5

🖥️  CPU USAGE
Max CPU: 385.5%
Average CPU: 353.8%
95th percentile CPU: 385.5%
CPU cores: 12

💾 MEMORY USAGE
Max RAM: 1277.9 MB
Average RAM: 903.4 MB
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
[Subprocess] Running with PID: 23507
pid: 23507
System: 12 CPU cores (theoretical max 1200%)
Target: 350% CPU for 5s
Strategy: 4 processes
  Process 0: 100%
  Process 1: 100%
  Process 2: 100%
  Process 3: 50%
Process 0: Starting 100% CPU burn for 5s
Process 2: Starting 100% CPU burn for 5s
Process 1: Starting 100% CPU burn for 5s
Process 3: Starting 50% CPU burn for 5s

============================================================
(virenv1) (base) lehoangviet@lehoangviet-MS-7D99:~/Desktop/python_projects/ProcStats-CPP$ 
```
## Requirements

- **Python**: >= 3.8
- **Required Dependencies**:
  - `psutil>=5.9.0` (for CPU and RAM monitoring)
- **Optional Dependencies**:
  - `pynvml>=11.0.0` (for NVIDIA GPU monitoring; install separately if needed)
  - `torch>=2.0.0` (for GPU-related testing, e.g., `heavy_gpu_task`; install separately if needed)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/Mikyx-1/ProcStats).

## Licence

This project is licenced under the MIT Licence. See the [LICENCE](LICENCE) file for details.

## Contact

- **Author**: Le Hoang Viet
- **Email**: lehoangviet2k@gmail.com
- **GitHub**: [Mikyx-1/ProcStats](https://github.com/Mikyx-1/ProcStats)