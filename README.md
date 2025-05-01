# ProcStats

ProcStats is a Python package for monitoring CPU, RAM, and GPU usage of processes. It provides tools to track resource utilization in real-time, making it ideal for profiling heavy computational tasks.

## Installation

Install ProcStats via pip:

```bash
pip install ProcStats
```

## Usage

Monitor CPU, RAM, and GPU usage of a target function:

```python
from procstats import full_resource_monitor

def heavy_task():
    import torch
    a = torch.randn(5000, 5000, device="cuda:0")
    for _ in range(10):
        b = torch.matmul(a, a.T)

result = full_resource_monitor(heavy_task, gpu_index=None, timeout=10.0, monitor="both")
print("Monitoring Results:", result)
```

## Requirements

- Python >= 3.8
- psutil>=5.9.0
- pynvml>=11.0.0
- torch>=1.10.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.