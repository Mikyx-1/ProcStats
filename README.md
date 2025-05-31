# ProcStats

A Python module for monitoring CPU, RAM, and GPU resources, including child process tracking.

## Installation

```bash
pip install procstats
```

## Requirements

- Linux (x86_64)
- Python 3.6+
- NVIDIA GPU with `libnvidia-ml.so` (e.g., driver version 535) for GPU monitoring
- Install dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install nvidia-driver-535
  ```

## Usage

```python
from procstats import combined_resource_monitor

def example_function():
    # Your code here
    pass

result = combined_resource_monitor(target=example_function, timeout=10.0, interval=0.1)
print(result)
```

## Troubleshooting

If you encounter `libnvidia-ml.so` errors, ensure NVIDIA drivers are installed and update the linker cache:
```bash
sudo ldconfig
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

For detailed build instructions, see [CONTRIBUTING.md](#) (optional).