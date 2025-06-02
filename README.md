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

## Usage

### Monitor a Function
Use `monitor_function_resources` to track resource usage of a function:

```python
from procstats import monitor_function_resources
from procstats.scripts.cpu_test_lib import burn_cpu_accurate

# Monitor a CPU-intensive function
result = monitor_function_resources(burn_cpu_accurate, base_interval=0.05, timeout=10.0)
print("Resource Usage Summary:")
print(f"CPU Max: {result['cpu_max']:.2f}%")
print(f"CPU Avg: {result['cpu_avg']:.2f}%")
print(f"CPU P95: {result['cpu_p95']:.2f}%")
print(f"RAM Max: {result['ram_max']:.2f} MB")
print(f"RAM Avg: {result['ram_avg']:.2f} MB")
print(f"RAM P95: {result['ram_p95']:.2f} MB")
if result['gpu_max_util']:
    for gpu_id in result['gpu_max_util']:
        print(f"GPU {gpu_id} Max Utilisation: {result['gpu_max_util'][gpu_id]:.2f}%")
        print(f"GPU {gpu_id} VRAM Max: {result['vram_max_mb'][gpu_id]:.2f} MB")
```

### Monitor a Process
Use `ComprehensiveMonitor` to monitor an existing process by PID:

```python
from procstats.scripts.full_monitoring import ComprehensiveMonitor
import multiprocessing as mp

# Example: Monitor a process
def target_function():
    # Your function here
    pass

if __name__ == "__main__":
    process = mp.Process(target=target_function)
    process.start()
    
    monitor = ComprehensiveMonitor(process.pid, base_interval=0.05)
    result_container = mp.Manager().list()
    monitor_proc = mp.Process(
        target=monitor.monitor_resources,
        args=(result_container, 10.0)
    )
    monitor_proc.start()
    
    process.join()
    monitor_proc.join()
    
    result = result_container[0]
    print("Process Resource Usage:", result)
```

## Requirements

- **Python**: >= 3.8
- **Required Dependencies**:
  - `psutil>=5.9.0` (for CPU and RAM monitoring)
- **Optional Dependencies**:
  - `pynvml>=11.0.0` (for NVIDIA GPU monitoring; install separately if needed)
  - `torch>=2.0.0` (for GPU-related testing, e.g., `heavy_gpu_task`; install separately if needed)
- **Testing** (optional):
  - `pytest>=7.0.0` (install with `pip install pytest`)

## Project Structure

```
ProcStats/
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── setup.py
└── src/
    └── procstats/
        ├── __init__.py
        ├── scripts/
        │   ├── cpu_monitor_setup.py
        │   ├── cpu_ram_monitoring.py
        │   ├── cpu_test_lib.py
        │   ├── full_monitoring.py
        │   ├── gpu_monitor_setup.py
        │   ├── gpu_monitoring.py
        │   ├── gpu_test_lib.py
        │   └── system_info.py
        └── tests/
            ├── test_burn_cpu_ram.py
            └── test_lib.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/Mikyx-1/ProcStats).

## Licence

This project is licenced under the MIT Licence. See the [LICENCE](LICENCE) file for details.

## Contact

- **Author**: Le Hoang Viet
- **Email**: lehoangviet2k@gmail.com
- **GitHub**: [Mikyx-1/ProcStats](https://github.com/Mikyx-1/ProcStats)