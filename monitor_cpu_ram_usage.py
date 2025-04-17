import os
import psutil
import numpy as np

dummy = np.random.randn(100, 1000, 1000)

# Get the current process
proc = psutil.Process(os.getpid())

# CPU usage (as a percentage over an interval)
cpu_usage = proc.cpu_percent(interval=1.0)

# RAM usage in bytes (resident set size)
memory_usage = proc.memory_info().rss

print(f"CPU usage: {cpu_usage}%")
print(f"Memory usage: {memory_usage / (1024 ** 2):.2f} MB")
