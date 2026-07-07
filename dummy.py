from procstats import monitor_function_resources
from procstats.tests.test_burn_cpu_ram import burn_cpu_accurate

result = monitor_function_resources(
    burn_cpu_accurate,
    kwargs={"cpu_percent": 350, "duration": 5},   # 350% -> spawns 4 processes, the worse case
    base_interval=0.05,
    timeout=10.0,
    warmup_time=1.0,   # exclude each process's first 1s from cpu_max/avg/p95 -- raise
                       # this if burn_cpu_accurate's numpy import is still leaking a spike
)

expected_samples = 5 / 0.05  # ~100
print("measurements_taken:", result["measurements_taken"], "expected ~", expected_samples)
print("cpu_max:", result["cpu_max"], "cpu_avg:", result["cpu_avg"])
print("warmup_time:", result["warmup_time"], "warmup_excluded_samples:", result["warmup_excluded_samples"])