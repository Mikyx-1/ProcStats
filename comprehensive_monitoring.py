import psutil
import time
import multiprocessing as mp
import functools
import numpy as np


class Benchmark:
    def __init__(self, interval=0.1):
        self.interval = interval

    def _monitor_process(self, pid, stats, stop_event):
        proc = psutil.Process(pid)
        cpu_usages = []
        ram_usages = []

        proc.cpu_percent(interval=None)  # warm up

        while not stop_event.is_set():
            if proc.is_running():
                try:
                    cpu_usages.append(proc.cpu_percent(interval=None))
                    ram_usages.append(proc.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
            time.sleep(self.interval)

        stats["cpu_max"] = max(cpu_usages, default=0)
        stats["cpu_avg"] = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        stats["ram_max"] = max(ram_usages, default=0) / 1024**2  # MB
        stats["ram_avg"] = (
            sum(ram_usages) / len(ram_usages) / 1024**2 if ram_usages else 0
        )

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target_func(return_dict, *a, **kw):
                result = func(*a, **kw)
                return_dict["result"] = result

            manager = mp.Manager()
            return_dict = manager.dict()
            stats = manager.dict()
            stop_event = manager.Event()

            proc = mp.Process(
                target=target_func, args=(return_dict, *args), kwargs=kwargs
            )

            start_time = time.time()
            proc.start()

            monitor = mp.Process(
                target=self._monitor_process, args=(proc.pid, stats, stop_event)
            )
            monitor.start()

            proc.join()
            stop_event.set()
            monitor.join()
            end_time = time.time()

            duration = end_time - start_time

            print(f"[{func.__name__}] Execution time: {duration:.2f} seconds")
            print(f"[{func.__name__}] CPU max usage: {stats.get('cpu_max', 0):.2f}%")
            print(f"[{func.__name__}] CPU avg usage: {stats.get('cpu_avg', 0):.2f}%")
            print(f"[{func.__name__}] RAM max usage: {stats.get('ram_max', 0):.2f} MB")
            print(f"[{func.__name__}] RAM avg usage: {stats.get('ram_avg', 0):.2f} MB")

            return return_dict.get("result", None)

        return wrapper


@Benchmark(interval=0.1)
def my_heavy_task():
    for _ in range(5):
        dummy = np.random.randn(100, 1000, 1000)


my_heavy_task()
