import os
import time
import tracemalloc
import psutil
from logger import Logger

class Benchmark(object):
    def __init__(self, logger: Logger):
        self.start_state = None
        self.logger = logger

    def _get_process_memory(self):
        """Gets current RSS memory usage of the process."""
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss # rss: Resident Set Size
        except psutil.NoSuchProcess:
            return 0 # Process may have ended

    def start_benchmark(self):
        """Starts timers and memory tracking, returns start state."""
        self.logger.print("Starting benchmark...")
        tracemalloc.start()
        mem_rss_before = self._get_process_memory()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.start_state = {
            "start_wall": start_wall,
            "start_cpu": start_cpu,
            "mem_rss_before": mem_rss_before
        }
        self.logger.print(f"Benchmarker start state: {self.start_state}")

    def end_benchmark(self):
        """Stops timers/memory tracking, calculates results, returns metrics dict."""
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        mem_rss_after = self._get_process_memory()
        try:
            current_py_mem, peak_py_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except ValueError: # tracemalloc might have been stopped or not started
            current_py_mem, peak_py_mem = 0, 0
            self.logger.print("Warning: tracemalloc was not active or already stopped.")


        wall_time_taken = end_wall - self.start_state["start_wall"]
        cpu_time_taken = end_cpu - self.start_state["start_cpu"]
        self.logger.print("Benchmark result: ")
        results = {
            "Wall Time (s)": wall_time_taken,
            "CPU Time (s)": cpu_time_taken,
            "Peak Python Memory (MiB)": peak_py_mem / 1024**2,
            #"Current Python Memory (MiB)": current_py_mem / 1024**2, # Less relevant than peak
            "Process Memory Start (MiB - psutil)": self.start_state["mem_rss_before"] / 1024**2,
            "Process Memory End (MiB - psutil)": mem_rss_after / 1024**2,
            # Peak RSS requires polling or external tools like /usr/bin/time -v
            "Process Memory Diff (MiB - psutil)": (mem_rss_after - self.start_state["mem_rss_before"]) / 1024**2
        }
        self.logger.print(results)
        self.logger.print("Benchmark ended.")