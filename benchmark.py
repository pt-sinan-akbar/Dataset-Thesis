import os
import time
import tracemalloc
import psutil
from logger import Logger
import threading

class PeakMemoryMonitor(threading.Thread):
    """Polls RSS memory usage to find the peak and calculate the average."""
    def __init__(self, pid, interval, logger: Logger):
        super().__init__(daemon=True) # Daemon thread exits when main program exits
        self.pid = pid
        self.interval = interval
        self.logger = logger

        self.peak_rss = 0
        self._rss_sum = 0
        self._sample_count = 0

        self._stopped = threading.Event()
        self.process = None

        try:
            self.process = psutil.Process(self.pid)
            # Take an initial sample if process exists
            initial_rss = self.process.memory_info().rss
            self.peak_rss = initial_rss
            self._rss_sum += initial_rss
            self._sample_count += 1
        except psutil.NoSuchProcess:
            self.logger.print(f"PeakMemoryMonitor: Process {pid} not found at init.")
        except Exception as e:
            self.logger.print(f"PeakMemoryMonitor: Error initializing for PID {pid}: {e}")
            self.process = None # Mark as unable to monitor

    def run(self):
        if not self.process:
            return

        # The initial sample was taken in __init__.
        # This loop will start after the first interval sleep.
        while not self._stopped.is_set():
            # Sleep first to represent the value over the interval,
            # then sample at the end of the interval.
            # However, for polling, it's more common to sample then sleep.
            # Let's stick to sample then sleep. The first sample is in __init__.
            # Subsequent samples are taken after each self.interval sleep.

            # Wait for the interval before taking the next sample
            # (unless it's the very first iteration after init's sample)
            # Actually, simpler: sleep at the end of the loop.
            # The first data point from init covers t0. Next data point after self.interval.

            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
                self._rss_sum += rss
                self._sample_count += 1
            except psutil.NoSuchProcess:
                break # Process ended, normal
            except Exception as e:
                self.logger.print(f"PeakMemoryMonitor: Error during polling for PID {self.pid}: {e}")
                break

            # Check stop condition more frequently if interval is long
            # but for simplicity, check once per loop after sleep
            if self._stopped.wait(self.interval): # Sleep for interval, but break if stopped
                break


    def stop(self):
        self._stopped.set()

    def get_average_rss(self) -> float:
        if self._sample_count == 0:
            return 0.0
        return self._rss_sum / self._sample_count

class PeakCpuMonitor(threading.Thread): # Unchanged from previous version
    """Polls CPU utilization to find the peak."""
    def __init__(self, pid, interval, logger: Logger):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.logger = logger
        self.peak_cpu_percent = 0.0
        self._stopped = threading.Event()
        self.process = None
        try:
            self.process = psutil.Process(self.pid)
            self.process.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            self.logger.print(f"PeakCpuMonitor: Process {pid} not found at init.")
        except Exception as e:
            self.logger.print(f"PeakCpuMonitor: Error initializing for PID {pid}: {e}")
            self.process = None

    def run(self):
        if not self.process:
            return

        while not self._stopped.is_set():
            try:
                current_cpu_percent = self.process.cpu_percent(interval=None)
                if current_cpu_percent > self.peak_cpu_percent:
                    self.peak_cpu_percent = current_cpu_percent
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                self.logger.print(f"PeakCpuMonitor: Error during polling for PID {self.pid}: {e}")
                break
            if self._stopped.wait(self.interval):
                break

    def stop(self):
        self._stopped.set()

# --- Main Benchmark Class ---

class Benchmark(object):
    def __init__(self, logger: Logger, polling_interval: float = 0.1):
        self.start_state = None
        self.logger = logger
        self.polling_interval = polling_interval

        self._peak_rss_monitor: PeakMemoryMonitor | None = None
        self._peak_cpu_monitor: PeakCpuMonitor | None = None

        self.polled_peak_process_rss = 0
        self.polled_average_process_rss = 0.0 # New attribute for average RSS
        self.polled_peak_cpu_utilization = 0.0
        self.pid = os.getpid()


    def _get_process_memory(self) -> int:
        """Gets current RSS memory usage of the process in bytes."""
        try:
            process = psutil.Process(self.pid)
            mem_info = process.memory_info()
            return mem_info.rss
        except psutil.NoSuchProcess:
            self.logger.print(f"Warning: Process {self.pid} not found during memory check.")
            return 0
        except Exception as e:
            self.logger.print(f"Warning: Could not get process memory for PID {self.pid}: {e}")
            return 0

    def _start_monitoring_threads(self):
        """Starts the polling threads for peak RSS and CPU utilization."""
        self._peak_rss_monitor = PeakMemoryMonitor(self.pid, self.polling_interval, self.logger)
        self._peak_rss_monitor.start()

        self._peak_cpu_monitor = PeakCpuMonitor(self.pid, self.polling_interval, self.logger)
        self._peak_cpu_monitor.start()

        # Give threads a moment to initialize and take their first sample/reading
        # This is especially important if the benchmarked code is very short.
        # A small fraction of the polling interval, or a fixed small amount.
        initial_settle_time = self.polling_interval / 5 if self.polling_interval > 0.05 else 0.01
        time.sleep(initial_settle_time)


    def _stop_monitoring_threads(self):
        """Stops the polling threads and collects their results."""
        join_timeout = self.polling_interval * 2 + 0.5

        if self._peak_rss_monitor:
            self._peak_rss_monitor.stop()
            self._peak_rss_monitor.join(timeout=join_timeout)
            if self._peak_rss_monitor.is_alive():
                self.logger.print("Warning: PeakMemoryMonitor thread did not terminate cleanly.")
            self.polled_peak_process_rss = self._peak_rss_monitor.peak_rss
            self.polled_average_process_rss = self._peak_rss_monitor.get_average_rss() # Get average

        if self._peak_cpu_monitor:
            self._peak_cpu_monitor.stop()
            self._peak_cpu_monitor.join(timeout=join_timeout)
            if self._peak_cpu_monitor.is_alive():
                self.logger.print("Warning: PeakCpuMonitor thread did not terminate cleanly.")
            self.polled_peak_cpu_utilization = self._peak_cpu_monitor.peak_cpu_percent


    def start_benchmark(self):
        """Starts timers, memory tracking, and monitoring threads."""
        self.logger.print("Starting benchmark...")
        self.pid = os.getpid()

        tracemalloc.start()

        mem_rss_before_bytes = self._get_process_memory()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        self.start_state = {
            "start_wall_s": start_wall,
            "start_cpu_s": start_cpu,
            "mem_rss_before_bytes": mem_rss_before_bytes
        }
        self.logger.print(f"Benchmarker initial state captured for PID {self.pid}.")

        self._start_monitoring_threads()

    def end_benchmark(self) -> dict:
        """Stops timers/memory tracking, calculates results, returns metrics dict."""
        if self.start_state is None:
            self.logger.print("Error: Benchmark was not started. Call start_benchmark() first.")
            return {}

        self._stop_monitoring_threads()

        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        mem_rss_after_bytes = self._get_process_memory()

        try:
            _, peak_py_mem_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except ValueError:
            peak_py_mem_bytes = 0
            self.logger.print("Warning: tracemalloc was not active or already stopped.")

        wall_time_taken_s = end_wall - self.start_state["start_wall_s"]
        cpu_time_taken_s = end_cpu - self.start_state["start_cpu_s"]

        avg_cpu_utilization_percent = 0.0
        if wall_time_taken_s > 1e-6:
            avg_cpu_utilization_percent = (cpu_time_taken_s / wall_time_taken_s) * 100

        mem_rss_diff_bytes = mem_rss_after_bytes - self.start_state["mem_rss_before_bytes"]
        mib_conversion_factor = 1024**2

        self.logger.print("\n--- Algorithm Benchmark Results ---")
        results = {
            "Wall Time (s)": f"{wall_time_taken_s:.4f}",
            "CPU Time (s)": f"{cpu_time_taken_s:.4f}",
            "Average CPU Utilization (%)": f"{avg_cpu_utilization_percent:.2f}",
            "Peak CPU Utilization (%, polled)": f"{self.polled_peak_cpu_utilization:.2f}",
            "Peak Python Memory (MiB - tracemalloc)": f"{peak_py_mem_bytes / mib_conversion_factor:.4f}",
            "Process Memory Start (MiB - psutil RSS)": f"{self.start_state['mem_rss_before_bytes'] / mib_conversion_factor:.4f}",
            "Process Memory End (MiB - psutil RSS)": f"{mem_rss_after_bytes / mib_conversion_factor:.4f}",
            "Process Memory Diff (MiB - psutil RSS)": f"{mem_rss_diff_bytes / mib_conversion_factor:.4f}",
            "Peak Process Memory (MiB - psutil RSS, polled)": f"{self.polled_peak_process_rss / mib_conversion_factor:.4f}",
            "Average Process Memory (MiB - psutil RSS, polled)": f"{self.polled_average_process_rss / mib_conversion_factor:.4f}" # New metric
        }

        for key, value in results.items():
            self.logger.print(f"  {key}: {value}")
        self.logger.print("--- Benchmark Ended ---")

        self.start_state = None
        return results