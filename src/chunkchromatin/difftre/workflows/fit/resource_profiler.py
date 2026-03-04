"""
Resource profiling utility for tracking CPU, GPU, memory, and I/O usage
during job execution to identify bottlenecks.
"""
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, List
import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Resource profiling will be limited.")

try:
    import pynvml  # type: ignore
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class ResourceProfiler:
    """Profiles resource usage over time and identifies bottlenecks."""
    
    def __init__(self, output_path: Path, sample_interval: float = 1.0):
        """
        Initialize profiler.
        
        Args:
            output_path: Path to save profiling results JSON file
            sample_interval: Seconds between samples (default: 1.0)
        """
        self.output_path = Path(output_path)
        self.sample_interval = sample_interval
        self.samples: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
            except:
                self.gpu_available = False
    
    def _sample_resources(self) -> Dict:
        """Collect a single sample of resource usage."""
        sample = {
            "timestamp": time.time(),
            "cpu_percent": None,
            "cpu_count": None,
            "memory_mb": None,
            "memory_percent": None,
            "gpu_utilization": None,
            "gpu_memory_mb": None,
            "gpu_memory_percent": None,
            "io_read_mb": None,
            "io_write_mb": None,
        }
        
        if not PSUTIL_AVAILABLE:
            return sample
        
        try:
            # CPU
            sample["cpu_percent"] = self.process.cpu_percent(interval=None)
            sample["cpu_count"] = psutil.cpu_count()
            
            # Memory
            mem_info = self.process.memory_info()
            sample["memory_mb"] = mem_info.rss / (1024 * 1024)
            sample["memory_percent"] = self.process.memory_percent()
            
            # I/O
            io_counters = self.process.io_counters()
            sample["io_read_mb"] = io_counters.read_bytes / (1024 * 1024)
            sample["io_write_mb"] = io_counters.write_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Warning: Error sampling CPU/memory/I/O: {e}")
        
        # GPU
        if self.gpu_available:
            try:
                # Sample first GPU (can be extended to multiple)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                sample["gpu_utilization"] = util.gpu
                sample["gpu_memory_mb"] = mem_info.used / (1024 * 1024)
                sample["gpu_memory_percent"] = (mem_info.used / mem_info.total) * 100
            except Exception as e:
                print(f"Warning: Error sampling GPU: {e}")
        
        return sample
    
    def _monitor_loop(self):
        """Background thread monitoring loop."""
        while self.running:
            sample = self._sample_resources()
            self.samples.append(sample)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start profiling in background thread."""
        if self.running:
            return
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop profiling and save results."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        # Analyze and save results
        self._save_results()
    
    def _analyze_bottlenecks(self) -> Dict:
        """Analyze samples to identify bottlenecks."""
        if not self.samples:
            return {}
        
        import numpy as np
        
        # Calculate averages and peaks
        cpu_values = [s["cpu_percent"] for s in self.samples if s["cpu_percent"] is not None]
        gpu_values = [s["gpu_utilization"] for s in self.samples if s["gpu_utilization"] is not None]
        memory_values = [s["memory_mb"] for s in self.samples if s["memory_mb"] is not None]
        
        analysis = {
            "cpu": {
                "avg_percent": float(np.mean(cpu_values)) if cpu_values else None,
                "max_percent": float(np.max(cpu_values)) if cpu_values else None,
                "min_percent": float(np.min(cpu_values)) if cpu_values else None,
            },
            "gpu": {
                "avg_utilization": float(np.mean(gpu_values)) if gpu_values else None,
                "max_utilization": float(np.max(gpu_values)) if gpu_values else None,
                "min_utilization": float(np.min(gpu_values)) if gpu_values else None,
            },
            "memory": {
                "avg_mb": float(np.mean(memory_values)) if memory_values else None,
                "max_mb": float(np.max(memory_values)) if memory_values else None,
                "peak_percent": float(np.max([s["memory_percent"] for s in self.samples if s["memory_percent"] is not None])) if any(s["memory_percent"] for s in self.samples) else None,
            },
        }
        
        # Determine dominant resource
        cpu_avg = analysis["cpu"]["avg_percent"] or 0
        gpu_avg = analysis["gpu"]["avg_utilization"] or 0
        
        if gpu_avg > 20:  # GPU is being used significantly
            if gpu_avg > cpu_avg:
                analysis["dominant_resource"] = "gpu"
            else:
                analysis["dominant_resource"] = "cpu_gpu_mixed"
        else:
            analysis["dominant_resource"] = "cpu"
        
        # Identify bottlenecks
        bottlenecks = []
        if cpu_avg > 80:
            bottlenecks.append("high_cpu_usage")
        if gpu_avg > 0 and gpu_avg < 20:
            bottlenecks.append("low_gpu_utilization")
        if analysis["memory"]["peak_percent"] and analysis["memory"]["peak_percent"] > 90:
            bottlenecks.append("high_memory_usage")
        
        analysis["bottlenecks"] = bottlenecks
        
        return analysis
    
    def _save_results(self):
        """Save profiling results to JSON file."""
        if not self.samples:
            return
        
        import numpy as np
        
        analysis = self._analyze_bottlenecks()
        
        # Calculate I/O deltas (total I/O during profiling)
        io_read_total = 0
        io_write_total = 0
        if len(self.samples) > 1:
            io_read_total = self.samples[-1]["io_read_mb"] - self.samples[0]["io_read_mb"]
            io_write_total = self.samples[-1]["io_write_mb"] - self.samples[0]["io_write_mb"]
        
        results = {
            "metadata": {
                "start_time": datetime.datetime.fromtimestamp(self.samples[0]["timestamp"]).isoformat(),
                "end_time": datetime.datetime.fromtimestamp(self.samples[-1]["timestamp"]).isoformat(),
                "duration_seconds": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
                "sample_count": len(self.samples),
                "sample_interval": self.sample_interval,
            },
            "summary": {
                **analysis,
                "io": {
                    "total_read_mb": float(io_read_total),
                    "total_write_mb": float(io_write_total),
                }
            },
            "samples": self.samples,  # Full time series (can be large)
        }
        
        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Resource profiling results saved to: {self.output_path}")
        print(f"  Dominant resource: {analysis.get('dominant_resource', 'unknown')}")
        if analysis.get('bottlenecks'):
            print(f"  Bottlenecks: {', '.join(analysis['bottlenecks'])}")


@contextmanager
def profile_job(job_name: str, run_root: Path, iter_num: int, replicate: Optional[int] = None, sample_interval: float = 1.0):
    """
    Context manager for profiling a job's resource usage.
    
    Usage:
        with profile_job("reweight", run_root, iter_num):
            # ... your job code ...
    
    Args:
        job_name: Name of the job (e.g., "reweight", "observables", "simulation")
        run_root: Root directory for the run
        iter_num: Iteration number
        replicate: Optional replicate number (for simulation jobs)
        sample_interval: Seconds between samples
    
    Returns:
        Context manager that starts/stops profiling automatically
    """
    if replicate is not None:
        output_path = run_root / "logs" / f"profile_{job_name}_iter_{iter_num:03d}_rep{replicate:02d}.json"
    else:
        output_path = run_root / "logs" / f"profile_{job_name}_iter_{iter_num:03d}.json"
    profiler = ResourceProfiler(output_path, sample_interval=sample_interval)
    
    try:
        profiler.start()
        yield profiler
    finally:
        profiler.stop()
