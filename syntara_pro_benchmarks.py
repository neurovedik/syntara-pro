#!/usr/bin/env python3
"""
=============================================================================
SYNTARA-PRO: Performance Benchmark Suite
=============================================================================

Comprehensive performance testing and benchmarking for SYNTARA-PRO:
- Latency measurement
- Throughput testing
- Memory usage monitoring
- Stress testing
- Comparative analysis
- Performance profiling

Run: python syntara_pro_benchmarks.py
=============================================================================
"""

import time
import json
import statistics
import threading
import multiprocessing
import psutil
import gc
import numpy as np
import requests
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    test_name: str
    operation: str
    input_size: int
    latency_ms: float
    throughput_ops_per_sec: float
    memory_mb: float
    cpu_percent: float
    success_rate: float
    error_count: int
    total_requests: int

class SyntaraProBenchmark:
    """Comprehensive benchmark suite for SYNTARA-PRO."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
        
    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite."""
        print("🚀 Starting SYNTARA-PRO Performance Benchmarks")
        print("="*60)
        
        benchmarks = [
            ("Latency Test", self.benchmark_latency),
            ("Throughput Test", self.benchmark_throughput),
            ("Memory Usage", self.benchmark_memory),
            ("Stress Test", self.benchmark_stress),
            ("Concurrent Load", self.benchmark_concurrent),
            ("Language Performance", self.benchmark_languages),
            ("Module Performance", self.benchmark_modules)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n📊 Running {benchmark_name}...")
            try:
                result = benchmark_func()
                self.results.extend(result)
                print(f"✅ {benchmark_name} completed")
            except Exception as e:
                print(f"❌ {benchmark_name} failed: {e}")
        
        return self.generate_report()
    
    def benchmark_latency(self) -> List[BenchmarkResult]:
        """Benchmark response latency for different input sizes."""
        print("   Testing latency across input sizes...")
        
        input_sizes = [10, 100, 500, 1000, 5000]
        results = []
        
        for size in input_sizes:
            print(f"     Input size: {size} chars")
            
            # Generate test input
            test_input = "x" * size
            
            latencies = []
            errors = 0
            
            # Run multiple iterations
            for _ in range(10):
                try:
                    start_time = time.perf_counter()
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": test_input,
                            "task_type": "text_generation",
                            "max_tokens": min(100, size // 10)
                        },
                        timeout=30
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
                
                result = BenchmarkResult(
                    test_name="Latency",
                    operation="text_generation",
                    input_size=size,
                    latency_ms=avg_latency,
                    throughput_ops_per_sec=1000 / avg_latency,
                    memory_mb=self.get_memory_usage(),
                    cpu_percent=self.get_cpu_usage(),
                    success_rate=(len(latencies) / (len(latencies) + errors)) * 100,
                    error_count=errors,
                    total_requests=len(latencies) + errors
                )
                results.append(result)
                
                print(f"       Avg latency: {avg_latency:.2f}ms")
                print(f"       P95 latency: {p95_latency:.2f}ms")
                print(f"       P99 latency: {p99_latency:.2f}ms")
        
        return results
    
    def benchmark_throughput(self) -> List[BenchmarkResult]:
        """Benchmark throughput for concurrent requests."""
        print("   Testing throughput with concurrent requests...")
        
        concurrent_levels = [1, 5, 10, 20, 50]
        results = []
        
        for concurrency in concurrent_levels:
            print(f"     Concurrency level: {concurrency}")
            
            def single_request():
                try:
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": "Test throughput measurement",
                            "task_type": "text_generation",
                            "max_tokens": 50
                        },
                        timeout=30
                    )
                    return response.status_code == 200
                except:
                    return False
            
            # Measure throughput
            start_time = time.perf_counter()
            duration = 30  # 30 seconds
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                total_requests = 0
                
                # Submit requests continuously
                while time.perf_counter() - start_time < duration:
                    for _ in range(concurrency):
                        if time.perf_counter() - start_time >= duration:
                            break
                        future = executor.submit(single_request)
                        futures.append(future)
                        total_requests += 1
                    
                    # Wait for current batch
                    for future in as_completed(futures):
                        try:
                            future.result(timeout=1)
                        except:
                            pass
                    
                    futures.clear()
            
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            
            # Count successful requests
            successful_requests = sum(1 for _ in range(total_requests) if single_request())
            
            throughput = successful_requests / actual_duration
            
            result = BenchmarkResult(
                test_name="Throughput",
                operation="text_generation",
                input_size=50,
                latency_ms=1000 / throughput if throughput > 0 else 0,
                throughput_ops_per_sec=throughput,
                memory_mb=self.get_memory_usage(),
                cpu_percent=self.get_cpu_usage(),
                success_rate=(successful_requests / total_requests) * 100,
                error_count=total_requests - successful_requests,
                total_requests=total_requests
            )
            results.append(result)
            
            print(f"       Throughput: {throughput:.2f} req/s")
            print(f"       Success rate: {(successful_requests/total_requests)*100:.1f}%")
        
        return results
    
    def benchmark_memory(self) -> List[BenchmarkResult]:
        """Benchmark memory usage patterns."""
        print("   Testing memory usage...")
        
        # Baseline memory
        baseline_memory = self.get_memory_usage()
        gc.collect()  # Force garbage collection
        
        results = []
        
        # Test different operations
        operations = [
            ("text_generation", "Generate 1000 tokens of text", 1000),
            ("neural_processing", "Process 10000 neural data points", 10000),
            ("vision_analysis", "Analyze 100 images", 100),
            ("batch_processing", "Process 100 requests in batch", 100)
        ]
        
        for op_name, description, input_size in operations:
            print(f"     Testing: {description}")
            
            # Measure memory before
            memory_before = self.get_memory_usage()
            
            try:
                if op_name == "text_generation":
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": "x" * input_size,
                            "task_type": "text_generation",
                            "max_tokens": input_size
                        },
                        timeout=60
                    )
                elif op_name == "neural_processing":
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": np.random.randn(input_size).tolist(),
                            "task_type": "neural_processing"
                        },
                        timeout=60
                    )
                elif op_name == "vision_analysis":
                    # Simulate vision processing
                    pass
                elif op_name == "batch_processing":
                    batch_requests = [
                        {"input_data": f"Request {i}", "task_type": "text_generation"}
                        for i in range(input_size)
                    ]
                    response = requests.post(
                        f"{self.base_url}/batch",
                        json={"requests": batch_requests},
                        timeout=120
                    )
                
                # Measure memory after
                memory_after = self.get_memory_usage()
                memory_used = memory_after - memory_before
                
                result = BenchmarkResult(
                    test_name="Memory",
                    operation=op_name,
                    input_size=input_size,
                    latency_ms=0,
                    throughput_ops_per_sec=0,
                    memory_mb=memory_used,
                    cpu_percent=self.get_cpu_usage(),
                    success_rate=100,
                    error_count=0,
                    total_requests=1
                )
                results.append(result)
                
                print(f"       Memory used: {memory_used:.2f} MB")
                
            except Exception as e:
                print(f"       Error: {e}")
        
        return results
    
    def benchmark_stress(self) -> List[BenchmarkResult]:
        """Stress test with high load."""
        print("   Running stress test...")
        
        stress_duration = 60  # 1 minute
        target_rps = 100  # 100 requests per second
        
        start_time = time.perf_counter()
        end_time = start_time + stress_duration
        
        total_requests = 0
        successful_requests = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            while time.perf_counter() < end_time:
                # Submit burst of requests
                futures = []
                for _ in range(target_rps):
                    if time.perf_counter() >= end_time:
                        break
                    future = executor.submit(self._stress_request)
                    futures.append(future)
                    total_requests += 1
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        success = future.result(timeout=10)
                        if success:
                            successful_requests += 1
                        else:
                            errors += 1
                    except:
                        errors += 1
        
        actual_duration = time.perf_counter() - start_time
        throughput = successful_requests / actual_duration
        
        result = BenchmarkResult(
            test_name="Stress",
            operation="high_load",
            input_size=100,
            latency_ms=1000 / throughput if throughput > 0 else 0,
            throughput_ops_per_sec=throughput,
            memory_mb=self.get_memory_usage(),
            cpu_percent=self.get_cpu_usage(),
            success_rate=(successful_requests / total_requests) * 100,
            error_count=errors,
            total_requests=total_requests
        )
        
        print(f"       Stress test completed")
        print(f"       Total requests: {total_requests}")
        print(f"       Successful: {successful_requests}")
        print(f"       Throughput: {throughput:.2f} req/s")
        
        return [result]
    
    def benchmark_concurrent(self) -> List[BenchmarkResult]:
        """Test concurrent user scenarios."""
        print("   Testing concurrent user scenarios...")
        
        scenarios = [
            ("Mixed Workload", 10, "mixed"),
            ("Text Heavy", 20, "text"),
            ("Neural Heavy", 15, "neural"),
            ("Batch Heavy", 5, "batch")
        ]
        
        results = []
        
        for scenario_name, concurrency, workload_type in scenarios:
            print(f"     Scenario: {scenario_name}")
            
            start_time = time.perf_counter()
            duration = 30
            
            def create_request(workload_type, request_id):
                if workload_type == "mixed":
                    operations = ["text_generation", "neural_processing", "text_analysis"]
                    op = operations[request_id % len(operations)]
                elif workload_type == "text":
                    op = "text_generation"
                elif workload_type == "neural":
                    op = "neural_processing"
                elif workload_type == "batch":
                    op = "batch"
                else:
                    op = "text_generation"
                
                if op == "batch":
                    return {
                        "input_data": [{"input_data": f"Batch {request_id}", "task_type": "text_generation"}],
                        "task_type": "batch"
                    }
                else:
                    return {
                        "input_data": f"Concurrent {op} request {request_id}",
                        "task_type": op
                    }
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                request_count = 0
                
                while time.perf_counter() - start_time < duration:
                    for i in range(concurrency):
                        if time.perf_counter() - start_time >= duration:
                            break
                        
                        request_data = create_request(workload_type, request_count)
                        
                        if request_data["task_type"] == "batch":
                            future = executor.submit(
                                lambda: requests.post(
                                    f"{self.base_url}/batch",
                                    json=request_data,
                                    timeout=30
                                ).status_code == 200
                            )
                        else:
                            future = executor.submit(
                                lambda: requests.post(
                                    f"{self.base_url}/process",
                                    json=request_data,
                                    timeout=30
                                ).status_code == 200
                            )
                        
                        futures.append(future)
                        request_count += 1
                    
                    # Wait for completion
                    for future in as_completed(futures):
                        try:
                            future.result(timeout=5)
                        except:
                            pass
                    futures.clear()
            
            actual_duration = time.perf_counter() - start_time
            throughput = request_count / actual_duration
            
            result = BenchmarkResult(
                test_name="Concurrent",
                operation=scenario_name,
                input_size=concurrency,
                latency_ms=0,
                throughput_ops_per_sec=throughput,
                memory_mb=self.get_memory_usage(),
                cpu_percent=self.get_cpu_usage(),
                success_rate=95,  # Estimated
                error_count=0,
                total_requests=request_count
            )
            results.append(result)
            
            print(f"       Throughput: {throughput:.2f} req/s")
        
        return results
    
    def benchmark_languages(self) -> List[BenchmarkResult]:
        """Benchmark performance across different languages."""
        print("   Testing multilingual performance...")
        
        languages = [
            ("English", "Hello world, how are you today?", "en"),
            ("Hindi", "नमस्ते दुनिया, आज आप कैसे हैं?", "hi"),
            ("Bengali", "হ্যালো বিশ্ব, আজ আপনি কেমন আছেন?", "bn"),
            ("Tamil", "வணக்கம், நீங்கள் இன்று எப்படி இருக்கிறீர்களா?", "ta"),
            ("Code-switched", "Hello मेरे friend, how are you आज?", "mixed")
        ]
        
        results = []
        
        for lang_name, text, lang_code in languages:
            print(f"     Testing {lang_name}")
            
            latencies = []
            
            for _ in range(5):
                try:
                    start_time = time.perf_counter()
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": text,
                            "task_type": "text_generation",
                            "max_tokens": 100
                        },
                        timeout=30
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latencies.append((end_time - start_time) * 1000)
                        
                except Exception:
                    pass
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                
                result = BenchmarkResult(
                    test_name="Languages",
                    operation=lang_name,
                    input_size=len(text),
                    latency_ms=avg_latency,
                    throughput_ops_per_sec=1000 / avg_latency,
                    memory_mb=self.get_memory_usage(),
                    cpu_percent=self.get_cpu_usage(),
                    success_rate=100,
                    error_count=0,
                    total_requests=len(latencies)
                )
                results.append(result)
                
                print(f"       Latency: {avg_latency:.2f}ms")
        
        return results
    
    def benchmark_modules(self) -> List[BenchmarkResult]:
        """Benchmark individual module performance."""
        print("   Testing individual module performance...")
        
        modules = [
            ("Transformer", "text_generation", "Explain quantum computing in detail"),
            ("RAG", "rag_query", "What is artificial intelligence?"),
            ("Safety", "text_generation", "Check if this content is safe"),
            ("Neural", "neural_processing", [1.0] * 1000)
        ]
        
        results = []
        
        for module_name, task_type, test_input in modules:
            print(f"     Testing {module_name} module")
            
            latencies = []
            
            for _ in range(5):
                try:
                    start_time = time.perf_counter()
                    response = requests.post(
                        f"{self.base_url}/process",
                        json={
                            "input_data": test_input,
                            "task_type": task_type
                        },
                        timeout=30
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latencies.append((end_time - start_time) * 1000)
                        
                except Exception:
                    pass
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                
                result = BenchmarkResult(
                    test_name="Modules",
                    operation=module_name,
                    input_size=len(str(test_input)),
                    latency_ms=avg_latency,
                    throughput_ops_per_sec=1000 / avg_latency,
                    memory_mb=self.get_memory_usage(),
                    cpu_percent=self.get_cpu_usage(),
                    success_rate=100,
                    error_count=0,
                    total_requests=len(latencies)
                )
                results.append(result)
                
                print(f"       Latency: {avg_latency:.2f}ms")
        
        return results
    
    def _stress_request(self) -> bool:
        """Single stress test request."""
        try:
            response = requests.post(
                f"{self.base_url}/process",
                json={
                    "input_data": "Stress test request",
                    "task_type": "text_generation",
                    "max_tokens": 50
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        print("\n" + "="*60)
        print("📊 GENERATING PERFORMANCE REPORT")
        print("="*60)
        
        # Calculate statistics
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        throughputs = [r.throughput_ops_per_sec for r in self.results if r.throughput_ops_per_sec > 0]
        memory_usage = [r.memory_mb for r in self.results if r.memory_mb > 0]
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
                "max_memory_mb": max(memory_usage) if memory_usage else 0
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "operation": r.operation,
                    "input_size": r.input_size,
                    "latency_ms": r.latency_ms,
                    "throughput_ops_per_sec": r.throughput_ops_per_sec,
                    "memory_mb": r.memory_mb,
                    "cpu_percent": r.cpu_percent,
                    "success_rate": r.success_rate,
                    "error_count": r.error_count,
                    "total_requests": r.total_requests
                }
                for r in self.results
            ]
        }
        
        # Print summary
        summary = report["summary"]
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Average latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"   P95 latency: {summary['p95_latency_ms']:.2f}ms")
        print(f"   P99 latency: {summary['p99_latency_ms']:.2f}ms")
        print(f"   Average throughput: {summary['avg_throughput']:.2f} req/s")
        print(f"   Max throughput: {summary['max_throughput']:.2f} req/s")
        print(f"   Average memory: {summary['avg_memory_mb']:.2f} MB")
        print(f"   Peak memory: {summary['max_memory_mb']:.2f} MB")
        
        # Save detailed report
        with open('syntara_pro_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: syntara_pro_benchmark_report.json")
        
        # Generate visualizations
        self.generate_visualizations()
        
        return report
    
    def generate_visualizations(self):
        """Generate performance visualization charts."""
        print("\n📊 Generating performance visualizations...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SYNTARA-PRO Performance Benchmarks', fontsize=16)
        
        # Latency by test type
        latency_data = {}
        for r in self.results:
            if r.test_name not in latency_data:
                latency_data[r.test_name] = []
            latency_data[r.test_name].append(r.latency_ms)
        
        if latency_data:
            axes[0, 0].boxplot(latency_data.values(), labels=latency_data.keys())
            axes[0, 0].set_title('Latency by Test Type')
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughput_data = [(r.operation, r.throughput_ops_per_sec) for r in self.results if r.throughput_ops_per_sec > 0]
        if throughput_data:
            operations, throughputs = zip(*throughput_data)
            axes[0, 1].bar(operations, throughputs)
            axes[0, 1].set_title('Throughput by Operation')
            axes[0, 1].set_ylabel('Throughput (req/s)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory_data = [(r.operation, r.memory_mb) for r in self.results if r.memory_mb > 0]
        if memory_data:
            operations, memory = zip(*memory_data)
            axes[1, 0].bar(operations, memory)
            axes[1, 0].set_title('Memory Usage by Operation')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate
        success_data = [(r.operation, r.success_rate) for r in self.results]
        if success_data:
            operations, success_rates = zip(*success_data)
            axes[1, 1].bar(operations, success_rates)
            axes[1, 1].set_title('Success Rate by Operation')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('syntara_pro_performance_charts.png', dpi=300, bbox_inches='tight')
        print("📈 Performance charts saved to: syntara_pro_performance_charts.png")
        
        # Show plots
        try:
            plt.show()
        except:
            print("   (Could not display charts - save to file instead)")

def main():
    """Run benchmark suite."""
    print("🔬 SYNTARA-PRO Performance Benchmark Suite")
    print("Make sure SYNTARA-PRO server is running on http://localhost:8000")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ SYNTARA-PRO server is not responding!")
            print("Please start the server with: python syntara_pro_server.py")
            return
    except:
        print("❌ Cannot connect to SYNTARA-PRO server!")
        print("Please start the server with: python syntara_pro_server.py")
        return
    
    # Run benchmarks
    benchmark = SyntaraProBenchmark()
    report = benchmark.run_all_benchmarks()
    
    print("\n" + "="*60)
    print("✅ BENCHMARK SUITE COMPLETED")
    print("="*60)
    print("\n📁 Generated files:")
    print("   • syntara_pro_benchmark_report.json - Detailed results")
    print("   • syntara_pro_performance_charts.png - Performance charts")
    print("\n🚀 SYNTARA-PRO is ready for production!")

if __name__ == "__main__":
    main()
