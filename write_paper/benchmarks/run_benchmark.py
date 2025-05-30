#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import asyncio
import logging
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import shutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main import generate_paper
from src.state import ResearchState

# Configure logging
logger = logging.getLogger("benchmark")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)

class BenchmarkResult:
    """Class to store benchmark run results"""
    def __init__(self, topic: str, model: str, config: Dict[str, Any]):
        self.topic = topic
        self.model = model
        self.config = config
        self.start_time = time.time()
        self.end_time = None
        self.success = False
        self.error = None
        self.output_path = None
        self.metrics = {}

    def complete(self, success: bool, output_path: Optional[Path] = None, error: Optional[str] = None):
        """Mark the benchmark as complete"""
        self.end_time = time.time()
        self.success = success
        self.output_path = output_path
        self.error = error

    def duration(self) -> float:
        """Get the duration of the benchmark run in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary"""
        return {
            "topic": self.topic,
            "model": self.model,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "success": self.success,
            "error": self.error,
            "output_path": str(self.output_path) if self.output_path else None,
            "metrics": self.metrics
        }


class Benchmark:
    """Main benchmark runner class"""
    def __init__(self, config_path: str, output_dir: str = "benchmark_results"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.config = self._load_config()
        self.results = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir()

    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            raise

    async def run_single_benchmark(self, topic: str, model: str, params: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark with the given parameters"""
        logger.info(f"Running benchmark - Topic: {topic}, Model: {model}")

        # Create result object
        result = BenchmarkResult(topic, model, params)

        try:
            # Prepare output directory
            topic_dir_name = topic.replace(' ', '_').replace('(', '').replace(')', '')
            output_dir = self.run_dir / f"{model.replace(':', '_')}_{topic_dir_name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run the paper generation
            await generate_paper(
                topic=topic,
                output_dir=str(output_dir),
                model=model,
                reference_num=params.get("reference_num", 100),
                num_sections=params.get("num_sections", 8),
                papers_per_section=params.get("papers_per_section", 5)
            )

            # Calculate metrics based on the output
            # Example: count tokens, measure completion time, etc.
            metrics = self._calculate_metrics(output_dir, topic)
            result.metrics = metrics

            # Mark as successful
            result.complete(True, output_dir)
            logger.info(f"Benchmark completed successfully - Topic: {topic}, Model: {model}")

        except Exception as e:
            logger.error(f"Benchmark failed - Topic: {topic}, Model: {model}, Error: {str(e)}")
            result.complete(False, error=str(e))

        return result

    def _calculate_metrics(self, output_dir: Path, topic: str) -> Dict[str, Any]:
        """Calculate metrics from benchmark output"""
        metrics = {}

        # Try to get metrics from the state file
        try:
            result_file = Path("result") / f"{topic}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result_data = json.load(f)

                # Calculate number of papers found
                papers_count = len(result_data.get("state", {}).get("related_papers", {}))
                metrics["papers_count"] = papers_count

                # Calculate number of sections
                sections = result_data.get("state", {}).get("generated_sections", {})
                metrics["sections_count"] = len(sections)

                # Calculate total token count (rough estimate)
                total_text = ""
                for section_text in sections.values():
                    total_text += section_text

                # Very rough token count (words / 0.75)
                word_count = len(total_text.split())
                metrics["word_count"] = word_count
                metrics["token_count_estimate"] = int(word_count / 0.75)
        except Exception as e:
            logger.warning(f"Failed to calculate metrics: {str(e)}")

        return metrics

    async def run_benchmark_set(self, name: str) -> List[BenchmarkResult]:
        """Run a benchmark set defined in the config"""
        logger.info(f"Starting benchmark set: {name}")

        if name not in self.config.get("benchmarks", {}):
            raise ValueError(f"Benchmark set '{name}' not found in config")

        # Get benchmark configuration
        benchmark_config = self.config["benchmarks"][name]

        # Get default parameters
        default_params = self.config.get("default", {})

        # Merge with benchmark-specific parameters
        params = {**default_params}
        for key, value in benchmark_config.items():
            if key not in ["models", "topics", "description"]:
                params[key] = value

        # Get models and topics
        models = benchmark_config.get("models", [])
        topics = benchmark_config.get("topics", [])

        # Expand model and topic names if they're specified as strings
        models = self._expand_names(models, self.config.get("models", []), "name")
        topics = self._expand_names(topics, self.config.get("topics", []), "name")

        # Run benchmarks
        tasks = []
        for model in models:
            for topic in topics:
                tasks.append(self.run_single_benchmark(topic, model, params))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        filtered_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Benchmark task failed: {str(result)}")
            else:
                filtered_results.append(result)
                self.results.append(result)

        return filtered_results

    def _expand_names(self, items: List[str], full_list: List[Dict[str, Any]], name_key: str) -> List[str]:
        """Expand item names from the config"""
        if not items:
            return [item.get(name_key) for item in full_list]

        # If items are already fully specified, return as is
        if isinstance(items[0], str):
            return items

        # Otherwise, extract names
        return [item.get(name_key) for item in items]

    async def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmark sets defined in the config"""
        for name in self.config.get("benchmarks", {}).keys():
            await self.run_benchmark_set(name)
        return self.results

    def generate_report(self):
        """Generate a report of benchmark results"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return

        # Create report directory
        report_dir = self.run_dir / "report"
        report_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(report_dir / "results.json", "w") as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2)

        # Generate summary report
        summary = {
            "timestamp": self.timestamp,
            "total_benchmarks": len(self.results),
            "successful_benchmarks": sum(1 for r in self.results if r.success),
            "failed_benchmarks": sum(1 for r in self.results if not r.success),
            "total_duration": sum(r.duration() for r in self.results),
            "results_by_model": {},
            "results_by_topic": {}
        }

        # Group by model
        for result in self.results:
            if result.model not in summary["results_by_model"]:
                summary["results_by_model"][result.model] = {
                    "count": 0,
                    "successful": 0,
                    "avg_duration": 0,
                    "topics": []
                }

            model_summary = summary["results_by_model"][result.model]
            model_summary["count"] += 1
            if result.success:
                model_summary["successful"] += 1
                model_summary["avg_duration"] = ((model_summary["avg_duration"] * (model_summary["successful"] - 1))
                                               + result.duration()) / model_summary["successful"]
            model_summary["topics"].append(result.topic)

        # Group by topic
        for result in self.results:
            if result.topic not in summary["results_by_topic"]:
                summary["results_by_topic"][result.topic] = {
                    "count": 0,
                    "successful": 0,
                    "avg_duration": 0,
                    "models": []
                }

            topic_summary = summary["results_by_topic"][result.topic]
            topic_summary["count"] += 1
            if result.success:
                topic_summary["successful"] += 1
                topic_summary["avg_duration"] = ((topic_summary["avg_duration"] * (topic_summary["successful"] - 1))
                                               + result.duration()) / topic_summary["successful"]
            topic_summary["models"].append(result.model)

        # Save summary
        with open(report_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate HTML report
        self._generate_html_report(report_dir, summary)

        logger.info(f"Report generated at {report_dir}")
        return report_dir

    def _generate_html_report(self, report_dir: Path, summary: Dict[str, Any]):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .card {{ background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px; padding: 15px; border-radius: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Run timestamp: {self.timestamp}</p>
                <p>Total benchmarks: {summary["total_benchmarks"]}</p>
                <p>Successful: <span class="success">{summary["successful_benchmarks"]}</span></p>
                <p>Failed: <span class="failure">{summary["failed_benchmarks"]}</span></p>
                <p>Total duration: {summary["total_duration"]:.2f}s ({(summary["total_duration"]/60):.2f} minutes)</p>
            </div>

            <h2>Results by Model</h2>
        """

        # Add model results
        for model, model_data in summary["results_by_model"].items():
            success_rate = (model_data["successful"] / model_data["count"]) * 100 if model_data["count"] > 0 else 0
            html_content += f"""
            <div class="card">
                <h3>{model}</h3>
                <p>Success rate: {success_rate:.1f}% ({model_data["successful"]}/{model_data["count"]})</p>
                <p>Average duration: {model_data["avg_duration"]:.2f}s ({(model_data["avg_duration"]/60):.2f} minutes)</p>
                <details>
                    <summary>Topics tested</summary>
                    <ul>
            """

            for topic in model_data["topics"]:
                html_content += f"<li>{topic}</li>\n"

            html_content += """
                    </ul>
                </details>
            </div>
            """

        html_content += "<h2>Results by Topic</h2>"

        # Add topic results
        for topic, topic_data in summary["results_by_topic"].items():
            success_rate = (topic_data["successful"] / topic_data["count"]) * 100 if topic_data["count"] > 0 else 0
            html_content += f"""
            <div class="card">
                <h3>{topic}</h3>
                <p>Success rate: {success_rate:.1f}% ({topic_data["successful"]}/{topic_data["count"]})</p>
                <p>Average duration: {topic_data["avg_duration"]:.2f}s ({(topic_data["avg_duration"]/60):.2f} minutes)</p>
                <details>
                    <summary>Models tested</summary>
                    <ul>
            """

            for model in topic_data["models"]:
                html_content += f"<li>{model}</li>\n"

            html_content += """
                    </ul>
                </details>
            </div>
            """

        # Add detailed results table
        html_content += """
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Topic</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Metrics</th>
                </tr>
        """

        for result in self.results:
            status = '<span class="success">Success</span>' if result.success else f'<span class="failure">Failure: {result.error}</span>'
            metrics_str = ", ".join([f"{k}: {v}" for k, v in result.metrics.items()]) if result.metrics else "N/A"

            html_content += f"""
                <tr>
                    <td>{result.model}</td>
                    <td>{result.topic}</td>
                    <td>{status}</td>
                    <td>{result.duration():.2f}s</td>
                    <td>{metrics_str}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(report_dir / "report.html", "w") as f:
            f.write(html_content)


async def main():
    """Main entry point for the benchmark runner"""
    parser = argparse.ArgumentParser(description="Run benchmarks for the research paper generation system")
    parser.add_argument("--config", default="benchmarks/configs/default.yaml", help="Path to the benchmark config file")
    parser.add_argument("--output", default="benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--run", default=None, help="Specific benchmark set to run (omit to run all)")
    parser.add_argument("--topic", default=None, help="Specific topic to run (overrides config)")
    parser.add_argument("--model", default=None, help="Specific model to run (overrides config)")
    parser.add_argument("--reference-num", type=int, default=None, help="Number of references (overrides config)")
    parser.add_argument("--num-sections", type=int, default=None, help="Number of sections (overrides config)")
    parser.add_argument("--papers-per-section", type=int, default=None, help="Papers per section (overrides config)")

    args = parser.parse_args()

    # Create benchmark runner
    benchmark = Benchmark(args.config, args.output)

    # Apply command-line overrides
    if args.topic or args.model or args.reference_num or args.num_sections or args.papers_per_section:
        # Create a custom benchmark config with overrides
        custom_params = {}

        if args.reference_num:
            custom_params["reference_num"] = args.reference_num

        if args.num_sections:
            custom_params["num_sections"] = args.num_sections

        if args.papers_per_section:
            custom_params["papers_per_section"] = args.papers_per_section

        # Run a single benchmark with overrides
        result = await benchmark.run_single_benchmark(
            topic=args.topic or "Custom Topic",
            model=args.model or benchmark.config.get("default", {}).get("model", "llama2"),
            params=custom_params
        )

        benchmark.results.append(result)
    elif args.run:
        # Run a specific benchmark set
        await benchmark.run_benchmark_set(args.run)
    else:
        # Run all benchmark sets
        await benchmark.run_all()

    # Generate report
    report_dir = benchmark.generate_report()
    print(f"Benchmark completed! Report available at: {report_dir}/report.html")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        logging.error("Benchmark failed", exc_info=True)
        sys.exit(1)
