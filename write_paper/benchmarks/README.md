# AI Research Topic Benchmarking System

This benchmark system allows you to evaluate the performance of different language models across various AI research topics. It automates the process of generating research papers on different topics and provides detailed metrics and comparisons.

## Features

- Benchmark multiple language models on various AI research topics
- Configurable parameters (reference numbers, sections, papers per section)
- Detailed HTML reports with metrics and comparisons
- Flexible configuration using YAML files
- Command-line interface for easy execution

## Requirements

- Python 3.7+
- Required packages: pyyaml, asyncio, pathlib
- Access to the language models specified in configurations

## Quick Start

Run a quick benchmark test:

```bash
./run_benchmark.py --run quick_test
```

This will run the `quick_test` benchmark defined in the default configuration file.

## Configuration

Benchmarks are configured using YAML files in the `configs/` directory. The default configuration is in `configs/default.yaml`.

Each configuration file defines:

1. Default parameters used for all benchmarks
2. List of models to benchmark
3. List of topics to benchmark
4. Custom benchmark configurations that override the defaults

### Example Configuration

```yaml
default:
  output_dir: "result"
  reference_num: 100
  num_sections: 8
  papers_per_section: 5

models:
  - name: "llama3:8b"
    description: "Llama 3 8B model"
  - name: "mistral:7b"
    description: "Mistral 7B model"

topics:
  - name: "Multimodal Large Language Models (MLLMs)"
    description: "Research on models that combine vision, text and other modalities"
  - name: "Retrieval-Augmented Generation (RAG) and Memory Systems"
    description: "Research on retrieval, context integration, and memory in LLMs"

benchmarks:
  quick_test:
    description: "Quick test with minimal settings"
    models: ["llama3:8b"]
    topics: ["Retrieval-Augmented Generation (RAG) and Memory Systems"]
    reference_num: 50
    num_sections: 4
    papers_per_section: 3
```

## Command-Line Options

```
usage: run_benchmark.py [-h] [--config CONFIG] [--output OUTPUT] [--run RUN]
                        [--topic TOPIC] [--model MODEL]
                        [--reference-num REFERENCE_NUM]
                        [--num-sections NUM_SECTIONS]
                        [--papers-per-section PAPERS_PER_SECTION]

Run benchmarks for the research paper generation system

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the benchmark config file
  --output OUTPUT       Output directory for benchmark results
  --run RUN             Specific benchmark set to run (omit to run all)
  --topic TOPIC         Specific topic to run (overrides config)
  --model MODEL         Specific model to run (overrides config)
  --reference-num REFERENCE_NUM
                        Number of references (overrides config)
  --num-sections NUM_SECTIONS
                        Number of sections (overrides config)
  --papers-per-section PAPERS_PER_SECTION
                        Papers per section (overrides config)
```

## Examples

### Run All Benchmarks

```bash
./run_benchmark.py
```

### Run a Specific Benchmark

```bash
./run_benchmark.py --run comprehensive
```

### Run with Custom Configuration

```bash
./run_benchmark.py --config configs/specialized.yaml --run alignment_focus
```

### Run a Single Topic/Model Combination

```bash
./run_benchmark.py --topic "Multimodal Large Language Models (MLLMs)" --model "llama3:8b" --reference-num 75 --num-sections 5 --papers-per-section 4
```

## Output and Reports

Benchmark results are saved in the `benchmark_results/` directory (unless overridden with `--output`). Each benchmark run creates a timestamped subdirectory containing:

1. Generated papers for each topic/model combination
2. A `report/` directory with:
   - `results.json`: Raw benchmark result data
   - `summary.json`: Summary statistics
   - `report.html`: Interactive HTML report with tables and charts

The HTML report includes:

- Overall success rates and durations
- Model-specific performance metrics
- Topic-specific performance metrics
- Detailed results table with all combinations

## Metrics

The benchmark collects various metrics for each run, including:

- Success/failure status
- Duration (execution time)
- Number of papers found and used
- Number of sections generated
- Estimated word and token counts
- (Custom metrics can be added in the `_calculate_metrics` method)

## Adding New Benchmarks

To add new benchmarks:

1. Create a new configuration file in `configs/`
2. Define your models, topics, and benchmark configurations
3. Run the benchmark with `--config` pointing to your new configuration file
