# Auto Research

A monorepo for automating research workflows: discovering literature, planning and generating survey/paper drafts, and syncing/uploading artifacts. It contains two primary apps:

- `write_paper/` — research agent pipelines to plan, generate, and assemble papers (LaTeX/Markdown), including benchmarking utilities.
- `sync_paper/` — utilities and pipelines to generate diagrams and upload/sync paper-related outputs; includes a simple Docker setup.

Refer to each sub-project’s README for detailed usage and configuration.

## Repository structure

```
auto-research/
├─ write_paper/           # Paper generation pipelines, configs, and providers
│  ├─ src/                # Core library and nodes (planning, research, generation)
│  ├─ topics.yaml         # High-level topics list
│  ├─ benchmarks/         # Benchmark configs and runner
│  ├─ requirements.txt    # Python deps for this app
│  └─ README.md           # Detailed instructions
├─ sync_paper/            # Diagram generation and upload/sync utilities
│  ├─ src/                # Research agent + upload logic
│  ├─ docker-compose.yml  # Optional containerized setup
│  ├─ requirements.txt    # Python deps for this app
│  └─ README.md           # Detailed instructions
└─ resources/             # Shared tools and helper scripts
```

## Prerequisites

- Linux or macOS recommended (tested on Linux)
- Python 3.10+
- Git
- Optional: Docker (for `sync_paper`)

## Setup

Create and activate a virtual environment, then install dependencies per module.

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate

# Install dependencies for each app
pip install -r write_paper/requirements.txt
pip install -r sync_paper/requirements.txt
```

If you prefer to keep environments isolated per app, create a separate venv in each folder and install only that app’s requirements.

## Quick start

### Write and generate papers (`write_paper/`)

- Generate papers via the library entry point:

```bash
python write_paper/src/main.py
```

- Generate multiple papers using topics/configs:

```bash
python write_paper/generate_multiple_papers.py \
  --topics write_paper/topics.yaml
```

- Benchmarks:

```bash
python write_paper/benchmarks/run_benchmark.py \
  --config write_paper/benchmarks/configs/default.yaml
```

See `write_paper/README.md` for providers (OpenAI, OpenRouter, Ollama), configuration files, and output directories.

### Sync and upload artifacts (`sync_paper/`)

- Run directly with Python:

```bash
python sync_paper/run.py
```

- Or use Docker Compose (optional):

```bash
docker compose -f sync_paper/docker-compose.yml up --build
```

Check `sync_paper/README.md` for details on inputs (e.g., arXiv metadata), outputs, and environment variables.

## Testing

This repo uses per-app tests. For example, to run tests in `sync_paper`:

```bash
pytest -q sync_paper/test
```

Add tests alongside each app’s `test/` folder.

## Data and resources

- Some utilities expect arXiv metadata JSON files under `write_paper/json2bibtex/data/` or `sync_paper/data/`.
- Additional helper tools live in `resources/tools/` and `write_paper/*` subfolders.

## Contributing

- Keep changes scoped to the relevant app folder (`write_paper` or `sync_paper`).
- Add or update tests when changing behavior.
- Use clear commit messages and include minimal repro steps when relevant.

## License

Specify your preferred license for this repository (e.g., MIT, Apache-2.0). If unsure, add a `LICENSE` file later and update this section.
