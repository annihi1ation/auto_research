# Auto Research Paper Generator

A tool for automatically generating research survey papers using Ollama and vector similarity search.

## Features

- Automated outline generation based on topic analysis
- Vector similarity search for finding relevant papers
- Content analysis and synthesis using Ollama
- arXiv-style paper formatting
- Support for customizable paper generation parameters

## Requirements

- Python 3.8+
- PostgreSQL with pgvector extension
- Ollama running locally

## Installation

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd write_paper
```

2. Install dependencies:
```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Ensure PostgreSQL with pgvector is running (see docker-compose.yml in parent directory)

4. Ensure Ollama is running locally with your chosen model:
```bash
ollama run llama2  # or your preferred model
```

## Usage

Generate a survey paper:

```bash
# From the write_paper directory
python -m src.main --topic "MLLM" \
                   --output output_directory \
                   --model llama2 \
                   --reference-num 1500
```

Arguments:
- `--topic`: Research topic for the survey paper (required)
- `--output`: Output directory for generated papers (default: "output")
- `--model`: Ollama model to use (default: "llama2")
- `--reference-num`: Number of reference papers to consider (default: 1500)

## Output

The tool generates a LaTeX file in arXiv style with:
- Title and abstract
- Structured sections based on topic analysis
- Content synthesized from relevant papers
- Bibliography placeholders

## Architecture

The paper generation process follows these steps:

1. Planning Phase:
   - Topic analysis
   - Outline generation using Ollama and reference papers

2. Research Phase:
   - Paper search using vector similarity
   - Content analysis of relevant papers
   - Content synthesis for each section

3. Generation Phase:
   - Abstract generation
   - arXiv-style formatting
   - Final paper assembly

## Contributing

Feel free to open issues or submit pull requests for improvements.
