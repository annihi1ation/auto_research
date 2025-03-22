# Auto Research Paper Generator

A tool for automatically generating research survey papers using Ollama and vector similarity search.

## Features

- Automated outline generation based on topic analysis
- Vector similarity search for finding relevant papers
- Content analysis and synthesis using Ollama
- IEEE-style LaTeX paper formatting
- Support for customizable paper generation parameters
- **NEW: AutoSurvey Pipeline** - A comprehensive 4-stage pipeline for generating high-quality surveys

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

### Using AutoSurvey Pipeline

To use the new AutoSurvey pipeline:

```bash
# From the write_paper directory
python -m src.main --topic "MLLM" \
                   --output output_directory \
                   --model llama2 \
                   --reference-num 1500 \
                   --autosurvey
```

Arguments:
- `--topic`: Research topic for the survey paper (required)
- `--output`: Output directory for generated papers (default: "output")
- `--model`: Ollama model to use (default: "llama2")
- `--reference-num`: Number of reference papers to consider (default: 1500)
- `--autosurvey`: Use the AutoSurvey pipeline for enhanced survey generation

## Output

The tool generates a LaTeX file in IEEE conference/journal style with:
- IEEE document class and required packages
- Title, authors, and abstract formatted to IEEE specifications
- Structured sections based on topic analysis
- Content synthesized from relevant papers
- IEEE-style citations and bibliography
- Ready to compile with IEEE LaTeX template

### Requirements for LaTeX Compilation
- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- IEEE LaTeX template (`IEEEtran.cls`)
- BibTeX for reference management

## Architecture

### Standard Pipeline

The standard paper generation process follows these steps:

1. Planning Phase:
   - Topic analysis
   - Outline generation using Ollama and reference papers

2. Research Phase:
   - Paper search using vector similarity
   - Content analysis of relevant papers
   - Content synthesis for each section

3. Generation Phase:
   - Abstract generation
   - IEEE LaTeX formatting
   - Final paper assembly with IEEE template

### AutoSurvey Pipeline

The AutoSurvey pipeline is a more comprehensive approach with the following stages:

1. **Stage 1: Initial Retrieval & Outline Generation**
   - Retrieves publications from a database
   - Generates a structured hierarchical outline

2. **Stage 2: Subsection Drafting**
   - Retrieves relevant publications for each section
   - Drafts each section and subsection of the outline

3. **Stage 3: Integration & Refinement**
   - Refines each section
   - Integrates the refined sections into a cohesive survey

4. **Stage 4: Rigorous Evaluation & Iteration**
   - Evaluates the survey based on coverage, structure, relevance, and faithfulness
   - Iterates to improve the survey
   - Selects the best version

The AutoSurvey pipeline produces higher quality surveys with:
- More comprehensive coverage
- Better structured content with hierarchical organization
- Targeted retrieval of relevant papers for each section
- Iterative refinement process
- Quality evaluation with concrete metrics

## Contributing

Feel free to open issues or submit pull requests for improvements.
