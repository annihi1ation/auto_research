# Auto Research Paper Generator

A tool for automatically generating research survey papers using Ollama and vector similarity search.

## Features

- Automated outline generation based on topic analysis
- Vector similarity search for finding relevant papers
- Content analysis and synthesis using Ollama
- IEEE-style LaTeX paper formatting
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
python -m src.main --topic "LLMs For Bioinformatics" --model qwen3 --output output_directory --reference-num 100 --use-default-outline
```

Arguments:
- `--topic`: Research topic for the survey paper (required)
- `--output`: Output directory for generated papers (default: "output")
- `--model`: Ollama model to use (default: "llama2")
- `--reference-num`: Number of reference papers to consider (default: 1500)

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
   - IEEE LaTeX formatting
   - Final paper assembly with IEEE template

## Survey Paper Generation System Documentation

### Project Overview

This system is designed to automatically generate survey papers by processing and analyzing academic papers from ArXiv. The system uses a PostgreSQL database with vector embeddings for semantic search and similarity matching.

### Project Structure

```
/data2/leyizhao/auto-research/
└── write_paper/
    ├── src/
    │   └── models/
    │       └── paper.py          # Database models for ArXiv papers
    └── README.md                 # This documentation file
```

### Core Components

#### 1. Database Models (`src/models/paper.py`)

##### ArxivPaper Model
The main data model representing academic papers from ArXiv with the following key features:

**Fields:**
- `id` (String, Primary Key): Unique identifier for the paper
- `submitter` (String): Person who submitted the paper
- `authors` (String): Paper authors
- `title` (String): Paper title
- `comments` (String): Additional comments
- `journal_ref` (String): Journal reference
- `doi` (String): Digital Object Identifier
- `report_no` (String): Report number
- `categories` (String): ArXiv categories
- `license` (String): Paper license
- `abstract` (Text): Paper abstract
- `versions` (JSONB): Version history
- `update_date` (String): Last update date
- `authors_parsed` (JSONB): Parsed author information
- `embedding` (Vector): 768-dimensional vector embedding for semantic search

##### Custom Vector Type
A custom SQLAlchemy type decorator for handling vector embeddings:
- Stores 768-dimensional vectors as strings in the database
- Automatically converts between Python lists and database string format
- Format: `[value1,value2,...,value768]`

### Key Features

#### 1. Vector Embeddings
- 768-dimensional embeddings for semantic similarity search
- Enables finding related papers based on content similarity
- Supports efficient nearest-neighbor queries

#### 2. JSONB Support
- Flexible storage for complex data structures
- Used for version history and parsed author information
- Enables rich querying capabilities

#### 3. Serialization
- `to_dict()` method for JSON serialization
- Excludes embeddings by default to reduce payload size
- Suitable for API responses and data export

### Workflow Architecture

#### Phase 1: Data Collection
1. **ArXiv Data Ingestion**
   - Fetch papers from ArXiv API
   - Parse metadata and abstracts
   - Store in PostgreSQL database

#### Phase 2: Embedding Generation
1. **Text Processing**
   - Extract relevant text (title, abstract)
   - Preprocess for embedding generation

2. **Vector Generation**
   - Generate 768-dimensional embeddings
   - Store in database for similarity search

#### Phase 3: Survey Generation
1. **Topic Analysis**
   - Identify key research areas
   - Cluster similar papers
   - Extract main themes

2. **Content Synthesis**
   - Aggregate findings from related papers
   - Generate structured survey sections
   - Create comprehensive bibliography

#### Phase 4: Output Generation
1. **Document Creation**
   - Format survey paper
   - Generate citations
   - Create final output (PDF/LaTeX/Markdown)

### Database Schema

```sql
CREATE TABLE arxiv_papers (
    id VARCHAR PRIMARY KEY,
    submitter VARCHAR,
    authors VARCHAR,
    title VARCHAR,
    comments VARCHAR,
    journal_ref VARCHAR,
    doi VARCHAR,
    report_no VARCHAR,
    categories VARCHAR,
    license VARCHAR,
    abstract TEXT,
    versions JSONB,
    update_date VARCHAR,
    authors_parsed JSONB,
    embedding VARCHAR  -- Stores vector as string
);
```

### Usage Example

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.paper import ArxivPaper, Base

# Database setup
engine = create_engine('postgresql://user:pass@localhost/arxiv_db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Query papers by category
papers = session.query(ArxivPaper).filter(
    ArxivPaper.categories.contains('cs.AI')
).all()

# Convert to JSON-serializable format
papers_data = [paper.to_dict() for paper in papers]
```

### Next Steps

To complete the survey paper generation system, consider implementing:

1. **Data Collection Module**
   - ArXiv API integration
   - Batch processing capabilities
   - Update scheduling

2. **Embedding Service**
   - Integration with embedding models (e.g., Sentence-BERT)
   - Batch embedding generation
   - Caching mechanism

3. **Survey Generation Engine**
   - Topic modeling algorithms
   - Content synthesis logic
   - Citation management

4. **Output Formatter**
   - LaTeX template generation
   - Bibliography management
   - Multiple output formats

### Dependencies

- SQLAlchemy: ORM for database interactions
- PostgreSQL: Database with JSONB support
- Vector embedding library (e.g., sentence-transformers)
- ArXiv API client

### Configuration

Ensure the following environment variables are set:
- `DATABASE_URL`: PostgreSQL connection string
- `EMBEDDING_MODEL`: Model name for generating embeddings
- `ARXIV_API_KEY`: (if required) ArXiv API credentials

## Contributing

Feel free to open issues or submit pull requests for improvements.
