# ArXiv Paper Uploader

This script downloads and processes ArXiv papers, filtering for machine learning and AI-related categories (cs.AI, cs.CL, cs.CV, cs.LG, stat.ML) and uploads them to a PostgreSQL database.

## Prerequisites

- Python 3.12+
- uv package manager
- Docker and Docker Compose
- Kaggle API credentials (see Kaggle setup below)

## Setup Instructions

1. **Reset Docker Environment**

   ```bash
   # Stop and remove existing containers
   docker compose down

   # Start fresh containers
   docker compose up -d
   ```

2. **Python Environment Setup**

   ```bash
   # Create and activate virtual environment using uv
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Kaggle Setup**
   - Go to your Kaggle account settings (https://www.kaggle.com/settings)
   - Create a new API token
   - Download the `kaggle.json` file
   - Place the `kaggle.json` file in:
     - Linux/macOS: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
   - Set appropriate permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json  # On Unix/macOS
     ```

## Running the Script

```bash
uv run src/upload_paper.py
```

The script will:

1. Create necessary directories
2. Download the ArXiv dataset from Kaggle
3. Create the database table if it doesn't exist
4. Process and upload papers that:
   - Match the specified categories
   - Don't already exist in the database

## Monitoring Progress

- Progress is displayed in real-time with a progress bar
- Detailed logs are written to `upload_paper.log`
- The script will show which papers are being processed and skipped

## Troubleshooting

If you encounter any issues:

1. Check the `upload_paper.log` file for detailed error messages
2. Ensure Docker containers are running: `docker compose ps`
3. Verify database connection: `docker exec -it auto-research-db-1 psql -U postgres -d research_db`
4. Make sure your Kaggle API credentials are correctly set up
