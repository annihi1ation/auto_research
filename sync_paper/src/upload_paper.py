#!/usr/bin/env python3
import os
from pathlib import Path
import zipfile
import json
import asyncio
import pandas as pd
from sqlalchemy import create_engine, text, Column, String, Text, inspect, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, Session
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Annotated, List
from pydantic_graph import BaseNode, End, Graph, GraphRunContext, Edge
import logging
from io import StringIO
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload_paper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy Base
Base = declarative_base()

class Vector(TypeDecorator):
    impl = String
    cache_ok = True

    def __init__(self, dimensions: int = 768):
        super().__init__()
        self.dimensions = dimensions

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return f'[{",".join(map(str, value))}]'

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return [float(x) for x in value.strip('[]').split(',')]

class ArxivPaper(Base):
    __tablename__ = 'arxiv_papers'

    id = Column(String, primary_key=True)
    submitter = Column(String)
    authors = Column(String)
    title = Column(String)
    comments = Column(String)
    journal_ref = Column(String)
    doi = Column(String)
    report_no = Column(String)
    categories = Column(String)
    license = Column(String)
    abstract = Column(Text)
    versions = Column(JSONB)
    update_date = Column(String)
    authors_parsed = Column(JSONB)
    embedding = Column(Vector(768))

# Load Kaggle credentials from local kaggle.json
logger.debug("Loading Kaggle credentials from local file")
current_dir = Path(__file__).parent.parent
kaggle_json_path = current_dir / "kaggle.json"
logger.debug(f"Looking for kaggle.json at: {kaggle_json_path}")
with open(kaggle_json_path) as f:
    credentials = json.load(f)
    os.environ["KAGGLE_USERNAME"] = credentials["username"]
    os.environ["KAGGLE_KEY"] = credentials["key"]
logger.debug("Kaggle credentials loaded and environment variables set")

# Import Kaggle API after setting credentials
from kaggle.api.kaggle_api_extended import KaggleApi
logger.debug("Kaggle API imported successfully")

# Initialize embedding model
model_choose = "nomic-ai/nomic-embed-text-v2-moe"
embedding_model = SentenceTransformer(model_choose, trust_remote_code=True)
embedding_model.to(torch.device('mps'))

# Categories to filter
CATEGORIES_TO_FILTER = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'stat.ML']

@dataclass
class UploadState:
    """State for the upload workflow"""
    dataset_name: str = "Cornell-University/arxiv"
    save_path: str = "data"
    table_name: str = "arxiv_papers"
    connection_string: str = "postgresql://postgres:postgres@localhost:5451/research_db"
    current_status: str = "Not started"
    total_rows: int = 0
    processed_rows: int = 0

@dataclass
class SetupDirectories(BaseNode[UploadState]):
    """Setup necessary directories for data download"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "DownloadDataset":
        logger.info("Setting up directories")
        Path(ctx.state.save_path).mkdir(parents=True, exist_ok=True)
        ctx.state.current_status = f"Created directory at {ctx.state.save_path}"
        logger.info(f"Created directory at {ctx.state.save_path}")
        return DownloadDataset()

@dataclass
class DownloadDataset(BaseNode[UploadState]):
    """Download dataset from Kaggle"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "UnzipFiles":
        logger.info("Starting dataset download from Kaggle")
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            ctx.state.dataset_name,
            path=ctx.state.save_path,
            unzip=True
        )
        ctx.state.current_status = f"Downloaded dataset to {ctx.state.save_path}"
        logger.info(f"Downloaded dataset to {ctx.state.save_path}")
        return UnzipFiles()

@dataclass
class UnzipFiles(BaseNode[UploadState]):
    """Unzip downloaded files"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "CheckDatabase":
        logger.info("Starting file extraction")
        for file in os.listdir(ctx.state.save_path):
            if file.endswith(".zip"):
                logger.info(f"Extracting {file}")
                with zipfile.ZipFile(os.path.join(ctx.state.save_path, file), "r") as zip_ref:
                    zip_ref.extractall(ctx.state.save_path)
        ctx.state.current_status = "Files unzipped successfully"
        logger.info("Files unzipped successfully")
        return CheckDatabase()

@dataclass
class CheckDatabase(BaseNode[UploadState]):
    """Check database state and create table if needed"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "CreateTable | ReadData":
        logger.info("Checking database state")
        engine = create_engine(ctx.state.connection_string)

        # Create vector extension
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()

        inspector = inspect(engine)
        if not inspector.has_table(ctx.state.table_name):
            ctx.state.current_status = "Table does not exist"
            logger.info("Table does not exist, proceeding to create")
            return CreateTable()
        logger.info("Table already exists")
        return ReadData()

@dataclass
class CreateTable(BaseNode[UploadState]):
    """Create the database table"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "ReadData":
        logger.info(f"Creating table {ctx.state.table_name}")
        engine = create_engine(ctx.state.connection_string)

        Base.metadata.create_all(engine)

        ctx.state.current_status = f"Created table {ctx.state.table_name}"
        logger.info(f"Successfully created table {ctx.state.table_name}")
        return ReadData()

@dataclass
class ReadData(BaseNode[UploadState]):
    """Read and sync the dataset from JSON file directly to database"""
    async def run(self, ctx: GraphRunContext[UploadState]) -> "End[str]":
        logger.info("Starting to read and sync JSON dataset")
        json_file = os.path.join(ctx.state.save_path, "arxiv-metadata-oai-snapshot.json")

        # Create engine with longer timeout
        engine = create_engine(
            ctx.state.connection_string,
            connect_args={'connect_timeout': 30}
        )

        chunk_size = 10000
        total_rows = 0
        processed_rows = 0

        try:
            with open(json_file, 'r') as f:
                # Count total lines first
                total_lines = sum(1 for _ in f)
                f.seek(0)  # Reset file pointer
                ctx.state.total_rows = total_lines

                with tqdm(total=total_lines, desc="Processing") as pbar:
                    while True:
                        chunk = []
                        for _ in range(chunk_size):
                            line = f.readline()
                            if not line:
                                break
                            chunk.append(line)
                            total_rows += 1

                        if not chunk:
                            break

                        # Process chunk
                        chunk_df = pd.read_json(
                            StringIO('\n'.join(chunk)),
                            lines=True,
                            dtype={
                                'id': str,
                                'submitter': str,
                                'authors': str,
                                'title': str,
                                'comments': str,
                                'journal-ref': str,
                                'doi': str,
                                'report-no': str,
                                'categories': str,
                                'license': str,
                                'abstract': str,
                                'versions': object,
                                'update_date': str,
                                'authors_parsed': object
                            }
                        )

                        # Filter by categories
                        chunk_df = chunk_df[chunk_df['categories'].apply(
                            lambda x: any(cat in x.split() for cat in CATEGORIES_TO_FILTER)
                        )]

                        if len(chunk_df) == 0:
                            pbar.update(len(chunk))
                            continue

                        # Check existing papers
                        with Session(engine) as session:
                            existing_ids = {
                                id_[0] for id_ in session.query(ArxivPaper.id).filter(
                                    ArxivPaper.id.in_(chunk_df['id'].tolist())
                                ).all()
                            }

                            # Remove existing papers
                            chunk_df = chunk_df[~chunk_df['id'].isin(existing_ids)]

                            if len(chunk_df) == 0:
                                pbar.update(len(chunk))
                                continue

                            # Prepare data
                            chunk_df = chunk_df.rename(columns={
                                'journal-ref': 'journal_ref',
                                'report-no': 'report_no'
                            })

                            # Generate embeddings
                            logger.info("Generating embeddings for abstracts")
                            embeddings = []
                            for abstract in chunk_df['abstract']:
                                if abstract:
                                    encoded = embedding_model.encode(abstract)
                                    embeddings.append(encoded.tolist())
                                else:
                                    embeddings.append(None)
                            chunk_df['embedding'] = embeddings

                            # Create ORM objects and bulk insert
                            papers = []
                            for _, row in chunk_df.iterrows():
                                paper = ArxivPaper(
                                    id=row['id'],
                                    submitter=row['submitter'],
                                    authors=row['authors'],
                                    title=row['title'],
                                    comments=row['comments'],
                                    journal_ref=row['journal_ref'],
                                    doi=row['doi'],
                                    report_no=row['report_no'],
                                    categories=row['categories'],
                                    license=row['license'],
                                    abstract=row['abstract'],
                                    versions=row['versions'],
                                    update_date=row['update_date'],
                                    authors_parsed=row['authors_parsed'],
                                    embedding=row['embedding']
                                )
                                papers.append(paper)

                            session.bulk_save_objects(papers)
                            session.commit()

                            processed_rows += len(papers)
                            ctx.state.processed_rows = processed_rows
                            logger.info(f"Inserted {len(papers)} new papers")

                        pbar.update(len(chunk))

                        ctx.state.current_status = f"Processed {processed_rows} of {total_lines} rows"

            final_message = f"Successfully processed {processed_rows} rows"
            logger.info(final_message)
            return End(final_message)

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return End(f"Failed: {error_msg}")

# Define the graph
upload_graph = Graph(
    nodes=[
        SetupDirectories,
        DownloadDataset,
        UnzipFiles,
        CheckDatabase,
        CreateTable,
        ReadData
    ]
)

async def main():
    """Main function to run the upload workflow"""
    try:
        logger.info("Starting upload workflow")
        state = UploadState()
        result, history = await upload_graph.run(CheckDatabase(), state=state)
        logger.info(f"Final result: {result}")
        print(f"\nFinal result: {result}")
        print("\nWorkflow history:")
        for step in history:
            if hasattr(step, 'node'):
                print(f"- {step.node.__class__.__name__}: {state.current_status}")
            else:
                print(f"- End: {state.current_status}")

    except Exception as e:
        error_msg = f"Error during upload process: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        raise e

if __name__ == "__main__":
    asyncio.run(main())
