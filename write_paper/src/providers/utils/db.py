#!/usr/bin/env python3
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np
from ...models.paper import ArxivPaper

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string: str = "postgresql://postgres:postgres@localhost:5451/research_db"):
        self.engine = create_engine(connection_string)
        self.embedding_model = None

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using sentence transformer"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v2-moe",
                trust_remote_code=True
            )
        return self.embedding_model.encode(text).tolist()

    def find_similar_papers(self, embedding: List[float], limit: int = 5) -> List[ArxivPaper]:
        """Find papers with similar embeddings using cosine similarity computed in Python"""
        try:
            logger.info("Starting vector similarity search with limit: %d", limit)
            with Session(self.engine) as session:
                logger.debug("Computing similarity in Python (pgvector not available)")

                # Get all papers with embeddings (limit to reasonable number for performance)
                query = text("""
                    SELECT * FROM arxiv_papers
                    WHERE embedding IS NOT NULL
                    LIMIT 1000
                """)

                logger.debug("Fetching papers for similarity computation")
                result = session.execute(query)

                papers_with_similarity = []
                target_embedding = np.array(embedding)

                for row in result:
                    try:
                        # Convert row to dictionary properly
                        row_dict = dict(row._mapping)

                        # Parse the embedding string back to array
                        embedding_str = row_dict.get('embedding')
                        if embedding_str:
                            # Remove brackets and split by comma
                            embedding_values = embedding_str.strip('[]').split(',')
                            paper_embedding = np.array([float(x.strip()) for x in embedding_values])

                            # Compute cosine similarity
                            similarity = np.dot(target_embedding, paper_embedding) / (
                                np.linalg.norm(target_embedding) * np.linalg.norm(paper_embedding)
                            )

                            paper_dict = {
                                'id': row_dict.get('id'),
                                'submitter': row_dict.get('submitter'),
                                'authors': row_dict.get('authors'),
                                'title': row_dict.get('title'),
                                'comments': row_dict.get('comments'),
                                'journal_ref': row_dict.get('journal_ref'),
                                'doi': row_dict.get('doi'),
                                'report_no': row_dict.get('report_no'),
                                'categories': row_dict.get('categories'),
                                'license': row_dict.get('license'),
                                'abstract': row_dict.get('abstract'),
                                'versions': row_dict.get('versions'),
                                'update_date': row_dict.get('update_date'),
                                'authors_parsed': row_dict.get('authors_parsed'),
                                'embedding': row_dict.get('embedding')
                            }

                            papers_with_similarity.append((similarity, ArxivPaper(**paper_dict)))

                    except Exception as e:
                        logger.error("Error processing paper row: %s", str(e))
                        continue

                # Sort by similarity (highest first) and return top papers
                papers_with_similarity.sort(key=lambda x: x[0], reverse=True)
                papers = [paper for similarity, paper in papers_with_similarity[:limit]]

                logger.info("Found %d similar papers", len(papers))

                # Log paper titles for debugging
                for i, paper in enumerate(papers, 1):
                    logger.debug("Paper %d: %s", i, paper.title)

                return papers

        except Exception as e:
            logger.error("Error in vector search: %s", str(e), exc_info=True)
            return []

    def get_paper_by_id(self, paper_id: str) -> Optional[ArxivPaper]:
        """Retrieve a paper by its ID"""
        try:
            logger.info("Retrieving paper with ID: %s", paper_id)
            with Session(self.engine) as session:
                paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
                if paper:
                    logger.debug("Found paper: %s", paper.title)
                else:
                    logger.debug("No paper found with ID: %s", paper_id)
                return paper
        except Exception as e:
            logger.error("Error retrieving paper: %s", str(e), exc_info=True)
            return None
