#!/usr/bin/env python3
from sqlalchemy import Column, String, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

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

    def to_dict(self):
        """Convert ArxivPaper instance to a dictionary for JSON serialization"""
        # For complex objects like embedding, we need to handle them specially
        embedding_value = None
        if self.embedding:
            # If embedding is a list/vector, convert to list of floats
            if isinstance(self.embedding, list):
                embedding_value = [float(x) for x in self.embedding]
            else:
                # If it's already processed by SQLAlchemy, just use as is
                embedding_value = self.embedding

        return {
            'id': self.id,
            'submitter': self.submitter,
            'authors': self.authors,
            'title': self.title,
            'comments': self.comments,
            'journal_ref': self.journal_ref,
            'doi': self.doi,
            'report_no': self.report_no,
            'categories': self.categories,
            'license': self.license,
            'abstract': self.abstract,
            'versions': self.versions,
            'update_date': self.update_date,
            'authors_parsed': self.authors_parsed,
            # Don't include embedding to save space - it can be very large
            # 'embedding': embedding_value
        }
