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
