from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext

@dataclass
class ResearchPaper:
    title: str
    authors: list[str]
    abstract: str
    url: str

