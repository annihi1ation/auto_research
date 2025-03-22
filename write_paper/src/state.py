#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from .models.paper import ArxivPaper
from dataclasses import field, dataclass

@dataclass
class OutlineConfig:
    """Configuration for outline generation"""
    reference_num: int = 1500
    model: str = "llama2"

@dataclass
class ResearchState:
    """State for the research paper generation workflow"""
    topic: str
    outline_config: OutlineConfig = field(default_factory=OutlineConfig)
    outline: List[str] = field(default_factory=list)
    related_papers: Dict[str, ArxivPaper] = field(default_factory=dict)
    generated_sections: Dict[str, str] = field(default_factory=dict)
    current_phase: str = "init"
    embedding_model: Optional[SentenceTransformer] = None

    def __post_init__(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v2-moe",
                trust_remote_code=True
            )
