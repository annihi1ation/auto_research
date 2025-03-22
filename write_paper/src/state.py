#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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

    # AutoSurvey Pipeline specific fields
    initial_publications: List[ArxivPaper] = field(default_factory=list)
    structured_outline: Dict[str, Any] = field(default_factory=dict)
    section_publications: Dict[str, List[ArxivPaper]] = field(default_factory=dict)
    section_drafts: Dict[str, str] = field(default_factory=dict)
    refined_sections: Dict[str, str] = field(default_factory=dict)
    integrated_survey: str = ""
    evaluation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    iteration_surveys: List[Dict[str, Any]] = field(default_factory=list)
    best_survey_idx: int = -1
    stage_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding_model is None:
            try:
                # First attempt with standard initialization
                self.embedding_model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v2-moe",
                    trust_remote_code=True
                )
            except FileNotFoundError as e:
                if "Pooling/config.json" in str(e):
                    # Handle missing Pooling configuration
                    from sentence_transformers import models

                    # Load the transformer model
                    word_embedding_model = models.Transformer(
                        "nomic-ai/nomic-embed-text-v2-moe",
                        trust_remote_code=True
                    )

                    # Create a mean pooling layer manually
                    pooling_model = models.Pooling(
                        word_embedding_model.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=True
                    )

                    # Create the SentenceTransformer with these components
                    self.embedding_model = SentenceTransformer(
                        modules=[word_embedding_model, pooling_model]
                    )
                else:
                    # If it's a different error, re-raise it
                    raise
