#!/usr/bin/env python3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from .models.paper import ArxivPaper
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class OutlineConfig:
    """Configuration for outline generation"""
    reference_num: int = 1500
    model: str = "llama2"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OutlineConfig":
        return cls(**data)

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

    def to_dict(self) -> dict:
        """Convert state to a JSON-serializable dictionary"""
        try:
            state_dict = {
                "topic": self.topic,
                "outline_config": self.outline_config.to_dict(),
                "outline": self.outline,
                "related_papers": {k: paper.to_dict() for k, paper in self.related_papers.items()},
                "generated_sections": self.generated_sections,
                "current_phase": self.current_phase,
                "initial_publications": [paper.to_dict() for paper in self.initial_publications],
                "structured_outline": self.structured_outline,
                "section_publications": {
                    k: [paper.to_dict() for paper in papers]
                    for k, papers in self.section_publications.items()
                },
                "section_drafts": self.section_drafts,
                "refined_sections": self.refined_sections,
                "integrated_survey": self.integrated_survey,
                "evaluation_results": self.evaluation_results,
                "iteration_surveys": self.iteration_surveys,
                "best_survey_idx": self.best_survey_idx,
                "stage_results": self.stage_results
            }
            return state_dict
        except Exception as e:
            logger.error(f"Error converting state to dict: {str(e)}")
            raise

    def save_state(self, output_dir: Path = Path("paper_states")) -> Path:
        """
        Save the current state to a JSON file

        Returns:
            Path: Path to the saved state file
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            state_dict = self.to_dict()
            state_dict["metadata"] = {
                "timestamp": timestamp,
                "phase": self.current_phase,
                "topic": self.topic,
                "version": "1.0"  # Added versioning
            }

            json_file = output_dir / f"research_state_{self.current_phase}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(state_dict, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved state to {json_file}")
            return json_file
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            raise

    @classmethod
    def load_state(cls, file_path: Path) -> "ResearchState":
        """
        Load state from a JSON file

        Args:
            file_path: Path to the state file

        Returns:
            ResearchState: Reconstructed state object
        """
        try:
            logger.info(f"Loading state from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)

            # Remove metadata if present
            metadata = state_dict.pop("metadata", {})
            logger.info(f"Loading state from phase: {metadata.get('phase', 'unknown')}")

            # Reconstruct OutlineConfig
            outline_config = OutlineConfig.from_dict(state_dict.pop("outline_config"))

            # Reconstruct ArxivPaper objects
            related_papers = {
                k: ArxivPaper(**paper_dict)
                for k, paper_dict in state_dict.pop("related_papers").items()
            }

            initial_publications = [
                ArxivPaper(**paper_dict)
                for paper_dict in state_dict.pop("initial_publications")
            ]

            section_publications = {
                k: [ArxivPaper(**paper_dict) for paper_dict in papers]
                for k, papers in state_dict.pop("section_publications").items()
            }

            # Create new state instance
            state = cls(
                topic=state_dict.pop("topic"),
                outline_config=outline_config,
                related_papers=related_papers,
                initial_publications=initial_publications,
                section_publications=section_publications,
                **state_dict
            )

            logger.info("Successfully loaded state")
            return state

        except Exception as e:
            logger.error(f"Failed to load state from {file_path}: {str(e)}")
            raise
