#!/usr/bin/env python3
import logging
from pathlib import Path
from datetime import datetime
import json
from src.state import ResearchState

logger = logging.getLogger(__name__)

def save_pipeline_state(stage: str, state: ResearchState, output_dir: Path = Path("paper_states/autosurvey")) -> None:
    """Save the current AutoSurvey pipeline state"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a serializable state representation
    state_data = {
        "timestamp": timestamp,
        "stage": stage,
        "topic": state.topic,
    }

    if stage == "initial_retrieval":
        state_data.update({
            "initial_publications": [paper.to_dict() for paper in state.initial_publications],
            "structured_outline": state.structured_outline
        })
    elif stage == "subsection_drafting":
        state_data.update({
            "section_publications": {
                section: [paper.to_dict() for paper in papers]
                for section, papers in state.section_publications.items()
            },
            "section_drafts": state.section_drafts
        })
    elif stage == "integration_refinement":
        state_data.update({
            "refined_sections": state.refined_sections,
            "integrated_survey": state.integrated_survey
        })
    elif stage == "evaluation_iteration":
        state_data.update({
            "evaluation_results": state.evaluation_results,
            "iteration_surveys": state.iteration_surveys,
            "best_survey_idx": state.best_survey_idx
        })

    # Save to JSON file
    with open(output_dir / f"{stage}_{timestamp}.json", "w") as f:
        json.dump(state_data, f, indent=2)

    logger.info(f"Saved AutoSurvey pipeline state: {stage}")
