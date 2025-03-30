#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
import logging

from src.state import ResearchState
from src.utils.ollama import OllamaClient
from src.utils.db import DatabaseManager
from .utils import save_pipeline_state
from .stage2 import Stage2_SubsectionDrafting

logger = logging.getLogger(__name__)

@dataclass
class Stage1_InitialRetrieval(BaseNode[ResearchState]):
    """Stage 1: Initial Retrieval & Outline Generation

    This stage retrieves initial publications from a database and generates
    a structured outline for the survey paper.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Stage2_SubsectionDrafting:
        logger.info("Stage 1: Initial Retrieval & Outline Generation")
        logger.info("DEBUG: Starting initial retrieval & outline generation process")

        # Initialize clients
        db = DatabaseManager()
        ollama = OllamaClient()
        logger.info("DEBUG: Initialized DatabaseManager and OllamaClient")

        # Step 1: Retrieve initial publications
        logger.info("Retrieving initial publications...")
        query_embedding = ctx.state.embedding_model.encode(ctx.state.topic).tolist()
        logger.info(f"DEBUG: Generated query embedding for topic: {ctx.state.topic}")

        initial_publications = db.find_similar_papers(query_embedding, limit=ctx.state.outline_config.reference_num)
        ctx.state.initial_publications = initial_publications

        logger.info(f"Retrieved {len(initial_publications)} initial publications")
        if len(initial_publications) > 0:
            logger.info(f"DEBUG: First publication: {initial_publications[0].title}")
            logger.info(f"DEBUG: Last publication: {initial_publications[-1].title}")

        # Step 2: Create a fixed structured outline with specific sections
        logger.info("Creating structured outline with standard research paper sections...")

        # Create a fixed structure with the requested sections
        structured_outline = {
            "title": f"Research on {ctx.state.topic}",
            "sections": [
                {
                    "title": "Introduction",
                    "description": "Overview of the research topic, research questions, objectives, and importance of the study.",
                    "subsections": [
                        {"title": "Research Background", "key_points": ["Context of the research", "Importance of the topic"]},
                        {"title": "Research Questions and Objectives", "key_points": ["Primary research questions", "Specific objectives"]},
                        {"title": "Significance of the Study", "key_points": ["Theoretical contributions", "Practical implications"]}
                    ]
                },
                {
                    "title": "Literature Review",
                    "description": "Summary of existing research related to the topic, pointing out gaps the study will address.",
                    "subsections": [
                        {"title": "Theoretical Framework", "key_points": ["Key theories and models", "Conceptual foundations"]},
                        {"title": "Previous Studies", "key_points": ["Related work", "Current state of knowledge"]},
                        {"title": "Research Gaps", "key_points": ["Limitations in existing literature", "Opportunities for contribution"]}
                    ]
                },
                {
                    "title": "Methodology",
                    "description": "Survey design, participant selection, recruitment, and data collection instruments.",
                    "subsections": [
                        {"title": "Research Design", "key_points": ["Survey approach", "Research framework"]},
                        {"title": "Participant Selection", "key_points": ["Sampling methods", "Inclusion criteria"]},
                        {"title": "Data Collection", "key_points": ["Survey instruments", "Questionnaire design", "Distribution methods"]}
                    ]
                },
                {
                    "title": "Results",
                    "description": "Findings of the survey analysis, using tables and graphs to illustrate key points.",
                    "subsections": [
                        {"title": "Demographic Information", "key_points": ["Participant characteristics", "Sample profile"]},
                        {"title": "Key Findings", "key_points": ["Main survey results", "Statistical analysis"]},
                        {"title": "Data Visualization", "key_points": ["Tables and charts", "Result patterns"]}
                    ]
                },
                {
                    "title": "Discussion",
                    "description": "Interpretation of results in relation to research questions, comparison with existing literature, and implications.",
                    "subsections": [
                        {"title": "Interpretation of Findings", "key_points": ["Analysis of results", "Patterns and trends"]},
                        {"title": "Comparison with Literature", "key_points": ["Alignment with previous studies", "Novel insights"]},
                        {"title": "Limitations", "key_points": ["Study constraints", "Methodological challenges"]}
                    ]
                },
                {
                    "title": "Conclusion",
                    "description": "Summary of main findings, importance of the study, and suggestions for further research.",
                    "subsections": [
                        {"title": "Summary of Findings", "key_points": ["Key takeaways", "Research contributions"]},
                        {"title": "Implications", "key_points": ["Theoretical implications", "Practical applications"]},
                        {"title": "Future Research", "key_points": ["Research opportunities", "Recommended directions"]}
                    ]
                }
            ]
        }

        # Store structured outline
        ctx.state.structured_outline = structured_outline

        # Also populate the flat outline for backward compatibility
        flat_outline = ["Abstract"]
        for section in structured_outline.get("sections", []):
            flat_outline.append(section["title"])
            for subsection in section.get("subsections", []):
                flat_outline.append(f"{section['title']} - {subsection['title']}")
        flat_outline.append("References")

        ctx.state.outline = flat_outline

        logger.info(f"Created structured outline with {len(structured_outline.get('sections', []))} main sections")

        # Save the state
        save_pipeline_state("initial_retrieval", ctx.state)

        # Output Stage 1 results
        results = {
            "stage": "Initial Retrieval & Outline Generation",
            "publications_retrieved": len(ctx.state.initial_publications),
            "structured_outline": ctx.state.structured_outline,
            "section_count": len(ctx.state.structured_outline.get("sections", [])),
            "sample_publications": [
                {"title": paper.title, "authors": paper.authors}
                for paper in ctx.state.initial_publications[:5]  # Show first 5 papers
            ]
        }

        # Print the results
        logger.info("=== STAGE 1 RESULTS ===")
        logger.info(f"Publications retrieved: {results['publications_retrieved']}")
        logger.info(f"Main sections: {results['section_count']}")
        logger.info("Section titles:")
        for i, section in enumerate(ctx.state.structured_outline.get("sections", [])):
            logger.info(f"  {i+1}. {section['title']} ({len(section.get('subsections', []))} subsections)")
        logger.info("Sample publications:")
        for i, pub in enumerate(results['sample_publications']):
            logger.info(f"  {i+1}. {pub['title']} by {pub['authors']}")

        # Store results in state for later reference
        ctx.state.stage_results = ctx.state.stage_results or {}
        ctx.state.stage_results["stage1"] = results

        return Stage2_SubsectionDrafting()
