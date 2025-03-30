#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
import logging

from src.state import ResearchState
from src.utils.ollama import OllamaClient
from .utils import save_pipeline_state
from .stage4 import Stage4_EvaluationIteration

logger = logging.getLogger(__name__)

@dataclass
class Stage3_IntegrationRefinement(BaseNode[ResearchState]):
    """Stage 3: Integration & Refinement

    This stage refines each section/subsection and integrates them into a
    cohesive survey paper.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Stage4_EvaluationIteration:
        logger.info("Stage 3: Integration & Refinement")
        logger.info("DEBUG: Starting integration & refinement process")

        # Initialize client
        ollama = OllamaClient()

        # Initialize storage for refined sections
        ctx.state.refined_sections = {}

        # First, refine each section and subsection
        for section_title, draft_content in ctx.state.section_drafts.items():
            logger.info(f"Refining section: {section_title}")

            # Skip abstract for now
            if section_title.lower() == "abstract":
                continue

            # Create refinement prompt
            refinement_prompt = f"""Refine the following draft content for the "{section_title}" section of a survey paper on "{ctx.state.topic}".

Draft content:
{draft_content}

Your task:
1. Improve the organization and flow
2. Enhance clarity and precision
3. Ensure academic writing style and formal tone
4. Add transitional phrases for better cohesion
5. Eliminate redundancies and verbosity
6. Fix any grammatical or stylistic issues
7. Ensure technical accuracy and proper terminology
8. Maintain citation style consistency (use \cite{key} format)
9. Keep the refined content comprehensive yet concise

Provide the refined section content:
"""

            refined_content = await ollama.generate(ctx.state.outline_config.model, refinement_prompt)
            ctx.state.refined_sections[section_title] = refined_content

        # Now, integrate all sections into a cohesive paper
        logger.info("Integrating refined sections into cohesive paper")

        # Build the paper structure
        paper_sections = []

        # Add main sections in order, with their subsections
        for section in ctx.state.structured_outline.get("sections", []):
            section_title = section["title"]
            if section_title in ctx.state.refined_sections:
                paper_sections.append(f"# {section_title}\n\n{ctx.state.refined_sections[section_title]}")

            # Add subsections
            for subsection in section.get("subsections", []):
                subsection_title = subsection["title"]
                full_subsection_title = f"{section_title} - {subsection_title}"
                if full_subsection_title in ctx.state.refined_sections:
                    paper_sections.append(f"## {subsection_title}\n\n{ctx.state.refined_sections[full_subsection_title]}")

        # Add conclusion
        if "Conclusion" in ctx.state.refined_sections:
            paper_sections.append(f"# Conclusion\n\n{ctx.state.refined_sections['Conclusion']}")

        # Now that we have the full paper, generate a proper abstract
        integrated_paper = "\n\n".join(paper_sections)
        ctx.state.integrated_survey = integrated_paper

        # After integrating sections
        logger.info(f"DEBUG: Integrated {len(ctx.state.refined_sections)} sections")
        for title in sorted(list(ctx.state.refined_sections.keys())[:3]):
            content = ctx.state.refined_sections[title]
            preview = content[:150].replace('\n', ' ').strip()
            logger.info(f"DEBUG: Integrated '{title}': {preview}...")

        # After integrating survey
        integrated_preview = ctx.state.integrated_survey[:300].replace('\n', ' ').strip()
        logger.info(f"DEBUG: Integrated survey preview: {integrated_preview}...")
        logger.info(f"DEBUG: Integrated survey total length: {len(ctx.state.integrated_survey)} characters")

        # Save the state
        save_pipeline_state("integration_refinement", ctx.state)

        # Output Stage 3 results
        results = {
            "stage": "Integration & Refinement",
            "refined_sections_count": len(ctx.state.refined_sections),
            "integrated_survey_length": len(ctx.state.integrated_survey),
            "section_titles": sorted(list(ctx.state.refined_sections.keys())),
            "abstract_preview": ctx.state.refined_sections.get("Abstract", "")[:200] + "...",
            "introduction_preview": ctx.state.refined_sections.get("Introduction", "")[:200] + "...",
            "conclusion_preview": ctx.state.refined_sections.get("Conclusion", "")[:200] + "..."
        }

        # Print the results
        logger.info("=== STAGE 3 RESULTS ===")
        logger.info(f"Refined sections: {results['refined_sections_count']}")
        logger.info(f"Integrated survey length: {results['integrated_survey_length']} characters")
        logger.info("Key section previews:")
        logger.info(f"  Abstract: {results['abstract_preview']}")
        logger.info(f"  Introduction: {results['introduction_preview']}")
        logger.info(f"  Conclusion: {results['conclusion_preview']}")

        # Store results in state
        ctx.state.stage_results = ctx.state.stage_results or {}
        ctx.state.stage_results["stage3"] = results

        return Stage4_EvaluationIteration()
