#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
from typing import List
from ..state import ResearchState
from ..providers.llm.outline import OutlineGenerator
from .research import PaperSearch
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlanningPhase(BaseNode[ResearchState]):
    """Initial planning phase for research paper generation"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "OutlineGeneration":
        ctx.state.current_phase = "planning"
        logger.info(f"Starting planning phase for topic: {ctx.state.topic}")
        return OutlineGeneration()

@dataclass
class OutlineGeneration(BaseNode[ResearchState]):
    """Generate paper outline using Ollama and reference papers"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "SectionPlanning":
        # Get provider and config from state
        provider_type = ctx.state.stage_results.get("provider", "ollama")
        provider_config = ctx.state.stage_results.get("provider_config", {"model": ctx.state.outline_config.model})
        logger.info(f"Generating outline using {provider_type} provider")

        # Initialize outline generator with config
        outline_generator = OutlineGenerator(
            provider_type=provider_type,
            provider_config=provider_config,
            reference_num=ctx.state.outline_config.reference_num,
            num_sections=ctx.state.outline_config.num_sections
        )

        # Generate outline
        try:
            outline = await outline_generator.generate_outline(ctx.state.topic)
            ctx.state.outline = outline
            logger.info(f"Generated outline with {len(outline)} sections")
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            # Fallback to basic outline
            ctx.state.outline = [
                "Abstract",
                "Introduction",
                "Background",
                "Current State of the Art",
                "Future Directions",
                "Conclusion"
            ]

        return SectionPlanning()

@dataclass
class SectionPlanning(BaseNode[ResearchState]):
    """Plan content generation for each section"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "PaperSearch":
        logger.info("Planning section content generation")

        # Initialize section content placeholders
        ctx.state.generated_sections = {
            section: "" for section in ctx.state.outline
        }

        # Move to paper search phase
        return PaperSearch()
