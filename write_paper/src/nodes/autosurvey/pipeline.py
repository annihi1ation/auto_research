#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
import logging

from src.state import ResearchState
from .stage1 import Stage1_InitialRetrieval

logger = logging.getLogger(__name__)

@dataclass
class AutoSurveyPipeline(BaseNode[ResearchState]):
    """Entry point for the AutoSurvey pipeline"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Stage1_InitialRetrieval:
        logger.info(f"Starting AutoSurvey pipeline for topic: {ctx.state.topic}")
        ctx.state.current_phase = "autosurvey"
        logger.info("DEBUG: ResearchState initialized with topic: " + ctx.state.topic)
        return Stage1_InitialRetrieval()
