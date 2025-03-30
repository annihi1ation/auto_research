#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from typing import Dict, List

from src.nodes.generation import PaperGeneration
from src.state import ResearchState
from src.utils.db import DatabaseManager
from src.utils.ollama import OllamaClient
import logging

logger = logging.getLogger(__name__)

@dataclass
class PaperSearch(BaseNode[ResearchState]):
    """Search for relevant papers for each section"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "ContentAnalysis":
        logger.info("Starting paper search for sections")
        db = DatabaseManager()

        # Store papers by section
        section_papers: Dict[str, List[str]] = {}

        for section in ctx.state.outline:
            # Skip Abstract as it's written last
            if section.lower() == "abstract":
                continue

            # Create section-specific search query
            query = f"{ctx.state.topic} {section}"

            # Get embedding for the query
            query_embedding = ctx.state.embedding_model.encode(query).tolist()

            # Find relevant papers
            papers = db.find_similar_papers(query_embedding, limit=20)  # Get top 20 papers per section

            # Store papers
            section_papers[section] = papers
            logger.info(f"Found {len(papers)} papers for section: {section}")

            # Add to state's related papers
            for paper in papers:
                if paper.id not in ctx.state.related_papers:
                    ctx.state.related_papers[paper.id] = paper

        # Store section-paper mapping in state
        ctx.state.section_papers = section_papers
        return ContentAnalysis()

@dataclass
class ContentAnalysis(BaseNode[ResearchState]):
    """Analyze content from related papers"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "ContentSynthesis":
        logger.info("Analyzing paper content")
        ollama = OllamaClient()

        # For each section, analyze the papers and extract key information
        section_analysis: Dict[str, str] = {}

        for section, papers in ctx.state.section_papers.items():
            # Skip Abstract
            if section.lower() == "abstract":
                continue

            # Create context from papers
            context = []
            for paper in papers[:5]:  # Use top 5 papers for analysis
                context.append(f"Title: {paper.title}")
                if paper.abstract:
                    context.append(f"Abstract: {paper.abstract}")

            # Create analysis prompt
            prompt = f"""Analyze these papers related to the section "{section}" of a survey paper about "{ctx.state.topic}":

{chr(10).join(context)}

Extract and summarize:
1. Key findings and contributions
2. Common themes and patterns
3. Contrasting viewpoints or approaches
4. Technical details and methodologies
5. Limitations and challenges

Format the analysis in a structured way that can be used to write the survey paper section."""

            # Get analysis from Ollama
            analysis = await ollama.generate(ctx.state.outline_config.model, prompt)
            section_analysis[section] = analysis

        # Store analysis in state
        ctx.state.section_analysis = section_analysis
        return ContentSynthesis()

@dataclass
class ContentSynthesis(BaseNode[ResearchState]):
    """Synthesize analyzed content into coherent sections"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "PaperGeneration":
        logger.info("Synthesizing content for sections")
        ollama = OllamaClient()

        # Generate content for each section
        for section in ctx.state.outline:
            if section.lower() == "abstract":
                continue  # Abstract is generated last

            # Get analysis for this section
            analysis = ctx.state.section_analysis.get(section, "")

            # Get related papers for this section
            related_papers = ctx.state.section_papers.get(section, [])

            # Create synthesis prompt
            prompt = f"""Write the "{section}" section of a survey paper about "{ctx.state.topic}".

Reference Papers:
{chr(10).join([f"[{paper.id}] Title: {paper.title}" for paper in related_papers])}

Use this analysis of relevant papers:
{analysis}

Requirements:
1. Write in an academic style suitable for a survey paper
2. Organize content logically and maintain flow
3. Cite references directly using the IEEE citation format (e.g., \cite{{paper id}}) instead of [REF]
4. Do NOT use subsections - organize content as a cohesive article without additional headers
5. Be comprehensive yet concise
6. Highlight key themes and developments
7. Discuss implications and connections

Generate the section content:"""

            # Generate section content
            content = await ollama.generate(ctx.state.outline_config.model, prompt)
            ctx.state.generated_sections[section] = content
            logger.info(f"Generated content for section: {section}")

        return PaperGeneration()
