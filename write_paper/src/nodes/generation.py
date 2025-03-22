#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
from typing import Dict
from datetime import datetime
from pathlib import Path
from ..state import ResearchState
from ..utils.ollama import OllamaClient
import logging
import json

logger = logging.getLogger(__name__)

def save_state_file(phase: str, state_data: dict, output_dir: Path = Path("paper_states")) -> None:
    """Save the current state to a markdown file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create markdown content
    content = [
        f"# Paper Generation State - {phase}",
        f"\n## Metadata",
        f"- Timestamp: {timestamp}",
        f"- Phase: {phase}",
        f"- Topic: {state_data.get('topic', 'N/A')}",
        "\n## Progress",
    ]

    # Add phase-specific content
    if phase == "planning":
        content.extend([
            "### Outline",
            *[f"- {section}" for section in state_data.get('outline', [])]
        ])
    elif phase == "research":
        content.extend([
            "### Related Papers",
            f"- Total papers: {len(state_data.get('related_papers', []))}",
            "### Papers by Section",
            *[f"- {section}: {len(papers)} papers"
              for section, papers in state_data.get('section_papers', {}).items()]
        ])
    elif phase == "analysis":
        content.extend([
            "### Analyzed Sections",
            *[f"- {section}" for section in state_data.get('section_analysis', {}).keys()]
        ])
    elif phase == "synthesis":
        content.extend([
            "### Generated Sections",
            *[f"- {section}" for section in state_data.get('generated_sections', {}).keys()]
        ])
    elif phase == "final":
        content.extend([
            "### Final Paper Sections",
            *[f"- {section}" for section in state_data.get('sections', {}).keys()],
            "\n## Content",
            *[f"\n### {section}\n{content}"
              for section, content in state_data.get('sections', {}).items()]
        ])

    # Save to file
    state_file = output_dir / f"paper_state_{phase}_{timestamp}.md"
    state_file.write_text("\n".join(content))
    logger.info(f"Saved state file: {state_file}")

@dataclass
class PaperGeneration(BaseNode[ResearchState]):
    """Generate final paper in Markdown format"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "End[Dict[str, str]]":
        logger.info("Generating final paper")
        ollama = OllamaClient()

        # Save planning state
        save_state_file("planning", {
            "topic": ctx.state.topic,
            "outline": ctx.state.outline
        })

        # Save research state
        save_state_file("research", {
            "topic": ctx.state.topic,
            "related_papers": [paper.id for paper in ctx.state.related_papers.values()],
            "section_papers": {
                section: [p.id for p in papers]
                for section, papers in getattr(ctx.state, 'section_papers', {}).items()
            }
        })

        # Save analysis state
        save_state_file("analysis", {
            "topic": ctx.state.topic,
            "section_analysis": ctx.state.section_analysis
        })

        # Save synthesis state
        save_state_file("synthesis", {
            "topic": ctx.state.topic,
            "generated_sections": ctx.state.generated_sections
        })

        # Generate abstract last, after all sections are written
        sections_summary = "\n\n".join([
            f"Section: {section}\nContent: {content[:500]}..."  # Use first 500 chars for summary
            for section, content in ctx.state.generated_sections.items()
            if section.lower() != "abstract"
        ])

        # Create abstract generation prompt
        abstract_prompt = f"""Generate an abstract for a survey paper about "{ctx.state.topic}".

The paper contains the following sections:
{sections_summary}

Requirements:
1. Follow standard academic abstract structure
2. Summarize the key themes and findings
3. Highlight the scope and contributions
4. Be concise (250-300 words)
5. Use formal academic language

Generate the abstract:"""

        # Generate abstract
        abstract = await ollama.generate(ctx.state.outline_config.model, abstract_prompt)
        ctx.state.generated_sections["Abstract"] = abstract

        # Format paper in Markdown
        formatted_paper = self._format_markdown(ctx.state.outline, ctx.state.generated_sections)

        # Save final state
        save_state_file("final", {
            "topic": ctx.state.topic,
            "sections": formatted_paper
        })

        logger.info("Paper generation completed")
        return End(formatted_paper)

    def _format_markdown(self, outline: list[str], sections: Dict[str, str]) -> Dict[str, str]:
        """Format the paper in Markdown"""
        formatted_sections = {}

        # Format title
        formatted_sections["Title"] = f"# {sections.get('title', '')}"

        # Format abstract
        formatted_sections["Abstract"] = sections.get("Abstract", "")

        # Add table of contents
        toc = ["## Table of Contents"]
        for section in outline:
            if section.lower() not in ["abstract", "title"]:
                toc.append(f"- [{section}](#{section.lower().replace(' ', '-')})")
        formatted_sections["Table of Contents"] = "\n".join(toc)

        # Format main sections
        for section in outline:
            if section.lower() not in ["abstract", "title"]:
                content = sections.get(section, "")
                # Add section header
                formatted_sections[section] = f"## {section}\n\n{content}"

        # Format references
        if "References" in sections:
            formatted_sections["References"] = "## References\n\n" + sections["References"]

        return formatted_sections
