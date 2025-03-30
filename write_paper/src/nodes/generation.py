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
import re

logger = logging.getLogger(__name__)

@dataclass
class PaperGeneration(BaseNode[ResearchState]):
    """Generate final paper in Markdown format"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "End[Dict[str, str]]":
        logger.info("Generating final paper")
        ollama = OllamaClient()

        # Generate abstract last, after all sections are written
        summary_parts = []
        for section, content in ctx.state.generated_sections.items():
            if section.lower() != "abstract":
                summary = f"Section: {section}\nContent: {content[:500]}..."  # Use first 500 chars
                summary_parts.append(summary)
        sections_summary = "\n\n".join(summary_parts)

        # Create abstract generation prompt
        abstract_prompt = '\n'.join([
            f'Generate an abstract for a survey paper about "{ctx.state.topic}".',
            '',
            'The paper contains the following sections:',
            sections_summary,
            '',
            'Requirements:',
            '1. Follow standard academic abstract structure',
            '2. Summarize the key themes and findings',
            '3. Highlight the scope and contributions',
            '4. Be concise (250-300 words)',
            '5. Use formal academic language',
            '',
            'Generate the abstract:'
        ])

        # Generate abstract
        abstract = await ollama.generate(ctx.state.outline_config.model, abstract_prompt)
        ctx.state.generated_sections["Abstract"] = abstract

        # Format paper in Markdown
        formatted_paper = self._format_markdown(ctx.state.outline, ctx.state.generated_sections)

        # Save complete state to JSON file
        output_dir = Path("result")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        complete_state = {
            "metadata": {
                "timestamp": timestamp,
                "topic": ctx.state.topic,
                "model": ctx.state.outline_config.model,
                "reference_num": ctx.state.outline_config.reference_num
            },
            "state": {
                "topic": ctx.state.topic,
                "outline": ctx.state.outline,
                "related_papers": [paper.id for paper in ctx.state.related_papers.values()],
                "section_papers": {
                    section: [f"[{p.id}] Title: {p.title}" for p in papers]
                    for section, papers in getattr(ctx.state, 'section_papers', {}).items()
                },
                "section_analysis": ctx.state.section_analysis,
                "generated_sections": ctx.state.generated_sections,
                "final_paper": formatted_paper
            }
        }

        json_file = output_dir / f"{ctx.state.topic}.json"
        with open(json_file, "w") as f:
            json.dump(complete_state, f, indent=2)
        logger.info(f"Saved complete state to: {json_file}")

        logger.info("Paper generation completed")
        return End(formatted_paper)

    def _format_latex(self, outline: list[str], sections: Dict[str, str]) -> Dict[str, str]:
        """Format the paper in IEEE LaTeX style"""
        formatted_sections = {}

        # IEEE LaTeX preamble
        preamble = [
            "\\documentclass[conference]{IEEEtran}",
            "\\IEEEoverridecommandlockouts",
            "% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.",
            "\\usepackage{cite}",
            "\\usepackage{amsmath,amssymb,amsfonts}",
            "\\usepackage{algorithmic}",
            "\\usepackage{graphicx}",
            "\\usepackage{textcomp}",
            "\\usepackage{xcolor}",
            "\\def\\BibTeX{{\\rm B\\kern-.05em{\\sc i\\kern-.025em b}\\kern-.08em",
            "    T\\kern-.1667em\\lower.7ex\\hbox{E}\\kern-.125emX}}",
            "\\begin{document}"
        ]
        formatted_sections["Preamble"] = "\n".join(preamble)

        # Format title and author information
        title = sections.get('title', '')
        # Check if title contains a subtitle (indicated by ':' or '-')
        title_parts = title.split(':', 1) if ':' in title else title.split(' - ', 1)
        main_title = title_parts[0].strip()
        subtitle = title_parts[1].strip() if len(title_parts) > 1 else None

        title_text = []
        if subtitle:
            title_text.extend([
                f"\\title{{{main_title}*\\\\",
                "{\\footnotesize \\textsuperscript{*}Note: Sub-titles are not captured in Xplore and",
                "should not be used}\\\\",
                f"{subtitle}",
                "\\thanks{Identify applicable funding agency here. If none, delete this.}",
                "}"
            ])
        else:
            title_text.extend([
                f"\\title{{{main_title}",
                "\\thanks{Identify applicable funding agency here. If none, delete this.}",
                "}"
            ])

        title_text.extend([
            "",
            "\\author{\\IEEEauthorblockN{1\\textsuperscript{st} Given Name Surname}",
            "\\IEEEauthorblockA{\\textit{dept. name of organization (of Aff.)} \\\\",
            "\\textit{name of organization (of Aff.)}\\\\",
            "City, Country \\\\",
            "email address or ORCID}",
            "\\and",
            "\\IEEEauthorblockN{2\\textsuperscript{nd} Given Name Surname}",
            "\\IEEEauthorblockA{\\textit{dept. name of organization (of Aff.)} \\\\",
            "\\textit{name of organization (of Aff.)}\\\\",
            "City, Country \\\\",
            "email address or ORCID}}",
            "",
            "\\maketitle"
        ])
        formatted_sections["Title"] = "\n".join(title_text)

        # Format abstract and keywords
        abstract = sections.get("Abstract", "")
        # Extract keywords from the abstract
        keywords = [
            "large language models",
            "multimodal learning",
            "natural language processing",
            "machine learning",
            "artificial intelligence"
        ]
        formatted_sections["Abstract"] = (
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
            "\\begin{IEEEkeywords}\n"
            f"{', '.join(keywords)}\n"
            "\\end{IEEEkeywords}\n"
        )

        # Format main sections
        for section in outline:
            if section.lower() not in ["abstract", "title"]:
                content = sections.get(section, "")
                # Escape special LaTeX characters
                content = content.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

                # Make sure we don't have subsections in the content
                content = re.sub(r'\\subsection\*?{([^}]+)}', r'\\textbf{\1}:', content)
                content = re.sub(r'\\subsubsection\*?{([^}]+)}', r'\\textit{\1}:', content)

                # Add section header
                if section.lower() == "acknowledgment":
                    formatted_sections[section] = f"\\section*{{{section}}}\n{content}"
                else:
                    formatted_sections[section] = f"\\section{{{section}}}\n{content}"

        # Format references
        if "References" in sections:
            refs = sections["References"]
            # Extract references from the content
            ref_pattern = r'\[(\d+)\]\s+(.*?)(?=\[\d+\]|\Z)'
            matches = re.findall(ref_pattern, refs, re.DOTALL)

            # First pass: Create bibliography entries
            bib_items = []
            for ref_num, ref_text in matches:
                ref_text = ref_text.strip()
                # Clean up reference text (remove unnecessary whitespace, fix formatting)
                ref_text = re.sub(r'\s+', ' ', ref_text)

                # Extract just the paper name (title) without URL, DOI, etc.
                # Look for paper title which is typically the first part of the reference
                paper_title = ref_text
                # Remove publication details (anything after first period or comma might be details)
                match = re.search(r'^([^.,]+)', paper_title)
                if match:
                    paper_title = match.group(1).strip()
                else:
                    # If no clear pattern, just use the first part of the text
                    paper_title = ' '.join(paper_title.split()[:10]) + "..."

                bib_items.append(f"\\bibitem{{b{ref_num}}} {paper_title}")

            # Second pass: Replace citations in all sections
            for section in formatted_sections:
                if section not in ["Preamble", "Title", "End"]:
                    content = formatted_sections[section]

                    # If direct \cite format already exists, preserve it
                    # Otherwise, replace [REF] and [1], [2], etc. with \cite
                    if "\\cite{" not in content:
                        content = re.sub(r'\[REF\]', '\\cite{b1}', content)
                        content = re.sub(r'\[(\d+)\](?!\s+[A-Za-z])', r'\\cite{b\1}', content)

                    formatted_sections[section] = content

            formatted_sections["References"] = (
                "\\section*{References}\n\n"
                "\\begin{thebibliography}{00}\n"
                + "\n".join(bib_items) + "\n"
                "\\end{thebibliography}\n"
            )

        # Add document end
        formatted_sections["End"] = "\\end{document}"

        return formatted_sections

    def _format_markdown(self, outline: list[str], sections: Dict[str, str]) -> Dict[str, str]:
        """Redirect to LaTeX formatting"""
        return self._format_latex(outline, sections)
