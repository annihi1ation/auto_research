#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
import logging

from src.state import ResearchState
from src.utils.ollama import OllamaClient
from src.utils.db import DatabaseManager
from .utils import save_pipeline_state
from .stage4 import Stage4_EvaluationIteration

logger = logging.getLogger(__name__)

@dataclass
class Stage2_SubsectionDrafting(BaseNode[ResearchState]):
    """Stage 2: Subsection Drafting

    This stage retrieves relevant publications for each section/subsection and
    drafts content for each part of the outline.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Stage4_EvaluationIteration:
        logger.info("Stage 2: Subsection Drafting")
        logger.info("DEBUG: Starting subsection drafting process")

        # Initialize clients
        db = DatabaseManager()
        ollama = OllamaClient()

        # Initialize storage for section drafts
        ctx.state.section_drafts = {}
        ctx.state.section_publications = {}

        # Process each section and subsection
        for section_idx, section in enumerate(ctx.state.structured_outline.get("sections", [])):
            section_title = section["title"]
            logger.info(f"Processing section: {section_title}")

            # For each section, retrieve relevant publications
            section_query = f"{ctx.state.topic} {section_title}"
            query_embedding = ctx.state.embedding_model.encode(section_query).tolist()
            section_papers = db.find_similar_papers(query_embedding, limit=20)

            # Store section papers
            ctx.state.section_publications[section_title] = section_papers
            logger.info(f"Retrieved {len(section_papers)} papers for section: {section_title}")

            # Track subsection drafts for this section
            subsection_drafts = {}

            # First process all subsections
            for subsection_idx, subsection in enumerate(section.get("subsections", [])):
                subsection_title = subsection["title"]
                full_subsection_title = f"{section_title} - {subsection_title}"
                logger.info(f"Processing subsection: {full_subsection_title}")

                # For each subsection, retrieve specific papers
                subsection_query = f"{ctx.state.topic} {section_title} {subsection_title}"
                query_embedding = ctx.state.embedding_model.encode(subsection_query).tolist()
                subsection_papers = db.find_similar_papers(query_embedding, limit=15)

                # Store subsection papers
                ctx.state.section_publications[full_subsection_title] = subsection_papers
                logger.info(f"Retrieved {len(subsection_papers)} papers for subsection: {full_subsection_title}")

                # Create context from subsection papers
                subsection_context = []
                for i, paper in enumerate(subsection_papers[:8]):  # Use top 8 papers
                    subsection_context.append(f"[{i+1}] Title: {paper.title}")
                    if paper.abstract:
                        subsection_context.append(f"Abstract: {paper.abstract[:250]}...")  # Truncate long abstracts

                # Add key points if available
                key_points = subsection.get("key_points", [])
                key_points_text = ""
                if key_points:
                    key_points_text = "Key points to address:\n" + "\n".join([f"- {point}" for point in key_points])

                # Draft content for the subsection
                subsection_prompt = f"""Draft content for the "{subsection_title}" subsection within the "{section_title}" section of a research paper on "{ctx.state.topic}".

Relevant publications:
{chr(10).join(subsection_context)}

{key_points_text}

Your task:
1. Write detailed content for this specific subsection, following academic research paper standards
2. Focus on the appropriate content based on the section type:
   - If Introduction: Provide context, research questions, and significance
   - If Literature Review: Summarize existing research and identify gaps
   - If Methodology: Describe survey design, participant selection, and data collection
   - If Results: Present findings with tables/graphs references
   - If Discussion: Interpret results in relation to research questions
   - If Conclusion: Summarize findings and suggest future research
3. Be thorough yet concise with academic writing style
4. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
5. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Write the subsection content:
"""

                subsection_draft = await ollama.generate(ctx.state.outline_config.model, subsection_prompt)
                ctx.state.section_drafts[full_subsection_title] = subsection_draft

                # Store for summarizing when creating the section draft
                subsection_drafts[subsection_title] = subsection_draft

            # Now generate the section draft based on subsection summaries (if any)
            if subsection_drafts:
                # Create summary of subsections
                subsection_summaries = []
                for sub_title, sub_content in subsection_drafts.items():
                    # Extract first 200 chars as a summary
                    summary = f"- {sub_title}: {sub_content[:200]}..."
                    subsection_summaries.append(summary)

                # Create section context from papers
                section_context = []
                for i, paper in enumerate(section_papers[:10]):  # Use top 10 papers
                    section_context.append(f"[{i+1}] Title: {paper.title}")
                    if paper.abstract:
                        section_context.append(f"Abstract: {paper.abstract[:300]}...")  # Truncate long abstracts

                # Draft content for the main section based on subsections
                section_prompt = f"""Draft content for the "{section_title}" section of a research paper on "{ctx.state.topic}".

This section will contain the following subsections:
{chr(10).join(subsection_summaries)}

Relevant publications:
{chr(10).join(section_context)}

Your task:
1. Write a comprehensive introduction for this section that ties together all subsections
2. Focus on the appropriate content based on the section type:
   - If Introduction: Overview of research topic, questions, objectives, and importance
   - If Literature Review: Summary of existing research and gaps
   - If Methodology: Description of survey design, sampling, and data collection
   - If Results: Presentation of survey findings with data visualization
   - If Discussion: Interpretation of results in relation to literature
   - If Conclusion: Summary of findings, implications, and future research
3. Use an academic writing style
4. Be informative, clear, and concise
5. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
6. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Write the section content:
"""
            else:
                # No subsections - create section context from papers
                section_context = []
                for i, paper in enumerate(section_papers[:10]):  # Use top 10 papers
                    section_context.append(f"[{i+1}] Title: {paper.title}")
                    if paper.abstract:
                        section_context.append(f"Abstract: {paper.abstract[:300]}...")  # Truncate long abstracts

                # Draft content for the main section without subsections
                section_prompt = f"""Draft content for the "{section_title}" section of a research paper on "{ctx.state.topic}".

Relevant publications:
{chr(10).join(section_context)}

Your task:
1. Write a comprehensive {section_title} section following standard research paper format
2. Focus on the appropriate content based on the section type:
   - If Introduction: Overview of research topic, questions, objectives, and importance
   - If Literature Review: Summary of existing research and gaps
   - If Methodology: Description of survey design, sampling, and data collection
   - If Results: Presentation of survey findings with data visualization
   - If Discussion: Interpretation of results in relation to literature
   - If Conclusion: Summary of findings, implications, and future research
3. Use an academic writing style
4. Be informative, clear, and concise
5. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
6. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Write the section content:
"""

            section_draft = await ollama.generate(ctx.state.outline_config.model, section_prompt)
            ctx.state.section_drafts[section_title] = section_draft

        # Handle special sections
        # Introduction
        if "Introduction" in ctx.state.section_drafts:
            logger.info("Introduction already drafted as a main section")
        else:
            logger.info("Drafting Introduction section")
            intro_prompt = f"""Write the Introduction section for a research paper on "{ctx.state.topic}".

Your task:
1. Begin with a clear articulation of the research topic and its importance
2. Provide background context and motivation for the study
3. State the specific research questions and objectives
4. Explain the significance of your study (theoretical and practical)
5. Outline the organization of the paper
6. Use an academic writing style that is clear and engaging
7. Be comprehensive yet concise
8. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
9. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Write the Introduction section:
"""
            intro_draft = await ollama.generate(ctx.state.outline_config.model, intro_prompt)
            ctx.state.section_drafts["Introduction"] = intro_draft

        # Conclusion
        logger.info("Drafting Conclusion section")
        sections_summary = []
        for section_title, content in ctx.state.section_drafts.items():
            if "-" not in section_title:  # Only include main sections
                summary = f"Section '{section_title}' discusses: {content[:200]}..."  # Short summary
                sections_summary.append(summary)

        conclusion_prompt = f"""Write the Conclusion section for a research paper on "{ctx.state.topic}".

Here's a brief summary of the paper's main sections:
{chr(10).join(sections_summary)}

Your task:
1. Summarize the key findings in relation to your research questions
2. Highlight the significance and implications of these findings
3. Discuss theoretical contributions and practical applications
4. Address limitations of the current research
5. Suggest specific directions for future research
6. End with a compelling statement about the overall contribution
7. Use an academic writing style
8. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
9. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Write the Conclusion section:
"""
        conclusion_draft = await ollama.generate(ctx.state.outline_config.model, conclusion_prompt)
        ctx.state.section_drafts["Conclusion"] = conclusion_draft

        # Save the state
        save_pipeline_state("subsection_drafting", ctx.state)

        # After processing each section
        logger.info(f"DEBUG: Total sections after processing: {len(ctx.state.section_drafts)}")
        logger.info(f"DEBUG: Section titles: {', '.join(sorted([t for t in ctx.state.section_drafts.keys() if '-' not in t]))}")
        logger.info(f"DEBUG: Subsection count: {len([t for t in ctx.state.section_drafts.keys() if '-' in t])}")

        # Save sample content for verification
        for title, content in list(ctx.state.section_drafts.items())[:2]:
            preview = content[:150].replace('\n', ' ').strip()
            logger.info(f"DEBUG: Sample content for '{title}': {preview}...")

        # Output Stage 2 results
        section_titles = sorted([t for t in ctx.state.section_drafts.keys() if '-' not in t])
        subsection_titles = sorted([t for t in ctx.state.section_drafts.keys() if '-' in t])

        results = {
            "stage": "Subsection Drafting",
            "main_sections_drafted": len(section_titles),
            "subsections_drafted": len(subsection_titles),
            "total_drafts": len(ctx.state.section_drafts),
            "main_sections": section_titles,
            "subsections": subsection_titles,
            "section_samples": {
                title: ctx.state.section_drafts[title][:200] + "..."
                for title in list(section_titles)[:3]  # First 3 main sections
            }
        }

        # Print the results
        logger.info("=== STAGE 2 RESULTS ===")
        logger.info(f"Main sections drafted: {results['main_sections_drafted']}")
        logger.info(f"Subsections drafted: {results['subsections_drafted']}")
        logger.info(f"Total drafts: {results['total_drafts']}")
        logger.info("Main section titles:")
        for title in section_titles[:5]:  # Show first 5 main sections
            section_len = len(ctx.state.section_drafts[title])
            logger.info(f"  - {title} ({section_len} characters)")
        logger.info("Sample content:")
        for title, sample in results['section_samples'].items():
            logger.info(f"  {title}: {sample}")

        # Store results in state
        ctx.state.stage_results = ctx.state.stage_results or {}
        ctx.state.stage_results["stage2"] = results

        # Initialize refined_sections with section_drafts to skip the refinement stage
        ctx.state.refined_sections = ctx.state.section_drafts.copy()

        # Build the integrated survey directly from section_drafts
        paper_sections = []

        # Add main sections in order, with their subsections
        for section in ctx.state.structured_outline.get("sections", []):
            section_title = section["title"]
            if section_title in ctx.state.section_drafts:
                paper_sections.append(f"# {section_title}\n\n{ctx.state.section_drafts[section_title]}")

            # Add subsections
            for subsection in section.get("subsections", []):
                subsection_title = subsection["title"]
                full_subsection_title = f"{section_title} - {subsection_title}"
                if full_subsection_title in ctx.state.section_drafts:
                    paper_sections.append(f"## {subsection_title}\n\n{ctx.state.section_drafts[full_subsection_title]}")

        # Add conclusion
        if "Conclusion" in ctx.state.section_drafts:
            paper_sections.append(f"# Conclusion\n\n{ctx.state.section_drafts['Conclusion']}")

        # Create integrated survey
        ctx.state.integrated_survey = "\n\n".join(paper_sections)

        # Add a note about skipping refinement
        logger.info("Skipping Stage 3: Integration & Refinement as requested")

        # Go directly to Stage 4
        return Stage4_EvaluationIteration()
