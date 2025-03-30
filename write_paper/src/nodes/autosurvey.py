#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

from src.state import ResearchState
from src.utils.ollama import OllamaClient
from src.utils.db import DatabaseManager

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

@dataclass
class AutoSurveyPipeline(BaseNode[ResearchState]):
    """Entry point for the AutoSurvey pipeline"""
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "Stage1_InitialRetrieval":
        logger.info(f"Starting AutoSurvey pipeline for topic: {ctx.state.topic}")
        ctx.state.current_phase = "autosurvey"
        logger.info("DEBUG: ResearchState initialized with topic: " + ctx.state.topic)
        return Stage1_InitialRetrieval()

@dataclass
class Stage1_InitialRetrieval(BaseNode[ResearchState]):
    """Stage 1: Initial Retrieval & Outline Generation

    This stage retrieves initial publications from a database and generates
    a structured outline for the survey paper.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "Stage2_SubsectionDrafting":
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

@dataclass
class Stage2_SubsectionDrafting(BaseNode[ResearchState]):
    """Stage 2: Subsection Drafting

    This stage retrieves relevant publications for each section/subsection and
    drafts content for each part of the outline.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "Stage4_EvaluationIteration":
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

@dataclass
class Stage3_IntegrationRefinement(BaseNode[ResearchState]):
    """Stage 3: Integration & Refinement

    This stage refines each section/subsection and integrates them into a
    cohesive survey paper.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "Stage4_EvaluationIteration":
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
8. Maintain citation style consistency (use [1], [2], etc.)
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

@dataclass
class Stage4_EvaluationIteration(BaseNode[ResearchState]):
    """Stage 4: Rigorous Evaluation & Iteration

    This stage evaluates the survey paper on multiple criteria and iterates to
    produce the best version.
    """
    # Add citation tracking dictionary
    _cited_papers = {}  # Dictionary to track papers cited in the content

    async def run(self, ctx: GraphRunContext[ResearchState]) -> "End[Dict[str, str]]":
        # Initialize the citation tracking dictionary
        self._cited_papers = {}
        # Initialize a global collection of all related papers
        self._global_related_papers = []

        # Store state for later use in helper methods
        self._state = ctx.state

        logger.info("Stage 4: Rigorous Evaluation & Iteration")
        logger.info("DEBUG: Starting evaluation & iteration process")

        # Initialize client
        ollama = OllamaClient()

        # Initialize storage for evaluation results
        ctx.state.evaluation_results = {}
        ctx.state.iteration_surveys = []

        # Add the current survey as the first iteration
        initial_survey = {
            "content": ctx.state.integrated_survey,
            "sections": ctx.state.refined_sections.copy()
        }
        ctx.state.iteration_surveys.append(initial_survey)

        # Collect all related papers from section_publications into global collection
        for section_title, papers in ctx.state.section_publications.items():
            for paper in papers:
                if paper.id not in [p.id for p in self._global_related_papers]:
                    self._global_related_papers.append(paper)

        logger.info(f"Collected {len(self._global_related_papers)} unique papers for potential citations")

        # Define evaluation criteria
        evaluation_criteria = ["coverage", "structure", "relevance", "faithfulness"]

        # Evaluate the initial survey
        logger.info("Evaluating initial survey")
        initial_evaluation = await self._evaluate_survey(ctx.state, ollama, 0)
        ctx.state.evaluation_results[0] = initial_evaluation

        # After evaluating initial survey
        logger.info(f"DEBUG: Initial evaluation scores: {ctx.state.evaluation_results[0]}")

        # Perform iterations to improve the survey
        num_iterations = 2  # Set the number of iterations
        for iteration in range(1, num_iterations + 1):
            logger.info(f"Starting iteration {iteration}")

            # Generate improvement suggestions based on previous evaluation
            prev_evaluation = ctx.state.evaluation_results[iteration - 1]
            lowest_score_criterion = min(prev_evaluation, key=prev_evaluation.get)
            logger.info(f"DEBUG: Iteration {iteration} focusing on improving: {lowest_score_criterion}")
            logger.info(f"DEBUG: Previous scores: {prev_evaluation}")

            improvement_prompt = f"""Analyze this survey paper on "{ctx.state.topic}" and suggest specific improvements focused on {lowest_score_criterion}.

Current evaluation:
"""
            for criterion, score in prev_evaluation.items():
                improvement_prompt += f"- {criterion.capitalize()}: {score:.2f}/10\n"

            improvement_prompt += f"""
The paper needs particular improvement in "{lowest_score_criterion}".

Paper content (partial):
{ctx.state.iteration_surveys[iteration-1]["content"][:5000]}...

Suggest specific, actionable improvements to enhance the {lowest_score_criterion} of this survey paper:
"""

            improvement_suggestions = await ollama.generate(ctx.state.outline_config.model, improvement_prompt)
            logger.info(f"Generated improvement suggestions for iteration {iteration}")

            # Create an improved version of the survey
            improved_survey = await self._improve_survey(ctx.state, ollama, improvement_suggestions, lowest_score_criterion, iteration-1)

            # Add to iterations
            ctx.state.iteration_surveys.append(improved_survey)

            # Evaluate the new survey
            new_evaluation = await self._evaluate_survey(ctx.state, ollama, iteration)
            ctx.state.evaluation_results[iteration] = new_evaluation

            logger.info(f"Completed iteration {iteration}")

        # Select the best survey based on average evaluation scores
        avg_scores = {}
        for idx, evaluation in ctx.state.evaluation_results.items():
            avg_scores[idx] = sum(evaluation.values()) / len(evaluation)

        best_idx = max(avg_scores, key=avg_scores.get)
        ctx.state.best_survey_idx = best_idx

        logger.info(f"Selected iteration {best_idx} as the best survey with avg score {avg_scores[best_idx]:.2f}")

        # After selecting best survey
        logger.info(f"DEBUG: Selected best survey (iteration {best_idx}) with sections: {', '.join(sorted(ctx.state.iteration_surveys[best_idx]['sections'].keys()))}")

        # Save the final state
        save_pipeline_state("evaluation_iteration", ctx.state)

        # Create a results directory for this run
        output_dir = Path("paper_states/autosurvey")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        paper_dir = output_dir / f"paper_{timestamp}"
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Format the best survey for output and process citations
        logger.info("Processing citations and formatting paper")
        formatted_sections = self._format_paper(ctx.state.iteration_surveys[best_idx]["sections"], paper_dir)

        # Generate the final abstract as the last step
        logger.info("Generating final abstract as the last step")

        # Create a concise summary of each main section from the best survey
        section_summaries = []
        for section_title, content in formatted_sections.items():
            if section_title not in ["Abstract", "References", "Preamble", "Title", "End"] and "-" not in section_title:
                # Get a short summary of each main section (first 100 chars or so)
                summary = f"{section_title}: {content[:100].replace('\n', ' ')}..."
                section_summaries.append(summary)

        # Final abstract prompt emphasizing brevity and simplicity
        final_abstract_prompt = f"""Write a clear, concise abstract for a research paper on "{ctx.state.topic}".

The research paper covers these main aspects:
{chr(10).join(section_summaries)}

Your task:
1. Keep the abstract SHORT (exactly 250 words maximum)
2. Follow the standard research paper abstract structure:
   - Background: Brief context of the research problem
   - Purpose: Clear statement of research objectives
   - Methods: Brief description of the study design and data collection
   - Results: Summary of key findings
   - Conclusion: Main implications and significance
3. Use precise, academic language while remaining accessible
4. Focus on the direct results of your research
5. Include 2-3 key findings only
6. Emphasize the significance and implications
7. DO NOT include citations in the abstract

Write a short, structured abstract of 250 words maximum:
"""

        final_abstract = await ollama.generate(ctx.state.outline_config.model, final_abstract_prompt)

        # Replace the abstract in the best survey
        formatted_sections["Abstract"] = (
            f"\\begin{{abstract}}\n{final_abstract}\n\\end{{abstract}}\n\n"
            "\\begin{IEEEkeywords}\n"
            "survey, literature review, comprehensive analysis, state-of-the-art\n"
            "\\end{IEEEkeywords}\n"
        )

        # Output the final results
        results = {
            "stage": "Evaluation & Iteration",
            "iterations_performed": len(ctx.state.evaluation_results),
            "best_iteration": best_idx,
            "evaluation_scores": ctx.state.evaluation_results,
            "reference_papers_count": len(self._cited_papers) if hasattr(self, '_cited_papers') else 0,
            "final_sections": list(formatted_sections.keys()),
            "paper_preview": {
                section: content[:150] + "..." for section, content in formatted_sections.items()
                if section not in ["Preamble", "Title", "End"]
            }
        }

        # Print the results
        logger.info("=== STAGE 4 RESULTS ===")
        logger.info(f"Iterations performed: {results['iterations_performed']}")
        logger.info(f"Best iteration: {results['best_iteration']}")
        logger.info("Evaluation scores by iteration:")
        for idx, scores in results['evaluation_scores'].items():
            avg = sum(scores.values()) / len(scores)
            logger.info(f"  Iteration {idx}: {scores} (avg: {avg:.2f})")
        logger.info(f"References count: {results['reference_papers_count']}")
        logger.info("Final paper preview:")
        for section, preview in results['paper_preview'].items():
            logger.info(f"  {section}: {preview}")

        # Store results in state
        ctx.state.stage_results = ctx.state.stage_results or {}
        ctx.state.stage_results["stage4"] = results

        # Combine all stage results for summary
        all_results = {
            "topic": ctx.state.topic,
            "pipeline": "AutoSurvey",
            "stages": {
                "stage1": ctx.state.stage_results.get("stage1", {}),
                "stage2": ctx.state.stage_results.get("stage2", {}),
                "stage3": ctx.state.stage_results.get("stage3", {}),
                "stage4": ctx.state.stage_results.get("stage4", {})
            }
        }

        # Save overall results
        output_dir = Path("paper_states/autosurvey")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(output_dir / f"complete_results_{timestamp}.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Store topic for file naming
        topic = ctx.state.topic

        # Write LaTeX file
        latex_content = []
        for section_name in ["Preamble", "Title", "Abstract"]:
            if section_name in formatted_sections:
                latex_content.append(formatted_sections[section_name])

        # Add remaining sections in proper order with their subsections immediately following
        section_order = [
            "Introduction",
            # Add all other main sections
            *[s for s in sorted(formatted_sections.keys())
              if s not in ["Preamble", "Title", "Abstract", "Introduction", "Conclusion", "References", "End"]
              and "-" not in s],
            "Conclusion",
            "References",
            "End"
        ]

        for section_name in section_order:
            if section_name in formatted_sections:
                latex_content.append(formatted_sections[section_name])

                # Add corresponding subsections immediately after each main section
                if "-" not in section_name and section_name not in ["Conclusion", "References", "End"]:
                    subsections = sorted([s for s in formatted_sections.keys() if s.startswith(f"{section_name} -")])
                    for subsec in subsections:
                        latex_content.append(formatted_sections[subsec])
                    if subsections:
                        logger.info(f"Added {len(subsections)} subsections for {section_name}")
                    else:
                        logger.info(f"No subsections found for {section_name}")

        # Write the LaTeX file
        latex_file_path = paper_dir / f"{topic.replace(' ', '_')}_survey.tex"
        with open(latex_file_path, "w") as f:
            f.write("\n\n".join(latex_content))

        # Also write a plain text version for easier reading
        text_content = []

        # Title and Abstract
        text_content.append(f"# A COMPREHENSIVE SURVEY OF {topic.upper()}")
        text_content.append("\n## Abstract")
        if "Abstract" in formatted_sections:
            # Extract abstract text without LaTeX commands
            abstract_text = formatted_sections["Abstract"]
            abstract_text = abstract_text.replace("\\begin{abstract}", "").replace("\\end{abstract}", "")
            abstract_text = abstract_text.replace("\\begin{IEEEkeywords}", "").replace("\\end{IEEEkeywords}", "")
            text_content.append(abstract_text)

        # Main sections in order with their subsections
        for section_name in section_order:
            if section_name in ["Preamble", "Title", "End"]:
                continue

            if section_name in formatted_sections:
                # Extract section text
                section_text = formatted_sections[section_name]

                # Clean up LaTeX commands for plain text
                for cmd in ["\\section{", "\\subsection{", "\\section*{"]:
                    if cmd in section_text:
                        # Extract section title from LaTeX command
                        start_idx = section_text.find(cmd) + len(cmd)
                        end_idx = section_text.find("}", start_idx)
                        if end_idx > start_idx:
                            section_title = section_text[start_idx:end_idx]
                            text_content.append(f"\n## {section_title}")
                            # Keep only the content after the command
                            section_text = section_text[end_idx+1:].strip()

                # Special handling for references section
                if section_name == "References":
                    section_text = section_text.replace("\\begin{thebibliography}{00}", "")
                    section_text = section_text.replace("\\end{thebibliography}", "")
                    section_text = section_text.replace("\\bibitem{", "[").replace("}", "]")

                text_content.append(section_text)

                # Add corresponding subsections after each main section with proper formatting
                if "-" not in section_name and section_name not in ["Conclusion", "References"]:
                    for subsec in sorted([s for s in formatted_sections.keys() if s.startswith(f"{section_name} -")]):
                        subsec_text = formatted_sections[subsec]
                        # Extract subsection title
                        start_idx = subsec_text.find("\\subsection{") + len("\\subsection{")
                        end_idx = subsec_text.find("}", start_idx)
                        if end_idx > start_idx:
                            subsec_title = subsec_text[start_idx:end_idx]
                            text_content.append(f"\n### {subsec_title}")
                            # Keep only the content after the command
                            subsec_text = subsec_text[end_idx+1:].strip()
                        text_content.append(subsec_text)

        # Write the plain text file
        text_file_path = paper_dir / f"{topic.replace(' ', '_')}_survey.txt"
        with open(text_file_path, "w") as f:
            f.write("\n\n".join(text_content))

        logger.info(f"Final paper written to:")
        logger.info(f"  - LaTeX: {latex_file_path}")
        logger.info(f"  - Text: {text_file_path}")

        logger.info("=== AUTOSURVEY PIPELINE COMPLETE ===")
        logger.info(f"Complete results saved to: {output_dir}/complete_results_{timestamp}.json")

        logger.info("AutoSurvey pipeline completed")
        return End(formatted_sections)

    async def _evaluate_survey(self, state: ResearchState, ollama: OllamaClient, iteration_idx: int) -> Dict[str, float]:
        """Evaluate the survey on multiple criteria and return scores"""
        logger.info(f"DEBUG: Evaluating survey iteration {iteration_idx}")

        survey = state.iteration_surveys[iteration_idx]

        evaluation_prompt = f"""Evaluate this survey paper on "{state.topic}" according to the following criteria:

1. Coverage: How comprehensively does the paper cover the topic?
2. Structure: How well-organized and logical is the paper's structure?
3. Relevance: How relevant and up-to-date is the content?
4. Faithfulness: How accurately does the paper represent the field without hallucinations?

Paper content (partial):
{survey["content"][:5000]}...

For each criterion, provide:
1. A score from 0-10 (where 10 is best)
2. A brief justification for the score

Format your response EXACTLY as JSON:
{{
  "coverage": score,
  "structure": score,
  "relevance": score,
  "faithfulness": score
}}

Only output the JSON, nothing else.
"""

        eval_result_str = await ollama.generate(state.outline_config.model, evaluation_prompt)

        # Parse evaluation results
        try:
            eval_result = json.loads(eval_result_str)
            # Ensure all required keys are present
            for criterion in ["coverage", "structure", "relevance", "faithfulness"]:
                if criterion not in eval_result:
                    eval_result[criterion] = 5.0  # Default score
            return eval_result
        except json.JSONDecodeError:
            logger.error(f"DEBUG: Failed to parse evaluation JSON: {eval_result_str[:200]}...")
            # Return default scores
            return {
                "coverage": 5.0,
                "structure": 5.0,
                "relevance": 5.0,
                "faithfulness": 5.0
            }

    async def _improve_survey(self, state: ResearchState, ollama: OllamaClient,
                             suggestions: str, criterion: str, prev_idx: int) -> Dict[str, Any]:
        """Improve the survey based on evaluation feedback"""
        logger.info(f"DEBUG: Improving survey focused on criterion: {criterion}")
        logger.info(f"DEBUG: Improvement suggestions preview: {suggestions[:200]}...")

        prev_survey = state.iteration_surveys[prev_idx]

        # Initialize improved sections
        improved_sections = prev_survey["sections"].copy()

        # Identify which sections to improve based on the criterion
        if criterion == "coverage":
            # Improve main content sections
            for section_title in improved_sections:
                if section_title.lower() not in ["abstract", "introduction", "conclusion", "references"]:
                    section_content = improved_sections[section_title]
                    improvement_prompt = f"""Improve the coverage of this "{section_title}" section for a research paper on "{state.topic}".

Current content:
{section_content}

Improvement suggestions:
{suggestions}

Your task:
1. Expand the coverage based on the section type:
   - If Introduction: Ensure research questions and objectives are clearly articulated
   - If Literature Review: Cover all relevant existing research and identify gaps
   - If Methodology: Provide thorough details on survey design and data collection
   - If Results: Present comprehensive findings with appropriate data
   - If Discussion: Thoroughly interpret results in relation to literature
   - If Conclusion: Address all key findings and implications
2. Address any gaps in the content
3. Ensure all relevant aspects are covered
4. Maintain the academic style and flow
5. Keep the improved version comprehensive yet concise
6. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
7. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Provide the improved section:
"""
                    improved_content = await ollama.generate(state.outline_config.model, improvement_prompt)
                    improved_sections[section_title] = improved_content

        elif criterion == "structure":
            # Improve the overall structure by enhancing transitions and organization
            for section_title in improved_sections:
                section_content = improved_sections[section_title]
                improvement_prompt = f"""Improve the structure of this "{section_title}" section for a research paper on "{state.topic}".

Current content:
{section_content}

Improvement suggestions:
{suggestions}

Your task:
1. Enhance the logical flow and organization based on the section type:
   - If Introduction: Progress from broad topic to specific research questions
   - If Literature Review: Organize by themes, chronology, or methodology
   - If Methodology: Present in logical sequence (design → participants → procedures → analysis)
   - If Results: Structure from descriptive to inferential, by research questions
   - If Discussion: Connect results to literature, then implications
   - If Conclusion: Flow from summary to implications to future work
2. Add clear transitions between paragraphs and ideas
3. Improve topic sentences and paragraph structure
4. Ensure proper signposting throughout the section
5. Maintain the academic style and content
6. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
7. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Provide the improved section:
"""
                improved_content = await ollama.generate(state.outline_config.model, improvement_prompt)
                improved_sections[section_title] = improved_content

        elif criterion == "relevance":
            # Focus on updating content with the most relevant information
            for section_title in improved_sections:
                if section_title.lower() not in ["abstract", "references"]:
                    section_content = improved_sections[section_title]
                    improvement_prompt = f"""Improve the relevance of this "{section_title}" section for a research paper on "{state.topic}".

Current content:
{section_content}

Improvement suggestions:
{suggestions}

Your task:
1. Ensure all content is directly relevant to the section's purpose:
   - If Introduction: Focus on research questions and significance
   - If Literature Review: Include only studies pertinent to your research questions
   - If Methodology: Provide only details necessary to understand your approach
   - If Results: Present findings directly related to research questions
   - If Discussion: Interpret only the results presented in your study
   - If Conclusion: Address only findings from your research
2. Remove tangential or less important information
3. Highlight the most significant aspects
4. Focus on content with direct implications
5. Maintain the academic style and comprehensiveness
6. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
7. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Provide the improved section:
"""
                    improved_content = await ollama.generate(state.outline_config.model, improvement_prompt)
                    improved_sections[section_title] = improved_content

        elif criterion == "faithfulness":
            # Focus on accuracy and avoiding hallucinations
            for section_title in improved_sections:
                if section_title.lower() not in ["abstract", "references"]:
                    section_content = improved_sections[section_title]
                    improvement_prompt = f"""Improve the faithfulness and accuracy of this "{section_title}" section for a research paper on "{state.topic}".

Current content:
{section_content}

Improvement suggestions:
{suggestions}

Your task:
1. Ensure all statements are well-supported by evidence, especially:
   - If Introduction: Claims about significance of the problem
   - If Literature Review: Representations of prior research findings
   - If Methodology: Description of procedures and instruments
   - If Results: Presentation of data and statistical claims
   - If Discussion: Interpretations of findings
   - If Conclusion: Claims about implications and contributions
2. Use measured language that avoids overstatements
3. Be precise with descriptions of methods and findings
4. Qualify claims appropriately
5. Maintain the academic style and content focus
6. IMPORTANT: When citing papers, use the IEEE citation format with \\cite{{key}} directly in the text.
   For example: "This approach was proposed by Smith et al. \\cite{{smith2019}}."
   Use descriptive citation keys combining author name and year (e.g., author2023).
   CITE RELEVANT PAPERS FREQUENTLY throughout the section.
7. DO NOT include a "Relevant Publications" section or list of references in your output.
   All references should ONLY appear as in-text citations using the \\cite{{}} format.

Provide the improved section:
"""
                    improved_content = await ollama.generate(state.outline_config.model, improvement_prompt)
                    improved_sections[section_title] = improved_content

        # Now regenerate the abstract to reflect the improvements
        abstract_prompt = f"""Generate an updated abstract for the improved research paper on "{state.topic}".

The paper has been improved particularly in terms of {criterion}.

Below is a summary of the main sections:
"""
        for section_title, content in improved_sections.items():
            if section_title.lower() not in ["abstract", "references"]:
                abstract_prompt += f"\n{section_title}: {content[:150]}..."

        abstract_prompt += f"""

Your task:
1. Create a compelling abstract (250-300 words)
2. Highlight the paper's improved {criterion}
3. Clearly state the purpose and scope of the research
4. Summarize key findings and insights
5. Note the significance and implications
6. Use formal academic language
7. Be concise yet comprehensive
8. DO NOT include citations in the abstract

Generate the abstract:
"""

        improved_abstract = await ollama.generate(state.outline_config.model, abstract_prompt)
        improved_sections["Abstract"] = improved_abstract

        # Build integrated survey content
        paper_sections = []
        paper_sections.append(f"# Abstract\n\n{improved_sections['Abstract']}")

        # Add main sections in order, with their subsections
        for section in state.structured_outline.get("sections", []):
            section_title = section["title"]
            if section_title in improved_sections:
                paper_sections.append(f"# {section_title}\n\n{improved_sections[section_title]}")

            # Add subsections
            for subsection in section.get("subsections", []):
                subsection_title = subsection["title"]
                full_subsection_title = f"{section_title} - {subsection_title}"
                if full_subsection_title in improved_sections:
                    paper_sections.append(f"## {subsection_title}\n\n{improved_sections[full_subsection_title]}")

        # Add conclusion
        if "Conclusion" in improved_sections:
            paper_sections.append(f"# Conclusion\n\n{improved_sections['Conclusion']}")

        integrated_content = "\n\n".join(paper_sections)

        # After improving sections
        logger.info(f"DEBUG: Improved {len(improved_sections)} sections")

        # After generating improved content
        logger.info(f"DEBUG: Improved integrated content length: {len(integrated_content)} characters")

        return {
            "content": integrated_content,
            "sections": improved_sections
        }

    def _format_paper(self, sections: Dict[str, str], paper_dir: Path) -> Dict[str, str]:
        """Format the paper sections for output in IEEE format"""
        logger.info("DEBUG: Formatting paper in IEEE format")

        # Extract and track citations from all sections
        self._extract_citations(sections)
        logger.info(f"Tracked {len(self._cited_papers)} unique citations")

        logger.info(f"DEBUG: Formatting {len(sections)} sections: {', '.join(sorted(sections.keys()))}")

        # Generate the BibTeX file
        bibtex_filename = self._generate_bibtex_file(paper_dir)

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
            "%",
            "% To compile this document with references:",
            "% 1. pdflatex filename.tex",
            "% 2. bibtex filename",
            "% 3. pdflatex filename.tex",
            "% 4. pdflatex filename.tex",
            "%",
            "\\begin{document}"
        ]
        formatted_sections["Preamble"] = "\n".join(preamble)

        # Title - Use the actual topic name from state
        actual_topic = self._state.topic
        title_text = [
            f"\\title{{Research on {actual_topic}}}",
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
        ]
        formatted_sections["Title"] = "\n".join(title_text)

        # Abstract
        abstract = sections.get("Abstract", "")
        formatted_sections["Abstract"] = (
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
            "\\begin{IEEEkeywords}\n"
            "survey, literature review, comprehensive analysis, state-of-the-art\n"
            "\\end{IEEEkeywords}\n"
        )

        # Process main sections and replace citation placeholders with proper BibTeX citations
        for section_title, content in sections.items():
            if section_title not in ["Abstract", "title"]:
                # Replace citation placeholders [ID] with \cite{key} format
                processed_content = self._replace_citations_with_bibtex_format(content)

                # Format the section properly
                if "-" in section_title:  # This is a subsection
                    main_section, subsection = section_title.split(" - ", 1)
                    latex_content = f"\\subsection{{{subsection}}}\n{processed_content}"
                    formatted_sections[section_title] = latex_content
                else:
                    if section_title.lower() == "acknowledgment":
                        formatted_sections[section_title] = f"\\section*{{{section_title}}}\n{processed_content}"
                    else:
                        formatted_sections[section_title] = f"\\section{{{section_title}}}\n{processed_content}"

        # Use BibTeX for bibliography without file extension (LaTeX will add it)
        bibtex_name_without_extension = bibtex_filename.replace(".bib", "")
        formatted_sections["References"] = "\\bibliographystyle{IEEEtran}\n\\bibliography{" + bibtex_name_without_extension + "}"

        # Document end
        formatted_sections["End"] = "\\end{document}"

        # After processing references
        logger.info(f"DEBUG: Generated BibTeX file with {len(self._cited_papers)} citations")

        return formatted_sections

    def _extract_citations(self, sections: Dict[str, str]) -> None:
        """Extract citation IDs from all sections and track cited papers"""
        import re

        # Regular expressions to match both citation patterns: [Smith2019] and \cite{smith2019}
        old_citation_pattern = r'\[([^\]]+)\]'
        cite_pattern = r'\\cite\{([^}]+)\}'

        # Use the global collection of related papers
        reference_papers = self._global_related_papers

        logger.info(f"Processing citations from {len(reference_papers)} potential reference papers")

        for section_title, content in sections.items():
            if section_title not in ["Abstract", "title"]:
                # Find all \cite{} citations in the content
                cite_citations = re.findall(cite_pattern, content)

                # Also find traditional [AuthorYear] citations for backward compatibility
                old_citations = re.findall(old_citation_pattern, content)

                # Process all citations, prioritizing \cite{} format
                all_citations = cite_citations + old_citations

                logger.info(f"Found {len(all_citations)} citations in section: {section_title} ({len(cite_citations)} IEEE style, {len(old_citations)} traditional style)")

                for citation_id in all_citations:
                    # Skip numeric citations that might be footnotes or other references
                    if citation_id.isdigit():
                        continue

                    # If we haven't seen this citation before, add it to our tracking
                    if citation_id not in self._cited_papers:
                        # For \cite{} format, we preserve the exact key
                        if citation_id in cite_citations:
                            # Try to find a paper that matches this citation key
                            matched_paper = None
                            for paper in reference_papers:
                                author = ""
                                if paper.authors:
                                    if "," in paper.authors:
                                        author = paper.authors.split(",")[0].strip().lower()
                                    else:
                                        author = paper.authors.split(" ")[0].lower()
                                    author = ''.join(c for c in author if c.isalnum())

                                year = ""
                                if paper.update_date and len(paper.update_date) >= 4:
                                    year = paper.update_date[:4]

                                # If the key has the author or year, consider it a match
                                if (author and author.lower() in citation_id.lower()) or (year and year in citation_id):
                                    matched_paper = paper
                                    break

                            # Use the matching paper or create a placeholder that preserves the exact key
                            if matched_paper:
                                self._cited_papers[citation_id] = matched_paper.to_dict()
                                logger.info(f"Found matching paper for citation: {citation_id}")
                            else:
                                # Create a fallback citation that preserves the exact key
                                self._cited_papers[citation_id] = {
                                    'id': citation_id,
                                    'title': f"Reference: {citation_id}",
                                    'authors': f"{citation_id.split('2')[0].capitalize() if '2' in citation_id else citation_id[:6]} et al.",
                                    'journal_ref': "Referenced in text",
                                    'update_date': ''.join(c for c in citation_id if c.isdigit())[:4] or "2023"
                                }
                                logger.warning(f"Created placeholder for citation with no matching paper: {citation_id}")
                        else:
                            # Try to find a paper that matches this citation ID (AuthorYear format)
                            matched_paper = None
                            for paper in reference_papers:
                                # Check if author name is in the citation ID (common format: Smith2019)
                                author_match = False
                                if paper.authors:
                                    first_author = paper.authors.split(',')[0].split(' ')[0] if ',' in paper.authors or ' ' in paper.authors else paper.authors
                                    if first_author.lower() in citation_id.lower():
                                        author_match = True

                                # Check if year is in the citation ID
                                year_match = False
                                if paper.update_date and len(paper.update_date) >= 4:
                                    year = paper.update_date[:4]
                                    if year in citation_id:
                                        year_match = True

                                # If we have a good match, use this paper
                                if author_match or year_match:
                                    matched_paper = paper
                                    break

                            # If we found a matching paper, add it to our citations
                            if matched_paper:
                                self._cited_papers[citation_id] = matched_paper.to_dict()
                                logger.info(f"Found matching paper for citation: {citation_id}")
                            else:
                                # Create a fallback citation when no matching paper found
                                clean_key = citation_id

                                # Extract potential author and year from the citation key
                                author_match = re.search(r'([a-z]+)', clean_key.lower())
                                author = author_match.group(1) if author_match else "unknown"

                                year_match = re.search(r'(\d{4})', clean_key)
                                year = year_match.group(1) if year_match else "2023"

                                self._cited_papers[citation_id] = {
                                    'id': clean_key,
                                    'title': f"Reference: {clean_key}",
                                    'authors': f"{author.capitalize()} et al.",
                                    'journal_ref': "Referenced in text",
                                    'update_date': year
                                }
                                logger.warning(f"Created placeholder for citation with no matching paper: {citation_id}")

        logger.info(f"Extracted {len(self._cited_papers)} unique citations from the survey content")
        if not self._cited_papers:
            logger.warning("WARNING: No citations found in any section! The references.bib file will be empty.")
            # Add a default citation to ensure we have at least one reference
            self._cited_papers["default"] = {
                'id': "default",
                'title': "Example reference paper",
                'authors': "Author, Example",
                'journal_ref': "Journal of Important Research",
                'update_date': "2023"
            }

    def _replace_citations_with_bibtex_format(self, content: str) -> str:
        """Replace citation IDs with BibTeX \cite{key} format and preserve existing \cite{} commands"""
        import re

        # First check if there are already \cite{} commands in the content
        cite_pattern = r'\\cite\{([^}]+)\}'
        existing_citations = re.findall(cite_pattern, content)

        if existing_citations:
            logger.info(f"Found {len(existing_citations)} existing \\cite{{}} citations")

        # Add existing citations to our tracking
        for citation_key in existing_citations:
            if citation_key not in self._cited_papers:
                # Try to find a paper that matches this citation key
                matched_paper = None
                for paper in self._global_related_papers:
                    author = ""
                    if paper.authors:
                        if "," in paper.authors:
                            author = paper.authors.split(",")[0].strip().lower()
                        else:
                            author = paper.authors.split(" ")[0].lower()
                        author = ''.join(c for c in author if c.isalnum())

                    year = ""
                    if paper.update_date and len(paper.update_date) >= 4:
                        year = paper.update_date[:4]

                    # Check if citation key contains author and year
                    if author and year and author.lower() in citation_key.lower() and year in citation_key:
                        matched_paper = paper
                        break

                # If we found a match, add it to our citations
                if matched_paper:
                    self._cited_papers[citation_key] = matched_paper.to_dict()
                    logger.info(f"Found matching paper for \\cite{{{citation_key}}}")
                else:
                    # Create a fallback citation - preserve the exact citation key
                    self._cited_papers[citation_key] = {
                        'id': citation_key,
                        'title': f"Reference: {citation_key}",
                        'authors': f"{citation_key.split('2')[0].capitalize() if '2' in citation_key else citation_key[:6]} et al.",
                        'journal_ref': "Referenced in text",
                        'update_date': ''.join(c for c in citation_key if c.isdigit())[:4] or "2023"
                    }
                    logger.warning(f"Created placeholder for \\cite{{{citation_key}}}")

        # If the content already uses \cite{} format throughout, return it as is
        if '[' not in content or ']' not in content:
            return content

        # Replace each [AuthorYear] citation with \cite{key} format (for backward compatibility)
        def replace_citation(match):
            citation_id = match.group(1)
            # If it's one of our tracked citations, replace it
            if citation_id in self._cited_papers:
                # Use the exact citation key if possible, otherwise generate a stable one
                citation_key = citation_id if citation_id in self._cited_papers else self._generate_citation_key(citation_id)
                return f"\\cite{{{citation_key}}}"
            # Otherwise, leave it as is
            return match.group(0)

        # Apply the replacements for [AuthorYear] format (keeping for backwards compatibility)
        citation_pattern = r'\[([^\]]+)\]'
        return re.sub(citation_pattern, replace_citation, content)

    def _generate_citation_key(self, citation_id: str) -> str:
        """Generate a stable BibTeX citation key from a citation ID"""
        # If the citation_id already looks like a proper citation key, return it
        if citation_id.startswith('\\cite{') and citation_id.endswith('}'):
            return citation_id[6:-1]

        # If this is from a \cite{key} in the paper, use the exact key
        # This is important to maintain the exact citation keys used in the paper
        if '\\cite{' in citation_id.lower():
            parts = citation_id.split('{')
            if len(parts) > 1:
                parts = parts[1].split('}')
                if parts:
                    return parts[0]

        # For existing citation keys without the \cite wrapper, use them directly
        # This preserves keys like "smith2019" or "levels2016"
        if citation_id in self._cited_papers and not citation_id.startswith('[') and not citation_id.endswith(']'):
            return citation_id

        paper = self._cited_papers[citation_id]

        # Extract author
        if paper.get('authors'):
            if "," in paper['authors']:
                author = paper['authors'].split(",")[0].strip().lower()
            else:
                author = paper['authors'].split(" ")[0].lower()
            # Remove non-alphanumeric characters
            author = ''.join(c for c in author if c.isalnum())
        else:
            author = "unknown"

        # Extract year
        year = ""
        if paper.get('update_date') and len(paper.get('update_date')) >= 4:
            year = paper.get('update_date')[:4]
        else:
            year = "0000"

        # Create a unique identifier based on paper ID
        if paper.get('id') and paper.get('journal_ref'):
            # Use the first few characters of the paper ID
            unique_id = paper.get('id', '').replace('.', '').replace('-', '')[:4]
        else:
            # Use a hash of the title if there's no ID
            import hashlib
            title_hash = hashlib.md5(paper.get('title', 'unknown').encode()).hexdigest()[:4]
            unique_id = title_hash

        return f"{author}{year}{unique_id}"

    def _generate_bibtex_file(self, paper_dir: Path) -> str:
        """Generate a BibTeX file from the citation data and return the filename"""
        # Create BibTeX entries
        bibtex_entries = []

        logger.info(f"Generating BibTeX entries for {len(self._cited_papers)} citations")

        for citation_id, paper in self._cited_papers.items():
            # Generate a stable citation key, prioritizing the exact key used in the paper
            # Use the direct citation_id if it's a citation key and not surrounded by brackets
            if not citation_id.startswith('[') and not citation_id.endswith(']'):
                citation_key = citation_id
            else:
                # Otherwise, generate a stable key
                citation_key = self._generate_citation_key(citation_id)

            # Format authors for BibTeX
            authors = paper.get('authors', "Unknown")

            # Format title
            title = paper.get('title', "Untitled")

            # Determine the type of publication
            if paper.get('journal_ref') and 'conference' in paper['journal_ref'].lower():
                entry_type = "inproceedings"
                venue_field = "booktitle"
                venue = paper['journal_ref']
            elif paper.get('journal_ref'):
                entry_type = "article"
                venue_field = "journal"
                venue = paper['journal_ref']
            else:
                entry_type = "misc"
                venue_field = "note"
                venue = f"arXiv:{paper.get('id', 'unknown')}"

            # Extract year
            year = ""
            if paper.get('update_date') and len(paper.get('update_date')) >= 4:
                year = paper.get('update_date')[:4]
            else:
                year = "2023"  # Default to current year if none available

            # Build the BibTeX entry
            entry = [
                f"@{entry_type}{{{citation_key},",
                f"  author = {{{authors}}},",
                f"  title = {{{title}}},",
                f"  {venue_field} = {{{venue}}},",
            ]

            if year:
                entry.append(f"  year = {{{year}}},")

            if paper.get('id') and entry_type != "misc":
                entry.append(f"  note = {{arXiv:{paper['id']}}},")

            entry.append("}")
            bibtex_entries.append("\n".join(entry))

        # If no citations, add a placeholder
        if not bibtex_entries:
            logger.warning("No citations found, adding placeholder reference")
            bibtex_entries.append("@article{placeholder,\n  author = {Placeholder, Author},\n  title = {This is a placeholder reference},\n  journal = {Journal of Examples},\n  year = {2023}\n}")

        # Write to file
        bibtex_filename = "references.bib"
        bibtex_path = paper_dir / bibtex_filename

        with open(bibtex_path, "w") as f:
            f.write("\n\n".join(bibtex_entries))

        logger.info(f"Generated BibTeX file with {len(bibtex_entries)} entries at {bibtex_path}")

        return bibtex_filename
