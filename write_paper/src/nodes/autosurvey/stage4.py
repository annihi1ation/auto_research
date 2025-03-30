#!/usr/bin/env python3
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from src.state import ResearchState
from src.utils.ollama import OllamaClient
from .utils import save_pipeline_state
from .citation import CitationManager

logger = logging.getLogger(__name__)

@dataclass
class Stage4_EvaluationIteration(BaseNode[ResearchState]):
    """Stage 4: Rigorous Evaluation & Iteration

    This stage evaluates the survey paper on multiple criteria and iterates to
    produce the best version.
    """
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "End[Dict[str, str]]":
        # Initialize the citation manager
        citation_manager = CitationManager()

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
        global_related_papers = []
        for section_title, papers in ctx.state.section_publications.items():
            for paper in papers:
                if paper.id not in [p.id for p in global_related_papers]:
                    global_related_papers.append(paper)

        # Set the global papers in the citation manager
        citation_manager.set_global_papers(global_related_papers)
        logger.info(f"Collected {len(global_related_papers)} unique papers for potential citations")

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
        formatted_sections = citation_manager.format_paper(
            ctx.state.iteration_surveys[best_idx]["sections"],
            paper_dir,
            ctx.state.topic
        )

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
            "reference_papers_count": len(citation_manager._cited_papers),
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
