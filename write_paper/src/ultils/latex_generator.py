import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def generate_latex_document(
    formatted_paper: Dict[str, str],
    state_outline: List[str],
    output_path: Path,
    topic: str
) -> Path:
    """
    Generate a LaTeX document from the formatted paper sections.

    Args:
        formatted_paper: Dictionary containing formatted paper sections
        state_outline: List of section names from the research state
        output_path: Path to the output directory
        topic: Topic of the paper

    Returns:
        Path to the generated LaTeX file
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the file path for the generated paper
    paper_path = output_path / f"{topic.replace(' ', '_')}_survey.tex"

    # Combine all sections in IEEE LaTeX order
    paper_content = []

    # Add preamble
    paper_content.append(formatted_paper["Preamble"])

    # Add title and author
    paper_content.append(formatted_paper["Title"])

    # Add abstract
    paper_content.append(formatted_paper["Abstract"])

    # Add main sections
    for section in state_outline:
        if section.lower() not in ["abstract", "title"]:
            if section in formatted_paper:
                paper_content.append(formatted_paper[section])

    # Add document end
    paper_content.append(formatted_paper["End"])

    # Write to file
    with open(paper_path, "w") as f:
        f.write("\n\n".join(paper_content))

    logger.info(f"Paper saved to: {paper_path}")
    return paper_path
