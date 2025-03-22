#!/usr/bin/env python3
import asyncio
import logging
import argparse
from pathlib import Path
from pydantic_graph import Graph
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.state import ResearchState, OutlineConfig
from src.nodes.planning import PlanningPhase, OutlineGeneration, SectionPlanning
from src.nodes.research import PaperSearch, ContentAnalysis, ContentSynthesis
from src.nodes.generation import PaperGeneration

def setup_logging(log_file: str = 'research_paper.log'):
    """Configure logging to output to both file and console with detailed formatting"""
    # Create formatters and handlers
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with detailed output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler with detailed output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured with output to console and %s", log_file)
    return logger

logger = setup_logging()

# Define the research paper generation graph
research_graph = Graph(
    nodes=[
        PlanningPhase,
        OutlineGeneration,
        SectionPlanning,
        PaperSearch,
        ContentAnalysis,
        ContentSynthesis,
        PaperGeneration
    ]
)

async def generate_paper(
    topic: str,
    output_dir: str = "output",
    model: str = "llama2",
    reference_num: int = 1500
) -> None:
    """Generate a research paper on the given topic"""
    try:
        logger.info(f"Starting paper generation for topic: {topic}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize state
        outline_config = OutlineConfig(
            model=model,
            reference_num=reference_num
        )
        state = ResearchState(
            topic=topic,
            outline_config=outline_config
        )

        # Run the workflow
        result, history = await research_graph.run(PlanningPhase(), state=state)

        # Save the generated paper in markdown format
        paper_path = output_path / f"{topic.replace(' ', '_')}_survey.md"

        # Combine all sections in order
        paper_content = []
        paper_content.append(result["Title"])
        paper_content.append("\n## Abstract\n")
        paper_content.append(result["Abstract"])
        paper_content.append("\n" + result["Table of Contents"] + "\n")

        for section in state.outline:
            if section.lower() not in ["abstract", "title"]:
                paper_content.append(result[section])

        if "References" in result:
            paper_content.append(result["References"])

        # Write to file
        with open(paper_path, "w") as f:
            f.write("\n\n".join(paper_content))

        logger.info(f"Paper saved to: {paper_path}")
        print(f"\nPaper generation completed. Output saved to: {paper_path}")
        print(f"State files are available in the paper_states directory")

    except Exception as e:
        error_msg = f"Error during paper generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\nError: {error_msg}")
        raise e

def main():
    """Main entry point"""
    logger.info("Starting paper generation process")
    parser = argparse.ArgumentParser(description="Generate a research survey paper")
    parser.add_argument("--topic", required=True, help="Topic of the survey paper")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--model", default="llama2", help="Ollama model to use")
    parser.add_argument("--reference-num", type=int, default=1500, help="Number of reference papers to consider")

    args = parser.parse_args()
    logger.info("Command line arguments: %s", args)

    asyncio.run(generate_paper(
        topic=args.topic,
        output_dir=args.output,
        model=args.model,
        reference_num=args.reference_num
    ))

if __name__ == "__main__":
    try:
        main()
        logger.info("Paper generation completed successfully")
    except Exception as e:
        logger.error("Paper generation failed", exc_info=True)
        raise
