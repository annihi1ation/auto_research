#!/usr/bin/env python3
import asyncio
import logging
import argparse
from pathlib import Path
from pydantic_graph import Graph
import sys
import os
import json
from src.ultils.latex_generator import generate_latex_document
from src.ultils.post_processor import post_process_paper

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.state import ResearchState, OutlineConfig, SearchConfig
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
    model: str = "",
    provider: str = "ollama",
    reference_num: int = 1500,
    num_sections: int = 8,
    papers_per_section: int = 20,
    use_default_outline: bool = False
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
            reference_num=reference_num,
            num_sections=num_sections,
            use_default_outline=use_default_outline
        )

        # Initialize search config
        search_config = SearchConfig(
            papers_per_section=papers_per_section
        )

        # Set provider in state metadata
        state = ResearchState(
            topic=topic,
            outline_config=outline_config,
            search_config=search_config
        )

        # Store the provider type in stage_results for access by nodes
        state.stage_results["provider"] = provider

        # Add the API key to the provider config if using OpenRouter
        if provider == "openrouter":
            state.stage_results["provider_config"] = {
                "api_key": os.environ.get("OPENROUTER_API_KEY"),
                "model": model
            }
        else:
            state.stage_results["provider_config"] = {"model": model}

        logger.info(f"Using provider: {provider} with model: {model}")


        logger.info("Using standard pipeline")
        result = await research_graph.run(PlanningPhase(), state=state)

        # Print the result object to understand its structure and attributes
        logger.info(f"GraphRunResult type: {type(result)}")
        logger.info(f"GraphRunResult dir: {dir(result)}")

        # Based on dir() output, we see that GraphRunResult has an 'output' attribute
        # Let's try to access it and log its properties
        if hasattr(result, 'output'):
            logger.info(f"Result output type: {type(result.output)}")
            if hasattr(result.output, 'value'):
                logger.info(f"Result output value type: {type(result.output.value)}")
                formatted_paper = result.output.value
            else:
                logger.info("Result output does not have a 'value' attribute")
                # Look for other potential attributes that might contain the data
                formatted_paper = result.output
        else:
            logger.info("Result does not have an 'output' attribute")
            # Try reading from a known output JSON file
            try:
                result_file = Path("result") / f"{topic}.json"
                logger.info(f"Attempting to load result from: {result_file}")
                if result_file.exists():
                    import json
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    formatted_paper = result_data.get('state', {}).get('generated_sections', {})
                    logger.info(f"Loaded generated sections from result file, sections: {list(formatted_paper.keys())}")
                else:
                    logger.error(f"Result file not found: {result_file}")
                    raise FileNotFoundError(f"Could not find result file: {result_file}")
            except Exception as e:
                logger.error(f"Failed to load result data: {str(e)}")
                raise

        # Generate the LaTeX document
        paper_path = generate_latex_document(
            formatted_paper=formatted_paper,
            state_outline=state.outline,
            output_path=output_path,
            topic=topic
        )

        logger.info(f"Paper saved to: {paper_path}")

        # Run post-processing on the generated paper
        logger.info("Starting post-processing workflow")
        if post_process_paper(str(paper_path)):
            logger.info("Post-processing completed successfully")
        else:
            logger.warning("Post-processing encountered issues, check logs for details")

        # Clean up any existing _dblp files from previous runs
        output_dir = os.path.dirname(paper_path)
        for file in os.listdir(output_dir):
            if file.endswith("_dblp.bib") or file.endswith("_temp_dblp.bib"):
                try:
                    os.remove(os.path.join(output_dir, file))
                    logger.info(f"Removed duplicate file: {file}")
                except OSError as e:
                    logger.warning(f"Could not remove file {file}: {e}")

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
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"],
                       help="LLM provider to use (ollama or openrouter)")
    parser.add_argument("--model", default="llama2",
                       help="Model to use (e.g., llama2 for Ollama, or openai/gpt-4 for OpenRouter)")
    parser.add_argument("--reference-num", type=int, default=1500, help="Number of reference papers to consider")
    parser.add_argument("--num-sections", type=int, default=8, help="Number of sections to generate")
    parser.add_argument("--papers-per-section", type=int, default=20, help="Number of papers to retrieve per section")
    parser.add_argument("--use-default-outline", action="store_true",
                       help="Use default outline instead of generating one with LLM")

    # Check if OpenRouter API key is set when using openrouter provider
    args = parser.parse_args()
    logger.info("Command line arguments: %s", args)

    if args.provider == "openrouter":
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print("Error: When using the 'openrouter' provider, you must set the OPENROUTER_API_KEY environment variable.")
            print("Example: export OPENROUTER_API_KEY=your_api_key")
            sys.exit(1)

    asyncio.run(generate_paper(
        topic=args.topic,
        output_dir=args.output,
        model=args.model,
        provider=args.provider,
        reference_num=args.reference_num,
        num_sections=args.num_sections,
        papers_per_section=args.papers_per_section,
        use_default_outline=args.use_default_outline
    ))

if __name__ == "__main__":
    try:
        main()
        logger.info("Paper generation completed successfully")
    except Exception as e:
        logger.error("Paper generation failed", exc_info=True)
        raise
