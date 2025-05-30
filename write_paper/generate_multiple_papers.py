#!/usr/bin/env python3
import asyncio
import logging
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.main import generate_paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_generator")

def load_topics(config_path="benchmarks/configs/default.yaml"):
    """Load topics from the benchmark configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return [topic["name"] for topic in config.get("topics", [])]
    except Exception as e:
        logger.error(f"Failed to load topics from {config_path}: {str(e)}")
        return []

async def generate_papers(topics, output_dir="generated_papers", model="llama2", provider="ollama", num_papers=5,
                         reference_num=100, num_sections=6, papers_per_section=5):
    """Generate papers for the specified topics"""
    os.makedirs(output_dir, exist_ok=True)

    # Select a subset of topics if num_papers is less than the total
    selected_topics = topics

    logger.info(f"Generating {len(selected_topics)} papers with topics:")
    for topic in selected_topics:
        logger.info(f"- {topic}")

    generated_papers = []

    for topic in selected_topics:
        try:
            logger.info(f"Generating paper for topic: {topic}")

            # Create a specific output directory for this topic
            topic_output_dir = os.path.join(output_dir, topic.replace(' ', '_'))

            # Generate the paper using the existing function from src/main.py
            await generate_paper(
                topic=topic,
                output_dir=topic_output_dir,
                model=model,
                provider=provider,
                reference_num=reference_num,
                num_sections=num_sections,
                papers_per_section=papers_per_section
            )

            # Clean up any duplicate bibliography files in the output directory
            logger.info(f"Cleaning up any duplicate bibliography files in {topic_output_dir}")
            for file in os.listdir(topic_output_dir):
                if "_dblp_" in file or "_temp_" in file:
                    try:
                        os.remove(os.path.join(topic_output_dir, file))
                        logger.info(f"Removed duplicate file: {file}")
                    except OSError as e:
                        logger.warning(f"Could not remove file {file}: {e}")

            # Record the expected output path
            paper_path = Path(topic_output_dir) / f"{topic.replace(' ', '_')}_survey.tex"
            generated_papers.append(paper_path)
            logger.info(f"Successfully generated paper: {paper_path}")

        except Exception as e:
            logger.error(f"Failed to generate paper for topic '{topic}': {str(e)}")

    return generated_papers

async def main_async():
    """Async main entry point"""
    parser = argparse.ArgumentParser(description="Generate multiple research papers")
    parser.add_argument("--config", default="benchmarks/configs/default.yaml", help="Path to the topics config file")
    parser.add_argument("--output", default="generated_papers", help="Output directory for generated papers")
    parser.add_argument("--num-papers", type=int, default=5, help="Number of papers to generate")
    parser.add_argument("--model", default="llama2", help="LLM model to use")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"],
                        help="LLM provider to use (ollama or openrouter)")
    parser.add_argument("--reference-num", type=int, default=100, help="Number of references per paper")
    parser.add_argument("--num-sections", type=int, default=6, help="Number of sections per paper")
    parser.add_argument("--papers-per-section", type=int, default=5, help="Papers per section")

    args = parser.parse_args()

    # Load topics from config
    topics = load_topics(args.config)

    if not topics:
        logger.error("No topics found in the configuration. Exiting.")
        return

    # Check if OpenRouter API key is set when using openrouter provider
    if args.provider == "openrouter":
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print("Error: When using the 'openrouter' provider, you must set the OPENROUTER_API_KEY environment variable.")
            print("Example: export OPENROUTER_API_KEY=your_api_key")
            sys.exit(1)

        # Set default OpenRouter model if still using the default Ollama model
        if args.model == "llama2":
            args.model = "openai/gpt-3.5-turbo"
            logger.info(f"Using default OpenRouter model: {args.model}")

    # Generate papers
    generated_papers = await generate_papers(
        topics=topics,
        output_dir=args.output,
        model=args.model,
        provider=args.provider,
        num_papers=args.num_papers,
        reference_num=args.reference_num,
        num_sections=args.num_sections,
        papers_per_section=args.papers_per_section
    )

    print(f"\nSuccessfully generated {len(generated_papers)} papers:")
    for paper in generated_papers:
        print(f"- {paper}")

def main():
    """Main entry point"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
