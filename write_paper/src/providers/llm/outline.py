#!/usr/bin/env python3
import logging
import re
from typing import List, Any, Dict, Optional
from .factory import LLMProviderFactory
from ..utils.db import DatabaseManager

logger = logging.getLogger(__name__)

class OutlineGenerator:
    """Generator for creating paper outlines using LLM models and reference papers"""

    def __init__(
        self,
        provider_type: str = "ollama",
        provider_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        reference_num: int = 1500,
        num_sections: int = 8
    ):
        """
        Initialize the outline generator

        Args:
            provider_type: Type of LLM provider to use (ollama, openai, openrouter)
            provider_config: Configuration for the LLM provider
            model: Model to use for generation (overrides provider config model)
            reference_num: Number of reference papers to use for context
            num_sections: Target number of sections for the outline
        """
        self.provider_type = provider_type
        self.provider_config = provider_config or {}

        # If model is provided, override the config
        if model:
            self.provider_config["model"] = model

        self.db = DatabaseManager()
        self.reference_num = reference_num
        self.num_sections = num_sections

        logger.info(f"Initialized OutlineGenerator with provider: {provider_type}")
        if model:
            logger.info(f"Using model: {model}")
        logger.info(f"Target number of sections: {num_sections}")

    async def generate_outline(self, topic: str) -> List[str]:
        """
        Generate paper outline based on topic and references

        Args:
            topic: The research topic for the paper

        Returns:
            List of section titles for the outline
        """
        logger.info(f"Generating outline for topic: {topic}")

        # Get relevant papers for outline generation
        logger.debug("Getting topic embedding")
        topic_embedding = self.db.get_embedding(topic)

        logger.info(f"Finding reference papers (limit: {self.reference_num})")
        reference_papers = self.db.find_similar_papers(topic_embedding, limit=self.reference_num)
        logger.info(f"Found {len(reference_papers)} reference papers")

        # Create context from references
        logger.debug("Creating reference context")
        context = self._create_reference_context(reference_papers)

        # Generate outline using context
        logger.debug("Creating outline prompt")
        prompt = self._create_outline_prompt(topic, context)

        logger.info(f"Generating outline with provider: {self.provider_type}")
        response = await LLMProviderFactory.generate_with_provider(
            self.provider_type,
            prompt,
            self.provider_config
        )

        # Parse outline
        outline = self._parse_outline(response)
        logger.info(f"Generated outline with {len(outline)} sections")
        logger.debug(f"Outline sections: {outline}")

        return outline

    def _create_reference_context(self, papers: List[Any]) -> str:
        """Create context from reference papers"""
        context = []
        for paper in papers:
            context.append(f"Title: {paper.title}")
            if paper.abstract:
                context.append(f"Abstract: {paper.abstract}")  # Include full abstract
        return "\n\n".join(context)

    def _create_outline_prompt(self, topic: str, context: str) -> str:
        """Create prompt for outline generation"""
        return f"""As an academic survey paper outline generator, create a detailed outline for a comprehensive survey paper on the topic: "{topic}".

Context (Recent papers in the field):
{context}

Requirements:
1. The outline should follow standard academic survey paper structure with clear progression of ideas
2. Each section should address specific aspects of the topic as follows:
   - Introduction: Define the scope, importance, and historical context of {topic}
   - Background/Fundamentals: Cover essential concepts, terminology, and foundational knowledge
   - Literature Review: Analyze existing research methodologies and approaches
   - Technical Sections: Detail key technologies, algorithms, and implementations
   - Comparative Analysis: Evaluate strengths and limitations of different approaches
   - Applications: Explore real-world implementations and use cases
   - Challenges: Identify current limitations and obstacles in the field
   - Future Directions: Discuss emerging trends and research opportunities
3. Balance theoretical foundations with practical applications
4. Ensure comprehensive coverage of all major developments in the field
5. Create exactly {self.num_sections} sections
6. Do NOT include any subsections in the outline
7. Exclude Abstract and Conclusion from the outline
8. Do not use markdown formatting like #, ** or *

Format the outline as a numbered list of section titles (1. Section Title, 2. Section Title, etc.), one per line.
The outline must contain exactly {self.num_sections} sections.

Think step by step to generate a cohesive and logically structured outline:"""

    def _parse_outline(self, text: str) -> List[str]:
        """Parse outline from generated text"""
        sections = []
        # Remove any conversational text before the actual outline
        if "Here is" in text:
            text = text[text.find("Here is"):].split("\n", 1)[1]

        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Skip common prefixes and conversational text
            if any(line.startswith(x) for x in [
                "Here's",
                "Here is",
                "I've created",
                "I have created",
                "The outline",
                "This outline"
            ]):
                continue

            # Extract section from numbered list format (1. Section Title)
            # Match patterns like "1. Introduction" or "1 - Introduction"
            match = re.match(r'^\d+[\.\s\-]*\s*(.+)$', line)
            if match:
                section_title = match.group(1).strip()
                if section_title and section_title not in sections:
                    sections.append(section_title)
                continue

            # Remove numbering and bullet points for any other format
            line = line.lstrip('0123456789.- *')
            line = line.strip()

            # Skip empty lines after cleaning
            if not line:
                continue

            # Skip lines that appear to be descriptions rather than section titles
            if len(line.split()) > 7 or line.endswith('.'):
                continue

            # Add section if it's not already included and looks like a title
            if line not in sections:
                sections.append(line)

        # Always ensure Abstract is first and Conclusion is last
        if "Abstract" not in sections:
            sections.insert(0, "Abstract")
        if "Conclusion" not in sections:
            sections.append("Conclusion")

        # # Ensure we have exactly the requested number of sections
        # # If we have too many, trim the middle sections
        # if len(sections) > self.num_sections + 2:  # +2 for Abstract and Conclusion
        #     # Keep Abstract, trim middle sections, keep Conclusion
        #     sections = [sections[0]] + sections[1:self.num_sections+1] + [sections[-1]]

        return sections
