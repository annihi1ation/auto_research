#!/usr/bin/env python3
import logging
from typing import List, Dict, Any
import aiohttp
from ..utils.db import DatabaseManager

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, model: str, prompt: str) -> str:
        """Generate text using Ollama model"""
        try:
            logger.info("Generating text with model: %s", model)
            logger.debug("Prompt: %s", prompt[:200] + "..." if len(prompt) > 200 else prompt)

            async with aiohttp.ClientSession() as session:
                logger.debug("Making request to Ollama API")
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False}
                ) as response:
                    if response.status == 200:
                        # Read the entire response text
                        response_text = await response.text()
                        # Parse the last line which contains the final response
                        try:
                            lines = [line for line in response_text.split('\n') if line.strip()]
                            if lines:
                                import json
                                last_response = json.loads(lines[-1])
                                generated_text = last_response.get("response", "")
                                logger.debug("Response (first 200 chars): %s",
                                           generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)
                                logger.info("Successfully generated text (length: %d chars)", len(generated_text))
                                return generated_text
                        except Exception as e:
                            logger.error("Error parsing Ollama response: %s", str(e))
                            return ""
                    else:
                        error_text = await response.text()
                        logger.error("Ollama API error (status %d): %s", response.status, error_text)
                        return ""
        except Exception as e:
            logger.error("Error calling Ollama API: %s", str(e), exc_info=True)
            return ""

class OutlineGenerator:
    def __init__(self, model: str = "llama2", reference_num: int = 1500):
        self.client = OllamaClient()
        self.model = model
        self.db = DatabaseManager()
        self.reference_num = reference_num

    async def generate_outline(self, topic: str) -> List[str]:
        """Generate paper outline based on topic and references"""
        logger.info("Generating outline for topic: %s", topic)

        # Get relevant papers for outline generation
        logger.debug("Getting topic embedding")
        topic_embedding = self.db.get_embedding(topic)

        logger.info("Finding reference papers (limit: %d)", self.reference_num)
        reference_papers = self.db.find_similar_papers(topic_embedding, limit=self.reference_num)
        logger.info("Found %d reference papers", len(reference_papers))

        # Create context from references
        logger.debug("Creating reference context")
        context = self._create_reference_context(reference_papers)

        # Generate outline using context
        logger.debug("Creating outline prompt")
        prompt = self._create_outline_prompt(topic, context)

        logger.info("Generating outline with model: %s", self.model)
        response = await self.client.generate(self.model, prompt)

        # Parse outline
        outline = self._parse_outline(response)
        logger.info("Generated outline with %d sections", len(outline))
        logger.debug("Outline sections: %s", outline)

        return outline

    def _create_reference_context(self, papers: List[Any]) -> str:
        """Create context from reference papers"""
        context = []
        for paper in papers:
            context.append(f"Title: {paper.title}")
            if paper.abstract:
                context.append(f"Abstract: {paper.abstract[:200]}...")  # Truncate long abstracts
        return "\n\n".join(context)

    def _create_outline_prompt(self, topic: str, context: str) -> str:
        """Create prompt for outline generation"""
        return f"""As an academic survey paper outline generator, create a detailed outline for a survey paper on the topic: "{topic}".

Context (Recent papers in the field):
{context}

Requirements:
1. The outline should follow standard survey paper structure
2. Sections should reflect the major themes and developments in the field
3. Include both high-level overview sections and detailed technical sections
4. Consider future directions and open challenges

Format the outline as a list of section titles, one per line.

Generate the outline:"""

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

            # Remove numbering and bullet points
            line = line.lstrip('0123456789.- *')
            line = line.strip()

            # Skip empty lines after cleaning
            if not line:
                continue

            # Add section if it's not already included
            if line not in sections:
                sections.append(line)

        # Always ensure Abstract is first and Conclusion is last
        if "Abstract" not in sections:
            sections.insert(0, "Abstract")
        if "Conclusion" not in sections:
            sections.append("Conclusion")

        return sections
