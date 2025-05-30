#!/usr/bin/env python3
import logging
import json
from typing import Dict, Any, Optional
import aiohttp
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama provider

        Args:
            config: Dictionary containing configuration
                base_url: Base URL for the Ollama API
                model: Default model to use for generation
        """
        super().__init__(config)
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.default_model = self.config.get("model", "llama2")
        logger.info(f"Initialized Ollama provider with base URL: {self.base_url}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama model

        Args:
            prompt: The input prompt for generation
            **kwargs: Additional parameters:
                model: Override the default model

        Returns:
            Generated text as a string
        """
        model = kwargs.get("model", self.default_model)

        try:
            logger.info(f"Generating text with Ollama model: {model}")
            logger.debug(f"Prompt: {prompt[:200] + '...' if len(prompt) > 200 else prompt}")

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
                                last_response = json.loads(lines[-1])
                                generated_text = last_response.get("response", "")

                                # Remove <think> tags from response if present
                                # Find and remove content between <think> and </think> tags
                                while "<think>" in generated_text and "</think>" in generated_text:
                                    think_start = generated_text.find("<think>")
                                    think_end = generated_text.find("</think>") + len("</think>")
                                    generated_text = generated_text[:think_start] + generated_text[think_end:]
                                    logger.debug("Removed thinking content from response")

                                logger.debug(f"Response (first 200 chars): {generated_text[:200] + '...' if len(generated_text) > 200 else generated_text}")
                                logger.info(f"Successfully generated text (length: {len(generated_text)} chars)")
                                return generated_text
                        except Exception as e:
                            logger.error(f"Error parsing Ollama response: {str(e)}")
                            return ""
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error (status {response.status}): {error_text}")
                        return ""
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
            return ""
