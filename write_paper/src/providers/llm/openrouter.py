#!/usr/bin/env python3
import logging
import os
from typing import Dict, Any, Optional
import aiohttp
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider implementation for accessing various AI models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenRouter provider

        Args:
            config: Dictionary containing configuration
                api_key: OpenRouter API key
                base_url: Base URL for the OpenRouter API
                model: Default model to use for generation
                route_prefix: Optional route prefix
        """
        super().__init__(config)
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided for OpenRouter provider")

        self.base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        self.default_model = self.config.get("model", "openai/gpt-3.5-turbo")
        self.route_prefix = self.config.get("route_prefix", "")

        logger.info(f"Initialized OpenRouter provider with model: {self.default_model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenRouter model

        Args:
            prompt: The input prompt for generation
            **kwargs: Additional parameters:
                model: Override the default model
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature (0-2)

        Returns:
            Generated text as a string
        """
        if not self.api_key:
            logger.error("Cannot generate text: No API key provided for OpenRouter")
            return ""

        model = kwargs.get("model", self.default_model)
        max_tokens = kwargs.get("max_tokens", 10000)
        temperature = kwargs.get("temperature", 0.9)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/researcher",  # Site URL for attribution
            "X-Title": "Research Paper Generator"  # Name of your app
        }

        if self.route_prefix:
            headers["X-Route-Prefix"] = self.route_prefix

        try:
            logger.info(f"Generating text with OpenRouter model: {model}")
            logger.debug(f"Prompt: {prompt[:200] + '...' if len(prompt) > 200 else prompt}")

            # OpenRouter uses the OpenAI-compatible chat completions endpoint
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            async with aiohttp.ClientSession() as session:
                logger.debug(f"Making request to OpenRouter API: {endpoint}")
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    response_json = await response.json()

                    if response.status == 200:
                        # Extract text from chat completions response
                        generated_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                        # Log additional information specific to OpenRouter
                        if "model" in response_json:
                            used_model = response_json.get("model", "")
                            logger.info(f"Used model: {used_model}")

                        # while "<think>" in generated_text and "</think>" in generated_text:
                        #         think_start = generated_text.find("<think>")
                        #         think_end = generated_text.find("</think>") + len("</think>")
                        #         generated_text = generated_text[:think_start] + generated_text[think_end:]
                        #         logger.debug("Removed thinking content from response")

                        # Remove ** symbols from the text
                        # generated_text = generated_text.replace("**", "")
                        # logger.debug("Removed ** symbols from response")

                        logger.debug(f"Response (first 200 chars): {generated_text[:200] + '...' if len(generated_text) > 200 else generated_text}")
                        logger.info(f"Successfully generated text (length: {len(generated_text)} chars)")
                        return generated_text
                    else:
                        error = response_json.get("error", {})
                        error_message = error.get("message", str(response_json))
                        logger.error(f"OpenRouter API error (status {response.status}): {error_message}")
                        return ""
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}", exc_info=True)
            return ""
