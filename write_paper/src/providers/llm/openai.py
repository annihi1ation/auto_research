#!/usr/bin/env python3
import logging
from typing import Dict, Any, Optional
import aiohttp
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI provider

        Args:
            config: Dictionary containing configuration
                api_key: OpenAI API key
                base_url: Base URL for the OpenAI API (defaults to official API)
                model: Default model to use for generation
                org_id: Optional organization ID
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            logger.warning("No API key provided for OpenAI provider")

        self.base_url = self.config.get("base_url", "https://api.openai.com/v1")
        self.default_model = self.config.get("model", "gpt-3.5-turbo-instruct")
        self.org_id = self.config.get("org_id")

        logger.info(f"Initialized OpenAI provider with model: {self.default_model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI model

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
            logger.error("Cannot generate text: No API key provided for OpenAI")
            return ""

        model = kwargs.get("model", self.default_model)
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id

        try:
            logger.info(f"Generating text with OpenAI model: {model}")
            logger.debug(f"Prompt: {prompt[:200] + '...' if len(prompt) > 200 else prompt}")

            # Choose endpoint based on model type
            if "instruct" in model.lower():
                # Completions endpoint for instruction-tuned models
                endpoint = f"{self.base_url}/completions"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            else:
                # Chat completions endpoint for chat models
                endpoint = f"{self.base_url}/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

            async with aiohttp.ClientSession() as session:
                logger.debug(f"Making request to OpenAI API: {endpoint}")
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    response_json = await response.json()

                    if response.status == 200:
                        if "instruct" in model.lower():
                            # Extract text from completions response
                            generated_text = response_json.get("choices", [{}])[0].get("text", "").strip()
                        else:
                            # Extract text from chat completions response
                            generated_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                        logger.debug(f"Response (first 200 chars): {generated_text[:200] + '...' if len(generated_text) > 200 else generated_text}")
                        logger.info(f"Successfully generated text (length: {len(generated_text)} chars)")
                        return generated_text
                    else:
                        error = response_json.get("error", {})
                        error_message = error.get("message", str(response_json))
                        logger.error(f"OpenAI API error (status {response.status}): {error_message}")
                        return ""
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
            return ""
