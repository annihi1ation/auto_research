#!/usr/bin/env python3
import logging
from typing import Dict, Any, Optional
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)

class LLMProviderFactory:
    """Factory class for creating LLM provider instances"""

    @staticmethod
    def create_provider(provider_type: str, config: Optional[Dict[str, Any]] = None) -> BaseLLMProvider:
        """
        Create and return an instance of the specified LLM provider

        Args:
            provider_type: Type of provider to create (ollama, openai, openrouter)
            config: Provider-specific configuration dictionary

        Returns:
            An instance of BaseLLMProvider
        """
        config = config or {}
        provider_type = provider_type.lower()

        logger.info(f"Creating LLM provider of type: {provider_type}")

        if provider_type == "ollama":
            return OllamaProvider(config)
        elif provider_type == "openai":
            return OpenAIProvider(config)
        elif provider_type == "openrouter":
            return OpenRouterProvider(config)
        else:
            logger.warning(f"Unknown provider type: {provider_type}, defaulting to Ollama")
            return OllamaProvider(config)

    @staticmethod
    async def generate_with_provider(
        provider_type: str,
        prompt: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Convenience method to create a provider and generate text in one call

        Args:
            provider_type: Type of provider to use
            prompt: The prompt to send to the LLM
            config: Provider configuration
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        provider = LLMProviderFactory.create_provider(provider_type, config)
        return await provider.generate(prompt, **kwargs)
