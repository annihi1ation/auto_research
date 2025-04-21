#!/usr/bin/env python3
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM provider with optional configuration

        Args:
            config: Dictionary containing provider-specific configuration
        """
        self.config = config or {}
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM model

        Args:
            prompt: The input prompt for generation
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text as a string
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
