"""
Client abstraction layer for LLM providers.
"""
from .base_client import BaseClient
from .ollama_client import OllamaClient

# Optional cloud provider clients (require additional packages)
try:
    from .anthropic_client import AnthropicClient
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AnthropicClient = None
    ANTHROPIC_AVAILABLE = False

try:
    from .openai_client import OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAIClient = None
    OPENAI_AVAILABLE = False

try:
    from .gemini_client import GeminiClient
    GEMINI_AVAILABLE = True
except ImportError:
    GeminiClient = None
    GEMINI_AVAILABLE = False


def get_available_providers():
    """Return dict of available providers and their status."""
    return {
        'ollama': True,
        'anthropic': ANTHROPIC_AVAILABLE,
        'openai': OPENAI_AVAILABLE,
        'gemini': GEMINI_AVAILABLE,
    }


def create_client(provider: str, **kwargs):
    """
    Factory function to create a client for the specified provider.

    Args:
        provider: Provider name ('ollama', 'anthropic', 'openai', 'gemini')
        **kwargs: Provider-specific parameters

    Returns:
        Client instance

    Raises:
        ValueError: If provider is not available or invalid
    """
    if provider == 'ollama':
        return OllamaClient(**kwargs)
    elif provider == 'anthropic':
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic client not available. Install with: pip install anthropic")
        return AnthropicClient(**kwargs)
    elif provider == 'openai':
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI client not available. Install with: pip install openai")
        return OpenAIClient(**kwargs)
    elif provider == 'gemini':
        if not GEMINI_AVAILABLE:
            raise ValueError("Gemini client not available. Install with: pip install google-generativeai")
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    'BaseClient',
    'OllamaClient',
    'AnthropicClient',
    'OpenAIClient',
    'GeminiClient',
    'get_available_providers',
    'create_client',
    'ANTHROPIC_AVAILABLE',
    'OPENAI_AVAILABLE',
    'GEMINI_AVAILABLE',
]
