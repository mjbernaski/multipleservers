# Multi-Client Refactoring Plan

## Overview
Plan to restructure the codebase to support multiple LLM providers (Ollama, Claude, OpenAI, Gemini) while maintaining current functionality.

## Proposed Module Structure

```
intermediator_dialog/
├── __init__.py
├── clients/                    # NEW: Client abstraction layer
│   ├── __init__.py
│   ├── base_client.py         # BaseClient abstract class (interface)
│   ├── ollama_client.py       # OllamaClient (refactored from current code)
│   ├── claude_client.py       # ClaudeClient (Anthropic API)
│   ├── openai_client.py       # OpenAIClient (OpenAI API)
│   ├── gemini_client.py       # GeminiClient (Google API)
│   └── client_factory.py      # Factory to create appropriate client
├── models.py                  # IntermediatorDialog class (~300 lines)
├── utils.py                   # Helper functions (~100 lines)
├── tts.py                     # TTS generation (~100 lines)
├── pdf_generator.py           # PDF generation (~350 lines)
├── gpu_monitor.py             # GPU monitoring (Ollama-only) (~150 lines)
├── config.py                  # Global state, API keys, configuration (~50 lines)
├── routes.py                  # Flask HTTP routes (~200 lines)
├── socketio_handlers.py       # WebSocket event handlers (~300 lines)
└── app.py                     # Main entry point (~100 lines)
```

## Key Architectural Changes

### 1. BaseClient Abstract Class

All clients will implement this interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable, Optional

class BaseClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, model: str, name: str, **kwargs):
        self.model = model
        self.name = name
        self.messages: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.stream_callback: Optional[Callable] = None

    @abstractmethod
    def ask(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
        """
        Send a question to the LLM and get response with token counts.

        Returns:
            Tuple of (response_text, token_info_dict)
        """
        pass

    @abstractmethod
    def check_server_available(self) -> bool:
        """Check if the service is available."""
        pass

    @abstractmethod
    def reset_conversation(self) -> None:
        """Reset conversation history and token counts."""
        pass

    def set_stream_callback(self, callback: Callable) -> None:
        """Set streaming callback (optional for non-streaming APIs)."""
        self.stream_callback = callback
```

### 2. Client Factory

```python
class ClientFactory:
    """Factory to create appropriate LLM client based on type."""

    @staticmethod
    def create(client_type: str, model: str, name: str, **kwargs) -> BaseClient:
        """
        Create a client instance.

        Args:
            client_type: 'ollama', 'claude', 'openai', or 'gemini'
            model: Model identifier
            name: Display name for the client
            **kwargs: Type-specific parameters
        """
        if client_type == 'ollama':
            return OllamaClient(
                host=kwargs['host'],
                model=model,
                name=name,
                num_ctx=kwargs.get('num_ctx', 8192),
                temperature=kwargs.get('temperature'),
                top_p=kwargs.get('top_p'),
                top_k=kwargs.get('top_k'),
                repeat_penalty=kwargs.get('repeat_penalty'),
                num_predict=kwargs.get('num_predict'),
                thinking=kwargs.get('thinking', False),
                be_brief=kwargs.get('be_brief', False)
            )
        elif client_type == 'claude':
            return ClaudeClient(
                model=model,
                name=name,
                api_key=kwargs['api_key'],
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens', 4096),
                thinking=kwargs.get('thinking', False)
            )
        elif client_type == 'openai':
            return OpenAIClient(
                model=model,
                name=name,
                api_key=kwargs['api_key'],
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens', 4096)
            )
        elif client_type == 'gemini':
            return GeminiClient(
                model=model,
                name=name,
                api_key=kwargs['api_key'],
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens', 4096)
            )
        else:
            raise ValueError(f"Unknown client type: {client_type}")
```

### 3. Configuration Format

Frontend configuration for each participant:

```python
{
    'type': 'claude',              # or 'ollama', 'openai', 'gemini'
    'model': 'claude-sonnet-4.5',  # Model identifier
    'name': 'Claude Moderator',    # Display name

    # Type-specific fields
    'api_key': 'sk-...',           # For API clients (claude, openai, gemini)
    'host': 'http://...',          # For Ollama only

    # Common model parameters (abstracted across providers)
    'temperature': 0.7,
    'max_tokens': 4096,
    'thinking': False,
    'be_brief': False
}
```

### 4. Client-Specific Implementation Notes

#### OllamaClient
- Keeps current implementation
- GPU monitoring remains Ollama-specific
- Model preloading support
- Streaming via requests library
- Custom parameters: num_ctx, top_p, top_k, repeat_penalty

#### ClaudeClient
- Uses Anthropic SDK
- Streaming via SDK's streaming API
- Supports extended thinking mode
- 200K context window
- Temperature: 0.0-1.0

#### OpenAIClient
- Uses OpenAI SDK
- Streaming via SDK
- Supports o1 thinking models
- Token counting from response
- Temperature: 0.0-2.0

#### GeminiClient
- Uses Google Generative AI SDK
- Streaming support
- Token counting from response
- Safety settings configuration

### 5. Feature Matrix

| Feature | Ollama | Claude | OpenAI | Gemini |
|---------|--------|--------|--------|--------|
| Streaming | ✅ | ✅ | ✅ | ✅ |
| GPU Monitoring | ✅ | ❌ | ❌ | ❌ |
| Model Preloading | ✅ | ❌ | ❌ | ❌ |
| Token Counting | ✅ | ✅ | ✅ | ✅ |
| System Messages | ✅ | ✅ | ✅ | ✅ |
| Context Window | Custom | 200K | Varies | 1M |
| Thinking Mode | ✅ | ✅ (Extended) | ✅ (o1) | ❌ |
| Keep-Alive | ✅ | ❌ | ❌ | ❌ |
| Temperature | ✅ | ✅ | ✅ | ✅ |
| Top-P | ✅ | ✅ | ✅ | ✅ |
| Top-K | ✅ | ❌ | ❌ | ✅ |

### 6. Client Caching Changes

Current caching key format:
```python
# Old: host:model:name
key = f"{host}:{model}:{name}"
```

New caching key format:
```python
# New: type:model:name
key = f"{client_type}:{model}:{name}"
```

This ensures separate cache entries for different providers even if model names overlap.

### 7. Frontend UI Changes

**Server Configuration Section:**
```
┌─────────────────────────────────────┐
│ Provider Type: [Ollama ▼]          │
│ Model: [gpt-oss:20b]                │
│ Name: [RT5090]                      │
│ Host: [http://localhost:11434]      │  ← Only for Ollama
│ API Key: [••••••••••••]             │  ← Only for APIs
│ Temperature: [0.7]                   │
│ Max Tokens: [4096]                   │
└─────────────────────────────────────┘
```

**Model Selection:**
- Dynamically filter available models based on provider type
- Ollama: Fetch from /api/tags
- Claude: Show preset list (sonnet-4.5, opus-4, haiku-4)
- OpenAI: Show preset list (gpt-4-turbo, gpt-4o, o1-preview)
- Gemini: Show preset list (gemini-2.0-flash, gemini-2.0-pro)

### 8. Environment Variables

Add to `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### 9. Dependencies to Add

```
anthropic>=0.40.0      # Claude API
openai>=1.50.0         # OpenAI API
google-generativeai    # Gemini API
```

## Two-Phase Implementation Approach

### Phase 1: Basic Module Split (Do Now)
1. Split existing code into modules:
   - Extract `OllamaClient` to `clients/ollama_client.py`
   - Create `models.py` with `IntermediatorDialog`
   - Extract utilities to `utils.py`
   - Extract TTS to `tts.py`
   - Extract PDF generation to `pdf_generator.py`
   - Extract GPU monitoring to `gpu_monitor.py`
   - Create `config.py` for global state
   - Split routes to `routes.py`
   - Split WebSocket handlers to `socketio_handlers.py`
   - Create main `app.py` entry point

2. Create `BaseClient` abstract class in `clients/base_client.py`

3. Make `OllamaClient` inherit from `BaseClient`

4. Design for extensibility but keep Ollama-only functionality

5. Test thoroughly with existing Ollama setup

### Phase 2: Add API Clients (Later)
1. Implement `ClaudeClient` with Anthropic SDK
2. Implement `OpenAIClient` with OpenAI SDK
3. Implement `GeminiClient` with Google SDK
4. Create `ClientFactory`
5. Update frontend UI for multi-provider selection
6. Add API key management
7. Update configuration handling
8. Test mixed configurations:
   - Ollama + Claude + GPT
   - All Ollama (regression test)
   - All Cloud APIs
9. Update documentation

## Testing Strategy

### Phase 1 Testing
- All existing Ollama debates should work exactly as before
- No functionality regression
- Module imports work correctly
- Client caching still functions

### Phase 2 Testing
- Test each provider independently
- Test mixed provider configurations
- Verify token counting across all providers
- Verify streaming works for all providers
- Test error handling for API failures
- Test API key validation

## Backward Compatibility

- Existing saved dialogs (JSON) should still work
- PDF generation should work with all client types
- TTS should work regardless of client
- GPU monitoring gracefully disabled for non-Ollama

## Migration Notes

### For Users
- No action required for Ollama-only users
- New users can use API providers without running Ollama
- Mixed mode allows using both local and cloud models

### For Code
- Old client instances in cache will need migration on first run
- Configuration format change may require updating saved presets

## Benefits of This Approach

1. **Flexibility**: Use any combination of providers
2. **Cost Optimization**: Use cheap local models where possible, expensive cloud models where needed
3. **Model Diversity**: Test different providers side-by-side
4. **Fallback Options**: If one provider is down, switch to another
5. **Future-Proof**: Easy to add new providers (Cohere, Mistral, etc.)
6. **Cleaner Code**: Each client isolated in its own module
7. **Better Testing**: Mock clients for unit tests

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Phase 1 focuses only on refactoring, no new features |
| API rate limits | Implement retry logic with exponential backoff |
| API costs | Add usage tracking and warnings |
| Different token counting methods | Normalize token info format across providers |
| Streaming inconsistencies | Abstract streaming behind common interface |

## Timeline Estimate

- **Phase 1** (Module Split): 2-3 hours
- **Phase 2** (API Clients): 4-6 hours
- **Testing & Debug**: 2-3 hours
- **Total**: 8-12 hours

## Open Questions

1. Should we support conversation import/export between providers?
2. How to handle provider-specific features (e.g., Claude artifacts)?
3. Should we add cost tracking for API calls?
4. Cache strategy for API responses (to save costs)?
5. Should we support local API-compatible servers (like vLLM)?

---

**Status**: Plan documented, pending bug fixes before implementation.
**Next Step**: Fix current bugs, then proceed with Phase 1.
