# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intermediated Dialog (IDi) is a three-AI dialog system where one AI acts as an intermediator/moderator facilitating a conversation between two participant AIs. It's a Flask-based web application with real-time streaming via WebSockets.

## Development Commands

### Starting the Application

```bash
# Using the startup script (recommended) - starts both main app and audio player
./run_intermediator_dialog.sh

# Or with custom host/port
./run_intermediator_dialog.sh 0.0.0.0 5005

# Or directly with Python (main app only)
python app.py --host 0.0.0.0 --port 5005
```

The startup script handles:
- Killing existing instances on the configured ports
- Activating the venv (.venv, venv, or ../venv)
- Starting the audio player server on port 5002
- Opening the browser automatically
- Running both servers in the background with proper cleanup on Ctrl+C

### Running Tests

```bash
python -m pytest tests/
python -m pytest tests/test_ollama_client.py  # Single test file
```

### Environment Setup

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Required for the Text-to-Speech (TTS) feature.

## Architecture

### Module Structure

The codebase uses a modular architecture:

```
app.py                              # Flask entry point, initializes SocketIO
├── config.py                       # Global state dictionaries and configuration
├── server_config.json              # Single source of truth for server/service config
├── routes.py                       # HTTP endpoints (/, /generate_pdf, /check_servers, etc.)
├── socketio_handlers.py            # WebSocket event handlers (start_dialog, reset_cache)
├── intermediator_dialog_refactored.py  # Refactored dialog with phase-aware prompts
├── prompt_templates.py             # PromptTemplates class with DialogMode/DialogPhase enums
├── clients/
│   ├── base_client.py              # Abstract BaseClient with TokenInfo, retry logic, error hierarchy
│   ├── ollama_client.py            # OllamaClient implementation
│   ├── anthropic_client.py         # AnthropicClient for Claude API
│   ├── openai_client.py            # OpenAIClient for OpenAI API
│   └── gemini_client.py            # GeminiClient for Google Gemini
├── tts.py                          # OpenAI TTS generation and participant summaries
├── pdf_generator.py                # ReportLab-based PDF generation
├── gpu_monitor.py                  # Optional GPU status polling
└── utils.py                        # Helper functions (save_dialog_to_files, debug_log)
```

### Configuration

All server addresses, service ports, and defaults are in `server_config.json`:
- `servers` - Ollama server hosts, names, providers, default models
- `services` - diagram service URL, GPU monitor port, audio player port, app port
- `defaults` - default max_turns
- `tts` - TTS model and voice mapping

### Core Classes

**BaseClient** (`clients/base_client.py`)
- Abstract base class for LLM providers
- Defines `ask()`, `check_server_available()`, `reset_conversation()` methods
- Tracks conversation history and token counts
- Retry logic with exponential backoff via `RetryConfig` and `_retry_with_backoff()`
- Typed error hierarchy: `ClientError`, `ClientConnectionError`, `ClientAuthError`, `ClientRateLimitError`, `ClientTimeoutError`

**OllamaClient** (`clients/ollama_client.py`)
- Implements BaseClient for Ollama servers
- Handles streaming responses with token metrics (TTFT, tokens/sec)
- Supports model parameters: temperature, top_p, top_k, repeat_penalty, num_predict, thinking, be_brief

**IntermediatorDialogRefactored** (`intermediator_dialog_refactored.py`)
- Uses DialogConfig dataclass for configuration
- Supports four dialog modes: DEBATE, EXPLORATION, INTERVIEW, CRITIQUE
- Implements draft -> critique -> final response flow for participants
- Phase-aware prompting (EARLY, MIDDLE, LATE phases affect moderation style)
- Consolidated `_handle_participant_turn()` method replaces duplicate code
- Pause/resume/stop support via threading.Event

**PromptTemplates** (`prompt_templates.py`)
- Centralized prompt management
- Mode-specific system prompts for moderator and participants
- Phase-aware moderation prompts with randomized variety
- Summary prompts tailored to dialog mode

### Dialog Flow

1. **Initialization**: System prompts set per mode, models preloaded in parallel
2. **Introduction**: Moderator introduces topic (mode-specific intro prompt)
3. **Turn Loop**:
   - Participant responds using draft -> self-critique -> final response pipeline
   - Moderator intervenes with phase-aware prompt (can signal early conclusion via `[CONCLUDE]`)
   - Context built with priority weighting (recent moderator/opponent messages prioritized)
4. **Summary**: Mode-specific summary prompt, declares winner for debates

### Client Caching

Global `client_instances` dictionary in `config.py` caches client instances by `{host}:{model}:{name}` (Ollama) or `{provider}:{model}:{name}` (cloud). This preserves conversation history across sequential dialogs. Reset via "Reset Cache" button or `reset_dialog_cache` SocketIO event. Access is thread-safe via locking.

### Output Files

Saved to `output/` directory:
- **JSON**: Complete dialog data with metadata and token counts
- **TXT**: Readable transcript with formatting
- **PDF**: Formatted report with participant table, token stats, GPU energy (if available)

Audio files saved to `output/audio/Debate_{topic}/`:
- TTS audio: `{sequence}_{speaker}.mp3` (with sentence-boundary chunking for long text)
- Argument summaries: `summary_{participant_name}.txt`
- Diagrams: `diagram_{participant_name}.png` (requires diagram service)

### Key SocketIO Events

**Client -> Server:**
- `start_dialog`: Initiates dialog with server configs, prompt_config, thinking_params
- `reset_dialog_cache`: Clears conversation cache
- `pause_dialog`, `resume_dialog`, `stop_dialog`: Dialog flow control

**Server -> Client:**
- `dialog_update`: Streaming content chunks
- `dialog_started`, `dialog_complete`: Lifecycle events
- `participant_draft_start/complete`, `participant_critique_start/complete`, `participant_final_response_start/complete`: Draft-critique-final phases
- `pdf_generated`, `summaries_generated`: Post-dialog artifacts
- `gpu_status_update`: Real-time GPU metrics
- `cost_update`: Running cost tracking for cloud API dialogs

## Key Implementation Details

### Draft-Critique-Final Response Pipeline

Participants generate responses in three phases (see `_handle_participant_turn()` in refactored dialog):
1. **Draft**: Initial response generation
2. **Critique**: Self-review for clarity, logic, tone
3. **Final**: Refined response incorporating critique

### Dialog Modes

- **DEBATE**: Adversarial, winner declared in summary
- **EXPLORATION**: Collaborative inquiry, synthesis-focused summary
- **INTERVIEW**: One participant questions, other responds
- **CRITIQUE**: One presents, other critiques

### Phase-Aware Prompting

Dialog divided into EARLY (30%), MIDDLE (40%), LATE (30%) phases. Moderation prompts adapt:
- EARLY: Establish positions, clarify framing
- MIDDLE: Probe arguments, find crux of disagreement
- LATE: Focus on resolution, prepare for summary

### Markdown Suppression

All system prompts include: "IMPORTANT: Use plain text only. No markdown formatting."

### External Services

All service addresses are configured in `server_config.json`:
- **Diagram service**: Generates argument structure diagrams (optional)
- **GPU monitoring**: On each Ollama server (optional)
- **Audio player**: Separate Flask app in `audio_player/` subdirectory

### Turn Logic

`max_turns` parameter means each participant speaks that many times. With `max_turns=4`:
- Participant 1 speaks 4 times, Participant 2 speaks 4 times (8 total participant turns)
- Moderator intervenes after each participant turn (7 moderation comments)
- Plus intro and final summary
