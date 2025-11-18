# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intermediated Dialog (IDi) is a three-AI dialog system where one AI acts as an intermediator/moderator facilitating a conversation between two participant AIs. It's a Flask-based web application with real-time streaming via WebSockets.

## Development Commands

### Starting the Application

```bash
# Using the startup script (recommended)
./run_intermediator_dialog.sh

# Or with custom host/port
./run_intermediator_dialog.sh 0.0.0.0 5006

# Or directly with Python
python intermediator_dialog.py --host 0.0.0.0 --port 5006
```

The startup script handles:
- Killing existing instances on the same port
- Activating the parent project's venv (../venv) if it exists
- Opening the browser automatically
- Running the server in the background

### Dependencies

```bash
# Install dependencies (use parent venv or create new one)
source ../venv/bin/activate  # If using parent project
pip install -r requirements.txt
```

Required packages:
- Flask (web framework)
- Flask-SocketIO (WebSocket support for real-time streaming)
- Requests (HTTP communication with Ollama servers)
- ReportLab (PDF generation)
- Python-dotenv (environment variable support)

### Running Tests

There are currently no automated tests in this project.

### Environment Setup

Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

This is required for the Text-to-Speech (TTS) feature.

## Architecture

### Core Components

**OllamaClient** (intermediator_dialog.py:34-205)
- Manages communication with individual Ollama servers
- Handles streaming responses with token counting
- Maintains conversation history (message cache) per client instance
- Supports extensive model parameters (temperature, top_p, top_k, repeat_penalty, etc.)
- Key feature: Client instances are cached by `host:model:name` to preserve conversation state across multiple dialogs

**IntermediatorDialog** (intermediator_dialog.py:207-637)
- Orchestrates the three-way dialog between intermediator and two participants
- Manages turn-taking logic (alternating participants with intermediator moderation)
- Builds and maintains separate conversation contexts for each AI
- Supports flexible prompt configuration:
  - Intermediator: pre_prompt + topic_prompt (topic is required)
  - Participants: pre_prompt + mid_prompt (personalization) + post_prompt
- Handles context file injection into all three AIs' conversation history
- Emits real-time events via callbacks for streaming to web interface

**Flask Application** (intermediator_dialog.py:614-1596)
- WebSocket-based architecture using Flask-SocketIO
- Key endpoints:
  - `/` - Main web interface
  - `/upload` - File upload for context documents
  - `/generate_pdf/<dialog_id>` - Generate PDF from saved dialog
  - `/check_servers` - Verify Ollama server availability
- Key SocketIO events:
  - `start_dialog` - Initiates a new dialog session
  - `reset_cache` - Clears all cached client instances

### Conversation Flow

1. **Initialization**: Intermediator introduces topic from its system prompt
2. **Turn Alternation**: Participants alternate turns (Participant 1 → Intermediator moderation → Participant 2 → Intermediator moderation)
3. **Context Passing**: Each participant sees recent conversation history (last 3 messages)
4. **Streaming**: All responses stream in real-time to the web interface via WebSocket callbacks
5. **Finalization**: Intermediator provides final summary after max_turns completed

### Client Caching System

The application maintains a global `client_instances` dictionary that caches OllamaClient instances by `host:model:name`. This is critical for:
- Preserving conversation history across multiple dialog sessions
- Maintaining context when running sequential dialogs
- Avoiding redundant server connections

To reset the cache: Use the "Reset Cache" button in the web UI or the `reset_cache` SocketIO event.

### Output Files

Dialogs are saved to `output/` directory in two formats:
- **JSON**: Complete dialog data including metadata, all messages, and token counts
- **PDF**: Formatted document with conversation flow and statistics

Filename format: `{sanitized_topic}_{timestamp}.{json|pdf}`

### Default Server Configuration

- **Intermediator**: RT5090 at http://192.168.6.40:11434 (model: gpt-oss:20b)
- **Participant 1**: NVIDIA DGX Spark 1 at http://192.168.5.40:11434 (model: gpt-oss:20b)
- **Participant 2**: NVIDIA DGX Spark 2 at http://192.168.5.46:11434 (model: gpt-oss:120b)

All configurations are customizable via the web interface.

## Key Implementation Details

### Markdown Suppression
All system prompts automatically append instructions to avoid markdown formatting. This ensures clean plain-text output for better readability in the dialog interface.

### Token Tracking
Each response includes comprehensive metrics:
- Prompt tokens, completion tokens, total tokens
- Tokens per second (throughput)
- Time to first token (TTFT)

These are tracked both per-response and cumulatively per client.

### Thinking Parameter
The application supports an optional "thinking" parameter that can be enabled per client, allowing models that support reasoning modes to use extended inference.

### Context File Support
Users can upload context files (text documents) that are injected into all three AIs' conversation history before the dialog begins. This provides shared background information for the discussion.

### PDF Generation
The `generate_pdf_from_dialog()` function (intermediator_dialog.py:886) creates formatted PDFs with:
- Conversation header with metadata
- Server configuration table
- Turn-by-turn dialog with speaker identification
- Token usage statistics
- Automatic page breaks and text wrapping
- Special handling for landscape mode when 3+ servers are involved

### Text-to-Speech (TTS) Feature

The application automatically generates audio narration for all responses using OpenAI's TTS API:

**Voice Assignments:**
- **Intermediator**: 'alloy' voice (neutral, balanced)
- **Participant 1**: 'echo' voice (clear, articulate)
- **Participant 2**: 'fable' voice (warm, expressive)

**Audio Files:**
- Saved to `output/audio/{dialog_id}/` directory
- Format: `{speaker}_turn{number}.mp3`
- Generated automatically after each response
- Includes intro, all participant responses, intermediator moderations, and final summary

**Turn Logic:**
When `max_turns` is set to N, each participant will speak N times (total of 2N participant turns), with intermediator moderation after each turn. For example:
- max_turns=4 means Participant 1 speaks 4 times, Participant 2 speaks 4 times (8 total participant turns)
- Plus intermediator intro and moderation after each response
- Plus final summary

The `generate_tts_audio()` function (intermediator_dialog.py:642) handles all TTS generation and file management.
