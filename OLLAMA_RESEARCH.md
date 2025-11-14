# Ollama API Capabilities Research

## Current Implementation Analysis

### What We're Using
- **Endpoint**: `/api/chat` (line 101 in `five_whys_parallel.py`)
- **Method**: Sending full `messages` array with every request (line 116)
- **Context Management**: No session persistence - each request is independent
- **Model Loading**: Models are loaded/unloaded automatically per request

### Performance Issues Identified

1. **Time to First Token (TTFT) is slow** because:
   - Ollama reprocesses the entire conversation history on every request
   - Models may be unloaded between requests (default `keep_alive: 5m`)
   - Full message array must be tokenized and processed each time

2. **Throughput is reduced** because:
   - Redundant processing of previous messages
   - No KV cache reuse between requests
   - Model reload overhead if `keep_alive` expires

3. **Memory inefficiency** for long conversations:
   - Context file + all conversation rounds sent every time
   - No truncation or summarization of old messages

---

## Ollama API Capabilities

### 1. `keep_alive` Parameter ⭐ **HIGH IMPACT**

**Available for**: Both `/api/generate` and `/api/chat`

**What it does**: Controls how long the model stays loaded in memory after a request completes.

**Default**: `5m` (5 minutes)

**Options**:
- Duration string: `"5m"`, `"10m"`, `"1h"`, etc.
- `"0"` or `0`: Immediately unload the model
- Negative number: Keep loaded indefinitely (until server restart)

**Example**:
```python
payload = {
    "model": self.model,
    "messages": self.messages,
    "stream": True,
    "keep_alive": "10m"  # Keep model loaded for 10 minutes
}
```

**Impact**: 
- **Eliminates model reload time** between requests
- **Reduces `load_duration` to near zero** for subsequent requests
- **Significantly improves TTFT** when model is already loaded

**Recommendation**: Set `keep_alive: "10m"` or longer for active analysis sessions.

---

### 2. Context Parameter (Deprecated for `/api/chat`)

**For `/api/generate` endpoint only**:
- Returns a `context` array in the response
- Can be passed back in the next request to maintain KV cache
- **Status**: Deprecated - not recommended for new code

**For `/api/chat` endpoint**:
- **No `context` parameter** - uses `messages` array instead
- The `messages` array IS the context management mechanism
- Ollama internally manages KV cache based on the messages array

**Current approach is correct**: Using `/api/chat` with `messages` array is the recommended modern approach.

---

### 3. Model Pre-loading

**Load a model** (without generating):
```python
# Empty prompt loads the model
payload = {
    "model": "llama3.2",
    "prompt": ""  # Empty prompt
}
```

**Unload a model**:
```python
payload = {
    "model": "llama3.2",
    "keep_alive": 0  # Unload immediately
}
```

**Use case**: Pre-load models at application startup to eliminate first-request latency.

---

### 4. Performance Metrics Available

Ollama returns detailed timing information in responses:

```json
{
  "total_duration": 10706818083,      // Total time (nanoseconds)
  "load_duration": 6338219291,        // Time loading model
  "prompt_eval_count": 26,            // Tokens in prompt
  "prompt_eval_duration": 130079000,  // Time evaluating prompt
  "eval_count": 259,                  // Tokens generated
  "eval_duration": 4232710000         // Time generating response
}
```

**Calculations**:
- **Tokens/second**: `eval_count / eval_duration * 10^9`
- **Prompt processing time**: `prompt_eval_duration`
- **Model load overhead**: `load_duration`

**Current code**: We're already capturing `prompt_tokens` and `completion_tokens`, but not the duration metrics.

---

### 5. Conversation History Management

**Current behavior**:
- We send the full `messages` array every time
- Ollama processes all messages from scratch each request
- This is the **standard approach** for chat endpoints

**Optimization options**:

1. **Message truncation** (if conversation gets very long):
   - Keep only last N messages
   - Summarize early messages
   - Use sliding window approach

2. **Context window management**:
   - Monitor `prompt_eval_count` vs `num_ctx`
   - Truncate if approaching context limit

3. **Smart context selection**:
   - Keep system prompt + recent messages
   - Remove intermediate messages if needed

**Note**: Ollama internally manages KV cache, but still needs to process the full message array to build it.

---

### 6. Streaming

**Current implementation**: ✅ Already using streaming (`"stream": True`)

**Benefits**:
- Lower latency (TTFT)
- Better user experience (progressive display)
- No changes needed here

---

## Recommended Optimizations

### Priority 1: Add `keep_alive` Parameter ⭐⭐⭐

**Impact**: High - Eliminates model reload overhead

**Implementation**:
```python
payload = {
    "model": self.model,
    "messages": self.messages,
    "stream": True,
    "keep_alive": "10m"  # Keep loaded during active session
}
```

**When to use**:
- During active analysis (keep model loaded)
- After analysis completes, can set to `0` to free memory

---

### Priority 2: Pre-load Models at Startup ⭐⭐

**Impact**: Medium - Eliminates first-request latency

**Implementation**:
- On server startup, send empty requests to pre-load models
- Or use `keep_alive: -1` to keep models loaded indefinitely

**Trade-off**: Uses more memory, but faster first response

---

### Priority 3: Add Performance Metrics Logging ⭐

**Impact**: Low - Better visibility into bottlenecks

**Implementation**:
- Capture `load_duration`, `prompt_eval_duration`, `eval_duration` from responses
- Log these metrics to identify bottlenecks
- Display in UI or debug logs

---

### Priority 4: Smart Message Truncation ⭐

**Impact**: Medium - Prevents context window overflow

**Implementation**:
- Monitor message count and total token count
- If approaching `num_ctx` limit, truncate older messages
- Keep system prompt + recent N messages

**When needed**: Only for very long conversations (10+ rounds with large context files)

---

## Limitations & Notes

1. **No true session persistence**: Ollama doesn't maintain sessions between requests. Each request is independent.

2. **Full message reprocessing**: Even with `keep_alive`, Ollama still processes the full `messages` array each time. The KV cache helps, but prompt evaluation still occurs.

3. **Context parameter deprecated**: The `context` parameter for `/api/generate` is deprecated. Using `/api/chat` with `messages` is the modern approach.

4. **Model memory**: Keeping models loaded (`keep_alive`) uses GPU/system memory. Balance between performance and resource usage.

5. **No built-in summarization**: Ollama doesn't automatically summarize old messages. This would need to be implemented client-side if needed.

---

## Summary

**Key Finding**: The biggest performance win is adding `keep_alive` parameter to keep models loaded in memory, eliminating the model reload overhead between requests.

**Current approach is correct**: Using `/api/chat` with `messages` array is the recommended way to handle conversations.

**Main optimization opportunity**: 
- ✅ Add `keep_alive: "10m"` to requests
- ✅ Consider pre-loading models at startup
- ✅ Add performance metrics logging
- ⚠️ Message truncation only needed for very long conversations

**Expected improvements**:
- **TTFT**: 30-50% faster (eliminates model load time)
- **Throughput**: 10-20% faster (no reload overhead)
- **User experience**: More consistent response times
