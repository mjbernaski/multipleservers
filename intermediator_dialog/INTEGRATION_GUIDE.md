# Integration Guide: Refactored Intermediator Dialog System

## Overview

This guide explains how to integrate the refactored components into your existing `intermediator_dialog.py`.

## New Files

1. **`prompt_templates.py`** - Centralized prompt management with:
   - Dialog modes (DEBATE, EXPLORATION, INTERVIEW, CRITIQUE)
   - Phase-aware prompts (EARLY, MIDDLE, LATE)
   - Randomized moderation prompts for variety
   - All prompts extracted and configurable

2. **`intermediator_dialog_refactored.py`** - Clean dialog implementation with:
   - Consolidated participant handling (no duplicate code)
   - Better context management
   - Integration with PromptTemplates
   - Early conclusion detection

## Quick Integration (Minimal Changes)

If you want to keep your existing structure but use the new prompts:

```python
# At the top of intermediator_dialog.py, add:
from prompt_templates import PromptTemplates, DialogMode, DialogPhase

# In IntermediatorDialog.__init__, add:
self.prompts = PromptTemplates(mode=DialogMode.DEBATE)  # or EXPLORATION

# Replace the hardcoded prompts with template calls (examples below)
```

## Key Changes to Make

### 1. Remove "Be Brief" Prepending (Line 130-131)

**Before:**
```python
def ask(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
    if self.be_brief:
        question = "Be Brief. " + question
```

**After:**
```python
def ask(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
    # Removed: "Be Brief" prepending - let num_predict control length instead
```

### 2. Replace Moderator System Prompt (Lines 321-342)

**Before:**
```python
default_pre = """You are a thoughtful moderator facilitating a dialog between two AI participants. Your role is to:
1. Guide the conversation to explore the topic deeply
2. Ask clarifying questions when needed
...
```

**After:**
```python
# In run_dialog method:
intermediator_system_prompt = self.prompts.get_moderator_system_prompt(
    topic=self.intermediator_topic_prompt,
    custom_instructions=self.intermediator_pre_prompt
)
```

### 3. Replace Participant System Prompts (Lines 356-366)

**Before:**
```python
def build_participant_prompt(mid_prompt: str = None) -> str:
    # ... complex string building logic
    default_prompt = """You are participating in a moderated dialog. Another AI will moderate...
```

**After:**
```python
# For participant 1:
p1_prompt = self.prompts.get_participant_system_prompt(
    participant_name=self.participant1.name,
    position=self.participant1_mid_prompt,  # Their position/role
    custom_instructions=self.participant_pre_prompt
)

# For participant 2:
p2_prompt = self.prompts.get_participant_system_prompt(
    participant_name=self.participant2.name,
    position=self.participant2_mid_prompt,
    custom_instructions=self.participant_pre_prompt
)
```

### 4. Replace Introduction Prompt (Line 412)

**Before:**
```python
intro_prompt = """Please introduce the topic to the two participants and start the conversation. Address both participants and set the stage for a productive discussion."""
```

**After:**
```python
intro_prompt = self.prompts.get_intro_prompt(
    participant1_name=self.participant1.name,
    participant2_name=self.participant2.name
)
```

### 5. Replace Moderation Prompts (Lines 523-532, 638-647)

**Before:**
```python
mod_prompt = f"""Participant 1 just said: "{p1_response[:400]}"

You have access to the entire conversation history through your message context. As the moderator, please:
1. Ensure the conversation stays focused on the topic
...
```

**After:**
```python
# Calculate total turns for phase awareness
total_turns = max_turns * 2

mod_prompt = self.prompts.get_moderation_prompt(
    speaker_name=self.participant1.name,  # Use actual name
    turn=turn,
    total_turns=total_turns
)
```

### 6. Replace Participant Turn Prompts (Lines 469-475, 584-590)

**Before:**
```python
p1_prompt = f"""Here's what has been said so far in our discussion:

{context_summary}

The {last_speaker} just said: "{last_message[:400]}"

Please respond thoughtfully. Engage with the points raised and contribute your perspective."""
```

**After:**
```python
total_turns = max_turns * 2

p1_prompt = self.prompts.get_participant_turn_prompt(
    context_summary=context_summary,
    last_message=last_message,  # Don't truncate here - template handles it
    last_speaker_name=self.names[last_speaker],  # Use actual name
    turn=turn,
    total_turns=total_turns
)
```

### 7. Replace Summary Prompt (Lines 691-709)

**Before:**
```python
summary_prompt = """The dialog has now concluded. Please provide a comprehensive summary and wrap-up of the entire conversation. 
...
```

**After:**
```python
summary_prompt = self.prompts.get_summary_prompt(
    participant1_name=self.participant1.name,
    participant2_name=self.participant2.name
)
```

### 8. Consolidate Participant Handling (Lines 458-686)

The original code has nearly identical blocks for participant 1 and participant 2. See `intermediator_dialog_refactored.py` for how to consolidate this into a single `_handle_participant_turn()` method.

**Key pattern:**
```python
def _handle_participant_turn(self, participant_num: int, turn: int, total_turns: int):
    """Handle a single participant's turn."""
    participant = self.participant1 if participant_num == 1 else self.participant2
    speaker_key = f'participant{participant_num}'
    # ... rest of logic is identical for both
```

### 9. Add Dialog Mode Support

In your `handle_start_dialog` function, add mode configuration:

```python
# In handle_start_dialog (around line 2194)
dialog_mode = prompt_config.get('dialog_mode', 'exploration')
mode_map = {
    'debate': DialogMode.DEBATE,
    'exploration': DialogMode.EXPLORATION,
    'interview': DialogMode.INTERVIEW,
    'critique': DialogMode.CRITIQUE
}
selected_mode = mode_map.get(dialog_mode, DialogMode.EXPLORATION)

# Pass to dialog config
config = DialogConfig(
    mode=selected_mode,
    max_turns=max_turns,
    participant1_position=prompt_config.get('participant1_position'),
    participant2_position=prompt_config.get('participant2_position'),
)
```

## Full Replacement Option

If you prefer a clean slate, you can:

1. Keep `OllamaClient` class unchanged (it works well)
2. Replace `IntermediatorDialog` class entirely with `IntermediatorDialogRefactored`
3. Update references in `run_dialog_thread` and `handle_start_dialog`

```python
# Import the new classes
from prompt_templates import PromptTemplates, DialogMode, DialogConfig
from intermediator_dialog_refactored import IntermediatorDialogRefactored

# In run_dialog_thread, replace:
dialog = IntermediatorDialog(...)

# With:
config = DialogConfig(
    mode=DialogMode.DEBATE,
    max_turns=max_turns,
    enable_tts=enable_tts,
    participant1_position=prompt_config.get('participant1_position'),
    participant2_position=prompt_config.get('participant2_position'),
)

dialog = IntermediatorDialogRefactored(
    intermediator=intermediator_client,
    participant1=participant1_client,
    participant2=participant2_client,
    topic=prompt_config.get('intermediator_topic_prompt', ''),
    config=config,
    dialog_id=dialog_id,
    tts_callback=generate_tts_audio  # Pass your existing TTS function
)
```

## Frontend Updates (Optional)

To support dialog modes in your frontend:

```javascript
// Add a mode selector to your UI
const dialogModes = [
    { value: 'debate', label: 'Debate (adversarial, winner declared)' },
    { value: 'exploration', label: 'Exploration (collaborative inquiry)' },
    { value: 'interview', label: 'Interview (one questions the other)' },
    { value: 'critique', label: 'Critique (one presents, other critiques)' }
];

// Add position inputs for debate mode
<input name="participant1_position" placeholder="Position for Participant 1" />
<input name="participant2_position" placeholder="Position for Participant 2" />
```

## Testing the Changes

1. **Test prompt generation:**
```python
from prompt_templates import PromptTemplates, DialogMode

# Create templates
templates = PromptTemplates(mode=DialogMode.DEBATE)

# Test each prompt type
print(templates.get_moderator_system_prompt(topic="AI ethics"))
print(templates.get_intro_prompt("Alice", "Bob"))
print(templates.get_moderation_prompt("Alice", turn=3, total_turns=6))
print(templates.get_summary_prompt("Alice", "Bob"))
```

2. **Test phase detection:**
```python
templates = PromptTemplates(mode=DialogMode.DEBATE)

# Should return EARLY
assert templates.get_phase(1, 10) == DialogPhase.EARLY

# Should return MIDDLE  
assert templates.get_phase(5, 10) == DialogPhase.MIDDLE

# Should return LATE
assert templates.get_phase(9, 10) == DialogPhase.LATE
```

3. **Run a test dialog:**
```python
# Use your existing test setup but with the new config
config = DialogConfig(
    mode=DialogMode.EXPLORATION,
    max_turns=2,  # Short test
    enable_tts=False
)
```

## Summary of Benefits

| Before | After |
|--------|-------|
| Prompts embedded in code | Prompts in dedicated config class |
| Same prompt every turn | Phase-aware prompts |
| Duplicate code for P1/P2 | Single consolidated method |
| Generic "Participant 1" | Actual participant names |
| 300 char context limit | 800 char with priority weighting |
| One dialog type | 4 dialog modes |
| "Be Brief" prepending | Removed (use num_predict) |
| Fixed moderation style | Randomized variations |

## Questions?

The refactored code maintains full backward compatibility with your existing:
- OllamaClient class
- TTS generation
- WebSocket events
- Flask routes
- PDF generation

You can adopt changes incrementally or all at once.
