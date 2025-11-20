# File Generation Summary

This document summarizes what files should be generated and under what conditions.

## Generation Flow

### 1. After Dialog Completes (`run_dialog_thread`)

**Always Generated:**
- **JSON file** (`{topic}_{timestamp}.json`): Saved via `save_dialog_to_files()`
  - Contains complete dialog data, metadata, conversation history
  - Location: `output/` directory
  
- **TXT file** (`{topic}_{timestamp}.txt`): Saved via `save_dialog_to_files()`
  - Human-readable transcript of the dialog
  - Location: `output/` directory

**Always Stored (for PDF generation):**
- **`complete_dialog_data[dialog_id]`**: Dictionary stored in memory
  - Contains: `dialog_data`, `prompt_config`, `intermediator_config`, `participant1_config`, `participant2_config`, `gpu_data`
  - Used later for on-demand PDF generation
  - Even if file saving fails, this data is still stored

**Events Emitted:**
- `dialog_saved`: Emitted if JSON/TXT files saved successfully
- `pdf_ready`: Always emitted (indicates PDF can be generated on-demand)
- `summaries_generated`: Emitted after participant summaries are generated

**Background Tasks:**
- **Participant summaries**: Generated in background thread
  - Creates transcript files for each participant
  - Location: `output/audio/{Debate_{topic}}/transcript_{participant_name}.txt`

### 2. When PDF is Requested (`/generate_pdf/<dialog_id>` endpoint)

**Generated On-Demand:**
- **PDF file** (`{topic}_{timestamp}.pdf`): Generated via `generate_pdf_from_dialog()`
  - Only generated when user requests it via the endpoint
  - Location: `output/` directory
  - Contains: Full dialog report with metadata, participants, prompts, conversation, statistics

## Validation Requirements

### For `save_dialog_to_files()`:
- ✅ `dialog_data` must exist and contain `conversation_history`
- ✅ `prompt_config` must exist
- ✅ `intermediator_config` must exist
- ✅ `participant1_config` must exist
- ✅ `participant2_config` must exist
- ✅ `dialog_id` must exist

### For `generate_pdf_from_dialog()`:
- ✅ `dialog_data` must exist and contain `conversation_history`
- ✅ `prompt_config` must exist
- ✅ `intermediator_config` must exist
- ✅ `participant1_config` must exist
- ✅ `participant2_config` must exist
- ✅ `dialog_id` must exist
- ✅ PDF file must be created and exist after generation

## Error Handling

### `save_dialog_to_files()`:
- Returns `(json_path, txt_path)` on success
- Returns `(None, None)` on error
- Errors are logged with full traceback via `debug_log()`
- Errors are also printed to console with traceback

### `generate_pdf_from_dialog()`:
- Returns PDF file path (string) on success
- Returns `None` on error
- Errors are logged with full traceback via `debug_log()`
- Errors are also printed to console with traceback
- Verifies PDF file exists after generation

### `run_dialog_thread()`:
- If `save_dialog_to_files()` fails, warning is logged and error event is emitted
- Data is still stored in `complete_dialog_data` even if file save fails
- This allows PDF generation to still work even if JSON/TXT save failed

## File Naming

All files use the same base filename generated from the topic prompt:
- Format: `{sanitized_topic}_{YYYYMMDD_HHMMSS}`
- Example: `Should_missionaries_in_Africa_pay_bribes_20251118_093626`

## Summary Table

| File Type | When Generated | Location | Required For |
|-----------|---------------|----------|--------------|
| JSON | After dialog completes | `output/` | Data persistence |
| TXT | After dialog completes | `output/` | Human-readable transcript |
| PDF | On-demand (user request) | `output/` | Formatted report |
| Participant Transcripts | After dialog completes (background) | `output/audio/{topic}/` | Individual participant transcripts |

