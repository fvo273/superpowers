# OpenRouter Subagents for Subagent-Driven Development

**Date:** 2026-05-05

## Problem

The `subagent-driven-development` skill dispatches implementer and reviewer subagents using Claude Code's built-in `Agent` tool, which only supports Claude models (haiku/sonnet/opus) via the Anthropic API. There is no way to use cheaper third-party models for routine tasks, making the skill expensive to run for large plans.

## Solution

Replace the built-in `Agent` tool dispatch with a Python script (`openrouter_agent.py`) that runs a full agentic loop against OpenRouter's OpenAI-compatible API. The conductor (Claude in the current session) remains on Anthropic; subagents run on any OpenRouter model. Users configure models per role in a local JSON file.

## Architecture

```
Conductor (Claude, current session — Anthropic API)
  ↓ writes prompt to temp file
  ↓ Bash: python openrouter_agent.py --model cheap --prompt-file /tmp/prompt.txt
      ↓ reads OPENROUTER_API_KEY from env
      ↓ reads ~/.claude/openrouter-models.json → resolves role → model string
      ↓ calls OpenRouter API (OpenAI-compatible: https://openrouter.ai/api/v1)
      ↓ agentic loop: tool calls → execute → feed results back → repeat
      ↓ prints final response to stdout when done
  ↓ Conductor reads stdout, proceeds to review phase
```

## Configuration

### API Key

Set environment variable `OPENROUTER_API_KEY`. Recommended: add to `.env` file or shell profile.

### Model Config

File: `~/.claude/openrouter-models.json`

```json
{
  "cheap": "deepseek/deepseek-v4-flash",
  "standard": "deepseek/deepseek-v4-flash",
  "capable": "deepseek/deepseek-v4-flash"
}
```

All three roles default to `deepseek/deepseek-v4-flash`. Users can override per role.

### Startup Validation

When the skill starts, the conductor checks:
1. `OPENROUTER_API_KEY` is set in environment
2. `~/.claude/openrouter-models.json` exists and contains valid JSON with `cheap`, `standard`, `capable` keys

If either check fails, the skill stops immediately with a clear error message and setup instructions.

## Python Agent Script (`openrouter_agent.py`)

Located at: `skills/subagent-driven-development/openrouter_agent.py`

### Dependencies

Requires the `openai` Python package (`pip install openai` or `uv add openai`). Uses OpenRouter's OpenAI-compatible endpoint — no Anthropic SDK needed.

### CLI Interface

```
python openrouter_agent.py --model <role> --prompt-file <path> [--working-dir <path>]
```

- `--model`: role name (`cheap`, `standard`, `capable`) — resolved to model string via config
- `--prompt-file`: path to file containing the full subagent prompt
- `--working-dir`: optional, defaults to current directory

Exits with code 0 on success, non-zero on error. Output to stdout.

### Agentic Loop

1. Load prompt from file
2. Send to OpenRouter with tool definitions
3. If response contains tool calls: execute each tool, append results, send back
4. Repeat until response contains no tool calls
5. Print final text response

### Tools Provided to Subagent

| Tool | Description |
|------|-------------|
| `read_file` | Read a file by path |
| `write_file` | Write content to a file (creates or overwrites) |
| `edit_file` | Replace exact string in a file |
| `run_bash` | Execute a shell command, capture stdout/stderr |
| `glob_files` | Find files matching a glob pattern |
| `grep_files` | Search file contents with regex |

These map 1:1 to Claude Code's Read/Write/Edit/Bash/Glob/Grep tools.

### Error Handling

- API errors (rate limit, auth, network): print error to stderr, exit non-zero
- Tool execution errors: return error string as tool result, let model decide how to proceed
- Infinite loop guard: max 50 tool call rounds, then exit with error

## Changes to Skill Files

### SKILL.md

1. Add **Setup** section at the top: config file format, env var, validation steps
2. Replace **Model Selection** section: reference roles by name (`cheap`/`standard`/`capable`) and state they map to OpenRouter model config
3. Add startup validation to **The Process** section: conductor checks config before dispatching first subagent

### Prompt Templates

Replace the dispatch instruction in each template:

**Before:**
```
Task tool (general-purpose):
  description: "..."
  prompt: |
    ...
```

**After:**
```
Bash tool:
  Write prompt to temp file, then:
  python skills/subagent-driven-development/openrouter_agent.py \
    --model <role> \
    --prompt-file /tmp/subagent-prompt.txt \
    --working-dir <project-dir>
```

Files affected:
- `implementer-prompt.md` → role: `cheap` or `standard` (based on task complexity)
- `spec-reviewer-prompt.md` → role: `standard`
- `code-quality-reviewer-prompt.md` → role: `capable`

## Files Changed

| File | Change |
|------|--------|
| `skills/subagent-driven-development/openrouter_agent.py` | New — Python agent script |
| `skills/subagent-driven-development/SKILL.md` | Modified — setup section + dispatch instructions |
| `skills/subagent-driven-development/implementer-prompt.md` | Modified — dispatch via Bash |
| `skills/subagent-driven-development/spec-reviewer-prompt.md` | Modified — dispatch via Bash |
| `skills/subagent-driven-development/code-quality-reviewer-prompt.md` | Modified — dispatch via Bash |

## Out of Scope

- Fallback to native Claude models when OpenRouter is unavailable
- Parallel subagent dispatch (existing skill already prohibits this)
- Support for OpenRouter models that do not support tool calling
- Streaming subagent output to conductor in real time
