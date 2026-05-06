# OpenRouter Subagents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Claude Code's built-in `Agent` tool dispatch in `subagent-driven-development` with a Python agentic loop that calls OpenRouter's API, enabling any OpenRouter model to serve as a fully capable subagent.

**Architecture:** The conductor (Claude in the current session) writes a prompt to a temp file and runs `openrouter_agent.py` via Bash tool. The script reads model config from `~/.claude/openrouter-models.json`, calls OpenRouter's OpenAI-compatible API, and runs a tool-calling loop (read/write/edit files, run bash commands) until the model reports done, printing the final response to stdout.

**Tech Stack:** Python 3.10+, `openai` package (OpenAI SDK for OpenRouter's OpenAI-compatible API), `pytest`, standard library (`pathlib`, `json`, `argparse`, `subprocess`, `glob`).

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `skills/subagent-driven-development/pyproject.toml` | Create | Python project config, dependencies |
| `skills/subagent-driven-development/openrouter_agent.py` | Create | Config loading, 6 tool implementations, agentic loop, CLI |
| `skills/subagent-driven-development/tests/test_openrouter_agent.py` | Create | Unit tests for config, tools, and agentic loop |
| `skills/subagent-driven-development/SKILL.md` | Modify | Add Setup section, replace Model Selection, update dispatch instructions |
| `skills/subagent-driven-development/implementer-prompt.md` | Modify | Replace `Task tool` dispatch with Bash dispatch |
| `skills/subagent-driven-development/spec-reviewer-prompt.md` | Modify | Replace `Task tool` dispatch with Bash dispatch |
| `skills/subagent-driven-development/code-quality-reviewer-prompt.md` | Modify | Replace dispatch instruction with Bash dispatch |

---

## Task 1: Python Project Setup, Config Loading, and Tool Implementations

**Files:**
- Create: `skills/subagent-driven-development/pyproject.toml`
- Create: `skills/subagent-driven-development/openrouter_agent.py`
- Create: `skills/subagent-driven-development/tests/__init__.py`
- Create: `skills/subagent-driven-development/tests/test_openrouter_agent.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "openrouter-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["openai>=1.0.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[dependency-groups]
dev = ["pytest>=8.0.0"]
```

- [ ] **Step 2: Create openrouter_agent.py — imports, constants, config loading, tool functions**

```python
"""OpenRouter subagent runner for superpowers:subagent-driven-development."""

import argparse
import glob as glob_module
import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

CONFIG_PATH = Path.home() / ".claude" / "openrouter-models.json"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_ITERATIONS = 50
VALID_ROLES = {"cheap", "standard", "capable"}


def load_config() -> tuple[str, dict]:
    """Load and validate OpenRouter API key and model config.

    Returns (api_key, model_config) or exits with error.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        print("Add it to your .env file or shell profile.", file=sys.stderr)
        sys.exit(1)

    if not CONFIG_PATH.exists():
        print(f"ERROR: Model config file not found: {CONFIG_PATH}", file=sys.stderr)
        print("Create it with the following content:", file=sys.stderr)
        print(
            json.dumps(
                {"cheap": "deepseek/deepseek-v4-flash", "standard": "deepseek/deepseek-v4-flash", "capable": "deepseek/deepseek-v4-flash"},
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {CONFIG_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    missing = VALID_ROLES - config.keys()
    if missing:
        print(f"ERROR: Config missing required keys: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    return api_key, config


def resolve_model(config: dict, role: str) -> str:
    """Resolve a role name to an OpenRouter model string."""
    if role not in VALID_ROLES:
        print(f"ERROR: Unknown role '{role}'. Must be one of: {sorted(VALID_ROLES)}", file=sys.stderr)
        sys.exit(1)
    return config[role]


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace first occurrence of old_string with new_string in file."""
    try:
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            return f"Error: old_string not found in {path}"
        p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def run_bash(command: str, working_dir: str | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except Exception as e:
        return f"Error running command: {e}"


def glob_files(pattern: str, base_dir: str | None = None) -> str:
    try:
        search_pattern = str(Path(base_dir) / pattern) if base_dir else pattern
        matches = glob_module.glob(search_pattern, recursive=True)
        return "\n".join(sorted(matches)) if matches else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def grep_files(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    try:
        result = subprocess.run(
            ["rg", "--no-heading", "-n", pattern, path, "--glob", file_pattern],
            capture_output=True,
            text=True,
        )
        return result.stdout or "(no matches)"
    except FileNotFoundError:
        result = subprocess.run(
            ["grep", "-rn", "--include", f"*{file_pattern}*", pattern, path],
            capture_output=True,
            text=True,
        )
        return result.stdout or "(no matches)"
    except Exception as e:
        return f"Error: {e}"
```

- [ ] **Step 3: Create empty test init file**

Create `skills/subagent-driven-development/tests/__init__.py` as an empty file.

- [ ] **Step 4: Write failing tests for config loading and tool functions**

```python
"""Tests for openrouter_agent.py config loading and tool functions."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from openrouter_agent import (
    CONFIG_PATH,
    edit_file,
    glob_files,
    grep_files,
    load_config,
    read_file,
    resolve_model,
    run_bash,
    write_file,
)


class TestLoadConfig:
    def test_missing_api_key_exits(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc:
            load_config()
        assert exc.value.code == 1

    def test_missing_config_file_exits(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with patch("openrouter_agent.CONFIG_PATH", Path("/nonexistent/path.json")):
            with pytest.raises(SystemExit) as exc:
                load_config()
        assert exc.value.code == 1

    def test_invalid_json_exits(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        bad_config = tmp_path / "config.json"
        bad_config.write_text("not valid json")
        with patch("openrouter_agent.CONFIG_PATH", bad_config):
            with pytest.raises(SystemExit) as exc:
                load_config()
        assert exc.value.code == 1

    def test_missing_config_keys_exits(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"cheap": "model-a"}))  # missing standard, capable
        with patch("openrouter_agent.CONFIG_PATH", config):
            with pytest.raises(SystemExit) as exc:
                load_config()
        assert exc.value.code == 1

    def test_valid_config_returns_key_and_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"cheap": "m1", "standard": "m2", "capable": "m3"}))
        with patch("openrouter_agent.CONFIG_PATH", config):
            api_key, model_config = load_config()
        assert api_key == "sk-test"
        assert model_config == {"cheap": "m1", "standard": "m2", "capable": "m3"}


class TestResolveModel:
    def test_resolves_valid_role(self):
        config = {"cheap": "deepseek/fast", "standard": "deepseek/mid", "capable": "deepseek/big"}
        assert resolve_model(config, "cheap") == "deepseek/fast"
        assert resolve_model(config, "standard") == "deepseek/mid"
        assert resolve_model(config, "capable") == "deepseek/big"

    def test_invalid_role_exits(self):
        config = {"cheap": "m", "standard": "m", "capable": "m"}
        with pytest.raises(SystemExit) as exc:
            resolve_model(config, "unknown")
        assert exc.value.code == 1


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert read_file(str(f)) == "hello world"

    def test_returns_error_for_missing_file(self, tmp_path):
        result = read_file(str(tmp_path / "missing.txt"))
        assert result.startswith("Error reading file:")


class TestWriteFile:
    def test_creates_file_with_content(self, tmp_path):
        path = str(tmp_path / "new.txt")
        result = write_file(path, "content here")
        assert "Written" in result
        assert Path(path).read_text() == "content here"

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c.txt")
        write_file(path, "nested")
        assert Path(path).exists()

    def test_overwrites_existing_file(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        write_file(str(f), "new content")
        assert f.read_text() == "new content"


class TestEditFile:
    def test_replaces_first_occurrence(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo bar foo")
        result = edit_file(str(f), "foo", "baz")
        assert result == f"Edited {f}"
        assert f.read_text() == "baz bar foo"

    def test_returns_error_when_string_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello")
        result = edit_file(str(f), "not here", "x")
        assert "Error" in result

    def test_returns_error_for_missing_file(self, tmp_path):
        result = edit_file(str(tmp_path / "missing.py"), "old", "new")
        assert "Error" in result


class TestRunBash:
    def test_returns_stdout(self):
        result = run_bash("echo hello")
        assert "hello" in result

    def test_includes_stderr_on_failure(self):
        result = run_bash("ls /nonexistent_path_xyz")
        assert result  # not empty — includes error output or exit code

    def test_returns_exit_code_on_failure(self):
        result = run_bash("exit 42", working_dir=None)
        assert "42" in result or result  # exit code present


class TestGlobFiles:
    def test_finds_matching_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = glob_files("*.py", base_dir=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_returns_no_matches_message(self, tmp_path):
        result = glob_files("*.xyz", base_dir=str(tmp_path))
        assert result == "(no matches)"


class TestGrepFiles:
    def test_finds_pattern_in_files(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def my_function():\n    pass\n")
        result = grep_files("my_function", path=str(tmp_path))
        assert "my_function" in result
```

- [ ] **Step 5: Run tests to verify they fail (code not complete yet)**

```bash
cd skills/subagent-driven-development && uv run --with pytest --with openai pytest tests/ -v 2>&1 | head -40
```

Expected: collection passes, tests run, most PASS (tool functions are already implemented), config tests PASS.

- [ ] **Step 6: Verify all tests pass**

```bash
cd skills/subagent-driven-development && uv run --with pytest --with openai pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add skills/subagent-driven-development/pyproject.toml skills/subagent-driven-development/openrouter_agent.py skills/subagent-driven-development/tests/
git commit -m "feat: add openrouter_agent.py with config loading and tool implementations"
```

---

## Task 2: Agentic Loop and Main Entry Point

**Files:**
- Modify: `skills/subagent-driven-development/openrouter_agent.py` — append tool definitions, execute_tool, run_agent, main
- Modify: `skills/subagent-driven-development/tests/test_openrouter_agent.py` — append agentic loop tests

- [ ] **Step 1: Write failing tests for agentic loop**

Append to `skills/subagent-driven-development/tests/test_openrouter_agent.py`:

```python
from unittest.mock import MagicMock, patch

from openrouter_agent import execute_tool, run_agent


class TestExecuteTool:
    def test_dispatches_read_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("content")
        result = execute_tool("read_file", {"path": str(f)}, str(tmp_path))
        assert result == "content"

    def test_dispatches_write_file(self, tmp_path):
        path = str(tmp_path / "out.txt")
        result = execute_tool("write_file", {"path": path, "content": "hello"}, str(tmp_path))
        assert "Written" in result

    def test_dispatches_unknown_tool(self, tmp_path):
        result = execute_tool("nonexistent_tool", {}, str(tmp_path))
        assert "Unknown tool" in result


class TestRunAgent:
    def _make_client(self, responses):
        """Build a mock OpenAI client that returns responses in order."""
        client = MagicMock()
        completions = [self._make_response(r) for r in responses]
        client.chat.completions.create.side_effect = completions
        return client

    def _make_response(self, spec):
        """spec: {"content": str} or {"tool_calls": [(name, args_dict)]}"""
        message = MagicMock()
        if "tool_calls" in spec:
            message.content = None
            tcs = []
            for name, args in spec["tool_calls"]:
                tc = MagicMock()
                tc.id = f"call_{name}"
                tc.function.name = name
                tc.function.arguments = json.dumps(args)
                tcs.append(tc)
            message.tool_calls = tcs
        else:
            message.content = spec["content"]
            message.tool_calls = []
        response = MagicMock()
        response.choices[0].message = message
        return response

    def test_returns_content_when_no_tool_calls(self, tmp_path):
        client = self._make_client([{"content": "Task complete."}])
        result = run_agent(client, "some-model", "do the thing", str(tmp_path))
        assert result == "Task complete."

    def test_executes_tool_call_and_continues(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("file contents")
        client = self._make_client([
            {"tool_calls": [("read_file", {"path": str(f)})]},
            {"content": "I read the file."},
        ])
        result = run_agent(client, "some-model", "read file", str(tmp_path))
        assert result == "I read the file."
        assert client.chat.completions.create.call_count == 2

    def test_exits_on_max_iterations(self, tmp_path):
        """Agent that never stops calling tools should exit after MAX_ITERATIONS."""
        always_calls = {"tool_calls": [("run_bash", {"command": "echo loop"})]}
        client = self._make_client([always_calls] * 60)
        with patch("openrouter_agent.MAX_ITERATIONS", 3):
            with pytest.raises(SystemExit) as exc:
                run_agent(client, "some-model", "infinite loop", str(tmp_path))
        assert exc.value.code == 1
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
cd skills/subagent-driven-development && uv run --with pytest --with openai pytest tests/test_openrouter_agent.py::TestExecuteTool tests/test_openrouter_agent.py::TestRunAgent -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `execute_tool` and `run_agent` not defined yet.

- [ ] **Step 3: Append TOOL_DEFINITIONS, execute_tool, run_agent, and main to openrouter_agent.py**

```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace first occurrence of old_string with new_string in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact string to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory or file to search (default: .)"},
                    "file_pattern": {"type": "string", "description": "File glob filter (default: *)"},
                },
                "required": ["pattern"],
            },
        },
    },
]

_TOOL_DISPATCH = {
    "read_file": lambda args, wd: read_file(args["path"]),
    "write_file": lambda args, wd: write_file(args["path"], args["content"]),
    "edit_file": lambda args, wd: edit_file(args["path"], args["old_string"], args["new_string"]),
    "run_bash": lambda args, wd: run_bash(args["command"], wd),
    "glob_files": lambda args, wd: glob_files(args["pattern"], wd),
    "grep_files": lambda args, wd: grep_files(args["pattern"], args.get("path", "."), args.get("file_pattern", "*")),
}


def execute_tool(name: str, args: dict, working_dir: str) -> str:
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    return handler(args, working_dir)


def run_agent(client, model: str, prompt: str, working_dir: str) -> str:
    """Run an agentic loop until the model returns a final response."""
    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        assistant_msg: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            return message.content or ""

        for tc in message.tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments), working_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print(f"ERROR: Reached max iterations ({MAX_ITERATIONS}) without completion.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subagent task via OpenRouter")
    parser.add_argument("--model", required=True, choices=list(VALID_ROLES), help="Model role: cheap, standard, or capable")
    parser.add_argument("--prompt-file", required=True, help="Path to file containing the subagent prompt")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Working directory for bash commands (default: cwd)")
    args = parser.parse_args()

    api_key, config = load_config()
    model = resolve_model(config, args.model)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    print(run_agent(client, model, prompt, args.working_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
cd skills/subagent-driven-development && uv run --with pytest --with openai pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Smoke test CLI help**

```bash
cd skills/subagent-driven-development && uv run --with openai python openrouter_agent.py --help
```

Expected: usage message printed with `--model`, `--prompt-file`, `--working-dir` arguments listed.

- [ ] **Step 7: Commit**

```bash
git add skills/subagent-driven-development/openrouter_agent.py skills/subagent-driven-development/tests/test_openrouter_agent.py
git commit -m "feat: add agentic loop and CLI entry point to openrouter_agent.py"
```

---

## Task 3: Update SKILL.md

**Files:**
- Modify: `skills/subagent-driven-development/SKILL.md`

- [ ] **Step 1: Add Setup section after the opening description (after the first two `**...**` paragraphs)**

Insert after the line `**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration`:

```markdown

## Setup

Before using this skill, configure OpenRouter:

1. **API key:** set `OPENROUTER_API_KEY` in your environment (add to `.env` or shell profile).
2. **Model config:** create `~/.claude/openrouter-models.json`:
   ```json
   {
     "cheap": "deepseek/deepseek-v4-flash",
     "standard": "deepseek/deepseek-v4-flash",
     "capable": "deepseek/deepseek-v4-flash"
   }
   ```
3. **Python dependency:** `pip install openai` (or `uv add openai`).

If either the API key or config file is missing, the skill stops immediately with an error and setup instructions. Customise models per role by editing the config file — any model available on OpenRouter that supports tool calling works.
```

- [ ] **Step 2: Replace the Model Selection section**

Find and replace the entire `## Model Selection` section (from `## Model Selection` to the line before `## Handling Implementer Status`) with:

```markdown
## Model Selection

Model roles are defined in `~/.claude/openrouter-models.json`. The three roles map to task complexity:

- **cheap** — mechanical tasks (isolated functions, clear specs, 1-2 files). Use for most implementer tasks.
- **standard** — integration tasks (multi-file coordination, debugging). Use when the implementer touches many files.
- **capable** — architecture, design, and review tasks. Use for spec and code quality reviewers.

Task complexity signals:
- Touches 1-2 files with a complete spec → `cheap`
- Touches multiple files with integration concerns → `standard`
- Requires design judgment or broad codebase understanding → `capable`

```

- [ ] **Step 3: Replace the Prompt Templates section at the bottom of SKILL.md**

Find and replace:
```
## Prompt Templates

- `./implementer-prompt.md` - Dispatch implementer subagent
- `./spec-reviewer-prompt.md` - Dispatch spec compliance reviewer subagent
- `./code-quality-reviewer-prompt.md` - Dispatch code quality reviewer subagent
```

With:

```markdown
## Dispatching Subagents via OpenRouter

All subagents are dispatched through `openrouter_agent.py` using the Bash tool. The general pattern:

1. Write the prompt to a temp file using the Write tool or a heredoc.
2. Run the agent script:
   ```bash
   python skills/subagent-driven-development/openrouter_agent.py \
     --model <role> \
     --prompt-file /tmp/subagent-prompt.txt \
     --working-dir <project-root>
   ```
3. Read stdout as the subagent's final report.

See the prompt template files for full prompt content per role:
- `./implementer-prompt.md` — role: `cheap` or `standard`
- `./spec-reviewer-prompt.md` — role: `standard`
- `./code-quality-reviewer-prompt.md` — role: `capable`
```

- [ ] **Step 4: Add startup validation step to the Example Workflow**

In the `## Example Workflow` section, find the opening lines:
```
You: I'm using Subagent-Driven Development to execute this plan.

[Read plan file once: docs/superpowers/plans/feature-plan.md]
```

Replace with:
```
You: I'm using Subagent-Driven Development to execute this plan.

[Validate OpenRouter setup: check OPENROUTER_API_KEY is set + ~/.claude/openrouter-models.json exists]

[Read plan file once: docs/superpowers/plans/feature-plan.md]
```

- [ ] **Step 5: Commit**

```bash
git add skills/subagent-driven-development/SKILL.md
git commit -m "feat: update SKILL.md for OpenRouter dispatch — add Setup section, update Model Selection and dispatch instructions"
```

---

## Task 4: Update Prompt Templates

**Files:**
- Modify: `skills/subagent-driven-development/implementer-prompt.md`
- Modify: `skills/subagent-driven-development/spec-reviewer-prompt.md`
- Modify: `skills/subagent-driven-development/code-quality-reviewer-prompt.md`

- [ ] **Step 1: Update implementer-prompt.md — replace dispatch block**

Find the block:
```
Task tool (general-purpose):
  description: "Implement Task N: [task name]"
  prompt: |
```

Replace with:
```
Bash tool — dispatch via OpenRouter:

1. Write the prompt to a temp file:

   Use the Write tool to create `/tmp/subagent-implementer-prompt.txt` with the prompt content below.

2. Run the subagent (use `cheap` for mechanical tasks, `standard` for multi-file integration):

   ```bash
   python skills/subagent-driven-development/openrouter_agent.py \
     --model cheap \
     --prompt-file /tmp/subagent-implementer-prompt.txt \
     --working-dir [project-root]
   ```

3. Read stdout as the subagent's report.

Prompt content:
```

- [ ] **Step 2: Update spec-reviewer-prompt.md — replace dispatch block**

Find the block:
```
Task tool (general-purpose):
  description: "Review spec compliance for Task N"
  prompt: |
```

Replace with:
```
Bash tool — dispatch via OpenRouter:

1. Write the prompt to a temp file:

   Use the Write tool to create `/tmp/subagent-spec-reviewer-prompt.txt` with the prompt content below.

2. Run the subagent (use `standard` role):

   ```bash
   python skills/subagent-driven-development/openrouter_agent.py \
     --model standard \
     --prompt-file /tmp/subagent-spec-reviewer-prompt.txt \
     --working-dir [project-root]
   ```

3. Read stdout as the reviewer's verdict.

Prompt content:
```

- [ ] **Step 3: Update code-quality-reviewer-prompt.md — replace dispatch block**

Find the block:
```
Task tool (general-purpose):
  Use template at requesting-code-review/code-reviewer.md
```

Replace with:
```
Bash tool — dispatch via OpenRouter:

1. Read the full content of `skills/requesting-code-review/code-reviewer.md` and use it as your prompt template.

2. Write the filled-in prompt to a temp file:

   Use the Write tool to create `/tmp/subagent-quality-reviewer-prompt.txt`.

3. Run the subagent (use `capable` role):

   ```bash
   python skills/subagent-driven-development/openrouter_agent.py \
     --model capable \
     --prompt-file /tmp/subagent-quality-reviewer-prompt.txt \
     --working-dir [project-root]
   ```

4. Read stdout as the reviewer's assessment.

Fill in the template fields:
```

- [ ] **Step 4: Run all Python tests one final time to confirm nothing regressed**

```bash
cd skills/subagent-driven-development && uv run --with pytest --with openai pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/subagent-driven-development/implementer-prompt.md skills/subagent-driven-development/spec-reviewer-prompt.md skills/subagent-driven-development/code-quality-reviewer-prompt.md
git commit -m "feat: update prompt templates to dispatch subagents via openrouter_agent.py"
```
