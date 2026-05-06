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
    """Read a file and return its contents, or an error string on failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
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
        return f"Edited {p}"
    except Exception as e:
        return f"Error editing file: {e}"


def run_bash(command: str, working_dir: str | None = None) -> str:
    """Execute a shell command and return stdout/stderr output."""
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
        # rg exit code 1 means "no matches" — not an error
        if result.returncode in (0, 1):
            return result.stdout or "(no matches)"
        return f"Error: rg exited with code {result.returncode}"
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["grep", "-rn", f"--include={file_pattern}", pattern, path],
            capture_output=True,
            text=True,
        )
        return result.stdout or "(no matches)"
    except FileNotFoundError:
        return "Error: ripgrep (rg) or grep not found in PATH"
    except Exception as e:
        return f"Error: {e}"
