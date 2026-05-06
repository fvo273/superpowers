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
        config.write_text(json.dumps({"cheap": "model-a"}))
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
        assert result

    def test_returns_exit_code_on_failure(self):
        result = run_bash("exit 42")
        assert "42" in result


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
        result = grep_files("my_function", path=str(tmp_path), file_pattern="*.py")
        assert "my_function" in result


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
