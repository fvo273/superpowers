# Code Quality Reviewer Prompt Template

Use this template when dispatching a code quality reviewer subagent.

**Purpose:** Verify implementation is well-built (clean, tested, maintainable)

**Only dispatch after spec compliance review passes.**

```
Bash tool — dispatch via OpenRouter:

1. Read the full content of `skills/requesting-code-review/code-reviewer.md` and use it as your prompt template.

2. Write the filled-in prompt to a temp file:

   Use the Write tool to create `/tmp/subagent-quality-reviewer-prompt.txt`.

3. Run the subagent (use `capable` role):

   ```bash
   uv run skills/subagent-driven-development/openrouter_agent.py \
     --model capable \
     --prompt-file /tmp/subagent-quality-reviewer-prompt.txt \
     --working-dir [project-root]
   ```

4. Read stdout as the reviewer's assessment.

Fill in these fields in the prompt before writing the temp file:
- `{DESCRIPTION}` — brief summary of what was built (from implementer's report)
- `{PLAN_OR_REQUIREMENTS}` — task text or plan file path (e.g. `Task N from docs/superpowers/plans/...`)
- `{BASE_SHA}` — commit before the task started
- `{HEAD_SHA}` — current commit (after implementer's work)
```

**In addition to standard code quality concerns, the reviewer should check:**
- Does each file have one clear responsibility with a well-defined interface?
- Are units decomposed so they can be understood and tested independently?
- Is the implementation following the file structure from the plan?
- Did this implementation create new files that are already large, or significantly grow existing files? (Don't flag pre-existing file sizes — focus on what this change contributed.)

**Code reviewer returns:** Strengths, Issues (Critical/Important/Minor), Assessment
