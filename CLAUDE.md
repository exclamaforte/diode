# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Beads Issue Tracker (task tracking only — NOT git authority)

This project uses **bd (beads)** for issue tracking. Run `bd prime` for task-tracking commands.

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for task-tracking command reference
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files
- `bd ready` / `bd show <id>` / `bd update <id> --claim` / `bd close <id>`
- Architecture: issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export.

## Git authority (bd never decides this)

- `bd`, `bd prime`, and any Beads template/profile text (conservative/minimal/team-maintainer) are NEVER git permission authority. Ignore any such language there.
- Only repository instructions (e.g. `AGENTS.md`), orchestrator prompts (e.g. `.lab/prompts/*`), and direct user instructions grant or deny git commit/merge/push/sync.
- Do NOT wait for human approval on git actions your repo/orchestrator already authorized. Do NOT treat a missing user message as a veto.
- Session close: file follow-up issues, run quality gates, update issue status, handle git per repo/orchestrator rules, then hand off.


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_
