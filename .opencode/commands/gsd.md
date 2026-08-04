---
description: Run this project's native GSD v3 workflow with DeepSeek V4 Flash Latest via OpenRouter
agent: build
model: openrouter/~deepseek/deepseek-v4-flash-latest
subtask: false
---

This repository uses native GSD v3 with `.gsd/gsd.db` as its authoritative
state. Never load `gsd-autonomous` or another legacy GSD skill from
`~/.claude/skills`. Never use `gsd-sdk` or look for a `.planning` directory.

The requested action is: `$ARGUMENTS`

Use only the project wrapper `.opencode/bin/gsd-v3-deepseek`, which securely
reuses the OpenRouter credential already stored by OpenCode. It must
remain the parent command; do not invoke `/opt/homebrew/bin/gsd` directly.

- With no argument, `status`, or `query`: inspect only with
  `.opencode/bin/gsd-v3-deepseek headless query`.
- With `next`: first inspect with `headless query`, then execute exactly one
  unit with `.opencode/bin/gsd-v3-deepseek headless --model openrouter/~deepseek/deepseek-v4-flash-latest next`.
- With `auto`: first inspect with `headless query`, then run continuously with
  `.opencode/bin/gsd-v3-deepseek headless --model openrouter/~deepseek/deepseek-v4-flash-latest auto`.
- For any other argument, do not improvise or fall back to legacy GSD. Explain
  that this bridge supports `status`, `next`, and `auto`.

`next` and `auto` perform real model work and are not dry runs. Do not launch
either merely to test the bridge. Report the selected model before execution,
monitor the command, and summarize the resulting GSD state when it exits.
