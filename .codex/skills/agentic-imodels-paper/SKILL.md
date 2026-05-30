---
name: agentic-imodels-paper
description: Use when analyzing the AGENTIC-IMODELS paper, mapping its mechanisms to this repo, or checking whether a toy experiment preserves the paper's core ideas.
---

# Agentic Imodels Paper

Use this skill to keep the toy project aligned with the AGENTIC-IMODELS paper.
This is a paper-knowledge and mechanism-mapping skill, not an experiment runner.

## Workflow

1. Read `references/paper-knowledge.md` for the concise paper summary.
2. Read `references/mechanism-mapping.md` when translating paper mechanisms into this repo.
3. Read `references/reproduction-checklist.md` before judging whether an experiment is faithful.
4. Use `.codex/agents/paper-analyst.md` or `.codex/agents/mechanism-mapper.md` when dispatching a focused analysis subagent.

## Boundaries

- Do not edit candidate models directly under this skill.
- Do not treat multi-dataset mean rank as required for the toy MVP.
- Preserve the distinction between runtime harness code and agent-editable experiment code.
