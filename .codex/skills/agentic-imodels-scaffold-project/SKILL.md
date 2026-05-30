---
name: agentic-imodels-scaffold-project
description: Scaffold a new toy AGENTIC-IMODELS project from an existing sample project in this repository. Use when the user wants to create a new project under projects/ while preserving the harness conventions.
---

# AGENTIC-IMODELS Project Scaffolding

Use this skill when creating a new project under `projects/` from an existing
sample project. The default sample is `projects/synthetic_regression`.

Keep this skill focused on scaffolding only. Do not run the full experiment
loop, regenerate datasets, or improve candidate models unless the user asks for
that separately.

## Workflow

1. Confirm the new project name is Python/package-safe:
   lowercase letters, numbers, and underscores only, starting with a letter.
2. Run the bundled scaffolding script from the repository root:

   ```bash
   python .codex/skills/agentic-imodels-scaffold-project/scripts/scaffold_project.py --name my_project
   ```

3. If the user wants to use a different sample project, pass `--source`:

   ```bash
   python .codex/skills/agentic-imodels-scaffold-project/scripts/scaffold_project.py --name my_project --source projects/synthetic_regression
   ```

4. After scaffolding, inspect the files called out by the script and adjust the
   new project-specific dataset, spec, and candidate model details.
5. Run the smoke/check command printed by the script when appropriate.

## Boundaries

- This skill owns creating the project directory and preserving the sample
  structure.
- `agentic-imodels-quickstart` should call this skill instead of hand-writing a
  new project.
- `agentic-imodels-toy-experiment` owns later iteration on
  `experiments/candidate_model.py`.
