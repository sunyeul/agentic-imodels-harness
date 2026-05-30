# Mechanism Mapper Prompt

Map an AGENTIC-IMODELS paper component to this repository.

Read first:

- `.codex/skills/agentic-imodels-paper/references/paper-knowledge.md`
- `.codex/skills/agentic-imodels-paper/references/mechanism-mapping.md`
- `.codex/skills/agentic-imodels-paper/references/reproduction-checklist.md`
- `README.md`

Use the repository vocabulary:

- Runtime harness: `toy_imodels/`
- Editable candidate: `projects/synthetic_regression/experiments/candidate_model.py`
- Experiment entrypoint: `projects.synthetic_regression.run_experiment`
- Repo-local skills: `.codex/skills/`

Return:

- A compact mapping table from paper mechanism to repo mechanism.
- The minimum implementation needed.
- What must remain fixed in the harness.
- What may be edited by a model-design agent.

Rules:

- Do not move scoring, splitting, aggregation, interpretability scoring, or
  leaderboard policy into `projects/synthetic_regression/experiments/candidate_model.py`.
- Do not edit code while acting as mechanism mapper.
