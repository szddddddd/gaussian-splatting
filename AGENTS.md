# AGENTS.md

## Role

You are a research-coding agent working on this repository. Prioritize correctness, reproducibility, and clear experiment records over quick edits.

## Mandatory environment

Before running any Python, pip, build, training, evaluation, or test command, activate the project conda environment in the same shell:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gaussian_splatting
```

If the command is non-interactive, use:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && <COMMAND>'
```

Do not install packages globally or into `base` unless explicitly requested.

## Worklog requirement

Maintain `worklog.md` at the repository root.

For every meaningful action, append a concise entry with:

````md
## YYYY-MM-DD HH:MM - <task title>

### Goal
- What I am trying to achieve.

### Files inspected
- `path/to/file`: reason inspected.

### Changes made
- `path/to/file`: concise summary of edits.

### Commands run
```bash
<exact command>
````

### Results

* Pass/fail, key output, errors, metrics, or observations.

### Current state

* What works now.
* What is still uncertain or broken.

### Next recommended step

* A clear next action for the user or next agent/model.

```

Never delete previous worklog entries. If a previous conclusion was wrong, add a correction entry instead.

## Research coding workflow

1. First inspect the repository structure, relevant README/docs, and current git status.
2. Before changing code, identify the minimal files related to the task.
3. Prefer small, explainable patches.
4. Do not rewrite large modules unless necessary.
5. Reproduce or understand the issue before patching when possible.
6. Treat “no code change needed” as a valid outcome if inspection shows the request is already satisfied.
7. After edits, run the smallest relevant check first, then broader checks if practical.
8. Record all important commands and results in `worklog.md`.

## Git safety

- Run `git status` before and after modifications.
- Do not run destructive commands such as `rm -rf`, `git reset --hard`, `git clean -fd`, or mass file moves unless explicitly requested.
- Do not modify large datasets, checkpoints, or generated outputs unless the task specifically requires it.
- Do not commit or push unless explicitly asked.

## Final response format

When finishing a task, report:

1. What changed.
2. Which files were modified.
3. Which commands/tests were run.
4. Whether the task is complete.
5. Any remaining risks or next steps.

Also mention that details were appended to `worklog.md`.
```
