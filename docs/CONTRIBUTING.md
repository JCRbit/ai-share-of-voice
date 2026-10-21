# Contributing

## Workflow
1. `git switch main && git pull --ff-only`
2. `git switch -c <type>/<dept>-<task>` (e.g. `feat/stats-stopping-rule`)
3. Small commits following Conventional Commits (`type(scope): subject`).
4. Before pushing: `make check`
5. `git fetch origin && git rebase origin/main`, then push and open a PR with `Closes #<issue>`.
6. Nobody approves their own PR. Review rotation: E -> J -> An -> E, Ab -> E.
7. Merge with squash; the PR title becomes the commit message on `main`.

## Rules
- `main` only changes through a reviewed PR with green CI.
- Shared contracts live in `src/ai_share_of_voice/schemas/` and change only via PR, notifying the team.
- Never commit `.env`. Each person uses their own API key and spending cap.
- PRs under 400 net lines.

## Conventional Commit types
`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `build`, `perf`.

## Local checks (while GitHub Actions is unavailable)
GitHub CI cannot run for now, so the checks run on every developer machine:

```bash
pip install -r requirements-dev.txt
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

- On commit: `ruff` and `ruff format` run, and the commit message must follow Conventional Commits.
- On push: `pytest` runs.
- Before opening a PR: `make check` (or the equivalent commands) must pass.
- The reviewer verifies the PR title follows Conventional Commits, because the squash commit takes it as message.
- When CI is available again: run `bash scripts/protect_main.sh` to require the `test` and `pr-title` checks.
