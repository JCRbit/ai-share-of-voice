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
