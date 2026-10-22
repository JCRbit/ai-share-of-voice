# AI Share of Voice — Phases, ownership, structure and Git workflow

Team: **E, J, An, Ab** (4 developers). Duration: 10 working days.

## 1. Role assignment (proposal — swap freely)

The architecture doc leaves this open ("decide who is each person according to interests and level"). Default proposal:

| Dept. | Dev | Why |
| --- | --- | --- |
| P1 Data & Statistics | **E** | Owns stats correctness and mentors Ab (needs patience + stats). |
| P2 Agents & Orchestration | **J** | Owns graph, crew, Skeptic, shared schemas. Highest design load. |
| P3 Integrations & Platform | **An** | Owns LLM client, MCP, vision, CI. Critical path (I3). |
| P4 Product & Docs | **Ab** | Beginner: bounded, off-critical-path tasks, always mentored by E. |

Cross-review (nobody approves their own PR): `E → J → An → E`, and `Ab → E`.

## 2. Opinion on the idea

**Strong points**
- Covers every rubric item: ≥3 agents, conditional edge, bounded loop, human interrupt, code tool, Pydantic tool, MCP, multimodal, automation, UI, logs.
- The narrative is coherent: *measure visibility, doubt it, never claim causality*. This is what C10 (ethics) asks for.
- The stopping rule + Skeptic + human pause are real, defensible control mechanisms, not decoration.

**Risks and recommendations**
1. **Seven CrewAI agents is a lot for 10 days.** The rubric needs ≥3 with no overlap. Keep 7 *roles* but implement the question generator and the stats analyst as thin agents over deterministic code. Cut candidates: Question generator (can be one structured LLM call).
2. **Do not let an LLM write the analysis code.** The "code execution" tool should run *vetted* pandas/matplotlib functions (Wilson intervals, shares, charts) with validated inputs. Safer, testable, reproducible, and still satisfies "figures come from code".
3. **Statistics: answers to the same question are correlated.** Use Wilson intervals for the simple case, and a **bootstrap clustered by question** as the headline interval. Mention this in the glossary and ethics section; it is exactly the kind of rigor that distinguishes the project.
4. **Stopping rule must be deterministic and unit-tested:** stop when CI half-width ≤ target, or marginal gain < ε for N batches, or spend cap, or max batches. Log the reason (already in your design).
5. **Separate the "subject model" (the one we measure) from the "agent model"** (the one that powers CrewAI). Different caps, different logs. Document it.
6. **Extractor accuracy is the weakest link.** Do deterministic matching on name variants first (`competitors.csv`), use the LLM only for recommendation/position/tone, and report accuracy against the 30 hand-labelled answers (also Cohen's kappa if two people label a subset).
7. **Vision is the most likely cut.** Make "interpret our own charts" the guaranteed path (rubric-compliant) and external screenshots the stretch goal. Your cut line already says this; keep it.
8. **Human-in-the-loop:** use LangGraph `interrupt()` with a persistent checkpointer (SQLite) and resume with `Command(resume=...)`. Make the final action idempotent (no duplicated rows in the tracking CSV on retry).
9. **Prompt injection:** the MCP server should return source text wrapped as untrusted data, strip/flag instruction-like content, and the writer/skeptic prompts must say so. Add one test with a malicious fixture page.
10. **Scheduling:** `I3` (LLM client) and `A1` (graph) are the critical path. Day-3 skeleton running in CI is the single most valuable milestone — protect it.
11. **Language:** the architecture doc uses Spanish file names (`preguntas.csv`, `competidores.csv`, `feat/datos-regla-parada`). The project rule is English everywhere, so this plan renames them (`questions.csv`, `competitors.csv`, `feat/stats-stopping-rule`).
12. **Skip the voice agent.** Optional, high cost, no rubric weight beyond bonus.
13. **Pin dependency versions** (CrewAI, LangGraph, `mcp`, Gradio move fast) and verify API names against the official docs on day 1 in a 1-hour spike.

## 3. Phases

| Phase | Days | Goal | Exit criterion |
| --- | --- | --- | --- |
| **0. Foundations** | 1 | Repo, CI, protected `main`, shared contracts | All 4 can run `pytest` green on a PR; `schemas/` merged |
| **1. Modules on mocks** | 2–4 | Each dept. delivers its module against fake data/clients | Unit tests green per module; `mock_run.py` streams events to the UI |
| **2. Skeleton** | 3 | End-to-end path, everything simulated, in CI | `python mock_run.py` runs the full graph in CI |
| **3. Real integration** | 5–6 | Swap mocks for real pieces: client → MCP → vision | One full real run with low spend cap; checkpoint with training team (day 5) |
| **4. Control & critique** | 6–8 | Skeptic loop, error paths, approval page, report + CSV | Error-path tests green; human pause blocks the final action |
| **5. Freeze & docs** | 8–10 | UI freeze, README, rehearsal, delivery | Clean-machine install works; repo review done |

**Cut line (day 7, no stable run):** tone metric → external-site vision → UI extras. Never cut: stopping rule, MCP, Skeptic, human pause.

### Current implementation status

The repository currently has the C1 shared contracts and the A1 LangGraph skeleton implemented. A1 is intentionally simulated: its nodes use fake clients and data so the graph can be exercised before the real integrations are added. The C0/I2 smoke runner is `mock_run.py`; it executes the simulated graph through the human approval pause and resumes it with a demo approval.

## 4. Tasks and files per person

### Everyone — Phase 0 (days 0–1)
| ID | Task | Files |
| --- | --- | --- |
| C4 | Close open decisions: sector and city, model/provider and spend cap, market-study fit (ask training team), Cloudista case, roles | `docs/decisions.md` |
| C1 | Agree contracts (half a day, all 4 in a call) | `src/ai_share_of_voice/schemas/*.py` (see authors below) |

Contract authors: `brief.py` J+Ab, `events.py` J, `analysis.py` J, `stopping.py` E, `llm.py` An, `sources.py` An, `vision.py` An.

### E — Data & Statistics (P1)
| ID | Task | Files | Days |
| --- | --- | --- | --- |
| D1 | Shares + intervals over a test CSV | `src/ai_share_of_voice/stats/metrics.py`, `tests/test_metrics.py`, `tests/fixtures/sample_answers.csv` | 2–3 |
| D2 | `should_stop(history, config)` | `src/ai_share_of_voice/stats/stopping.py`, `tests/test_stopping.py` | 3–4 |
| D3 | Code-execution tool + charts | `src/ai_share_of_voice/tools/stats_tool.py`, `src/ai_share_of_voice/stats/charts.py`, `tests/test_stats_tool.py` | 3–4 |
| D4 | Results & downloads tab | `src/ai_share_of_voice/ui/results_tab.py` | 5–6 |
| — | Mentor Ab: examples, 2 pair sessions/week, PR reviews | `docs/mentoring_notes.md` (optional) | all |
| — | README: Methodology section | `README.md` (section) | 8–9 |

### J — Agents & Orchestration (P2)
| ID | Task | Files | Days |
| --- | --- | --- | --- |
| A1 | Graph with fake nodes: typed state, conditional edge, loop ≤2, interrupt + SQLite checkpointer, JSONL event log | `src/ai_share_of_voice/state.py`, `src/ai_share_of_voice/nodes.py`, `src/ai_share_of_voice/graph.py`, `tests/test_graph_routing.py` | 2–3 |
| A2 | Seven CrewAI agents (role, goal, backstory, tools) | `src/ai_share_of_voice/agents/{question_generator,mention_extractor,analyst,source_researcher,visual_analyst,skeptic,writer}.py`, `src/ai_share_of_voice/agents/__init__.py` | 3–5 |
| A3 | Mention extractor (Pydantic custom tool) | `src/ai_share_of_voice/tools/mention_tool.py`, `tests/test_mention_tool.py` | 3–5 |
| A4 | Activity & approval tab | `src/ai_share_of_voice/ui/activity_tab.py` | 5–6 |
| A5 | Skeptic and its loop (max 2 rounds) | `agents/skeptic.py`, `nodes.py`, `tests/test_skeptic_loop.py` | 6–7 |
| A6 | Error paths: tool failure, no evidence, budget exhausted → partial close | `nodes.py`, `graph.py`, `tests/test_error_paths.py` | 6–7 |
| A7 | Report + tracking CSV (post-approval, idempotent) | `src/ai_share_of_voice/automation/report.py`, `src/ai_share_of_voice/automation/tracking_csv.py`, `tests/test_automation.py` | 6–8 |
| — | README: Architecture section + diagram + process justification; `docs/ethics.md` | `README.md`, `docs/architecture.md`, `docs/ethics.md` | 8–9 |

### An — Integrations & Platform (P3)
| ID | Task | Files | Days |
| --- | --- | --- | --- |
| I1 | Repo, env, CI, protected `main`, day-1 spike (CrewAI, LangGraph, MCP, Gradio install on Python 3.12, versions pinned) | `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` (ruff/pytest config), `.env.example`, `.github/workflows/ci.yml`, `.github/pull_request_template.md`, `CODEOWNERS` | 1 |
| I2 | `mock_run.py` with fake events | `mock_run.py` | 2 |
| I3 | Model client, spend cap, token/cost log | `src/ai_share_of_voice/llm_client.py`, `src/ai_share_of_voice/exceptions.py`, `tests/test_llm_client.py` | 2–4 |
| I4 | MCP sources server (`search_sources`, `read_source`, `check_url`, robots.txt, untrusted-content wrapping) + client adapter | `src/ai_share_of_voice/mcp_server/server.py`, `src/ai_share_of_voice/tools/sources_tool.py`, `tests/test_mcp_server.py`, `tests/fixtures/malicious_page.html` | 2–5 |
| I5 | Vision (`analyze_image`) | `src/ai_share_of_voice/vision.py`, `tests/test_vision.py` | 3–5 |
| I6 | `app.py` skeleton: tabs, background run, spend bar | `app.py`, `src/ai_share_of_voice/ui/brief_tab.py` | 3–4 |
| I7 | Real 50-answer sample for labelling | `scripts/generate_sample.py`, `data/sample_answers.csv` | 4–5 |
| — | README: Configuration & usage | `README.md` (section) | 8–9 |

### Ab — Product & Documentation (P4, beginner, never on the critical path)
| ID | Task | Files | Days |
| --- | --- | --- | --- |
| P1 | Sector, city, 2–5 competitors with name variants | `data/competitors.csv` | 1–2 |
| P2 | 30 questions (id, text, intent), after P1 is agreed | `data/questions.csv` | 2–4 |
| P3 | Glossary + 5–8 screenshots with date and URL | `docs/glossary.md`, `data/screenshots/*.png`, `data/screenshots/index.csv` | 2–5 |
| P4 | Review Brief form texts and error messages; one Issue per problem | `docs/ui_copy_review.md` + GitHub Issues | 5–6 |
| P5 | Hand-label 30 answers (mentions? recommends?) | `data/labeled_answers.csv`, `docs/labeling_guide.md` | 6–7 |
| P6 | Manual interface tests | `docs/manual_test_checklist.md` | 8 |
| P7 | README Installation section, tested on a clean machine | `README.md` (section) | 8–9 |

Rules for Ab: one task per day, clear acceptance criterion, code from templates, every PR reviewed by E, observes (does not integrate) during sessions.

### Shared milestones
| ID | Task | Owner | Day |
| --- | --- | --- | --- |
| C0 | Day-3 end-to-end skeleton, all simulated, running in CI | All | 3 |
| C2 | Full real run (swap mocks one at a time) | All, mob session | 5–6 |
| C5 | Demo script and presentation slides updated with real results | All | 8–9 |
| — | Presentation rehearsal, final repo review | All | 9–10 |

## 5. Project structure

```
ai-share-of-voice/
├── .github/
│   ├── workflows/ci.yml
│   └── pull_request_template.md
├── CODEOWNERS
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── app.py                       # Gradio UI entry point (An)
├── mock_run.py                  # simulated events for UI work (An)
├── scripts/
│   └── generate_sample.py       # real 50-answer sample (An)
├── data/
│   ├── competitors.csv          # Ab
│   ├── questions.csv            # Ab
│   ├── labeled_answers.csv      # Ab
│   ├── sample_answers.csv       # An
│   └── screenshots/             # Ab (+ index.csv)
├── docs/
│   ├── architecture.md          # J
│   ├── ethics.md                # J
│   ├── glossary.md              # Ab
│   ├── labeling_guide.md        # Ab
│   ├── manual_test_checklist.md # Ab
│   └── ui_copy_review.md        # Ab
├── runs/                        # gitignored: logs, reports, tracking CSV, checkpoints
├── src/ai_share_of_voice/
│   ├── __init__.py
│   ├── exceptions.py            # An (BudgetExceeded, ...)
│   ├── llm_client.py            # An
│   ├── vision.py                # An
│   ├── state.py                 # J
│   ├── nodes.py                 # J
│   ├── graph.py                 # J
│   ├── schemas/                 # shared contracts, change only via PR
│   │   ├── brief.py  events.py  analysis.py  stopping.py
│   │   └── llm.py  sources.py  vision.py
│   ├── stats/
│   │   ├── metrics.py  stopping.py  charts.py        # E
│   ├── agents/
│   │   └── question_generator.py  mention_extractor.py  analyst.py
│   │       source_researcher.py  visual_analyst.py  skeptic.py  writer.py   # J
│   ├── tools/
│   │   ├── stats_tool.py        # E (code execution)
│   │   ├── mention_tool.py      # J (Pydantic custom tool)
│   │   └── sources_tool.py      # An (MCP client)
│   ├── mcp_server/
│   │   └── server.py            # An
│   ├── automation/
│   │   ├── report.py  tracking_csv.py                # J
│   └── ui/
│       ├── brief_tab.py         # An
│       ├── activity_tab.py      # J
│       └── results_tab.py       # E
└── tests/
    ├── fixtures/
    └── test_*.py
```

## 6. Bash commands

### 6.1 Day 1 — An creates the repository (I1)

```bash
mkdir ai-share-of-voice && cd ai-share-of-voice
git init -b main

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

mkdir -p .github/workflows scripts data/screenshots docs runs \
         src/ai_share_of_voice/{schemas,stats,agents,tools,mcp_server,automation,ui} tests/fixtures
touch src/ai_share_of_voice/__init__.py src/ai_share_of_voice/schemas/__init__.py \
      src/ai_share_of_voice/stats/__init__.py src/ai_share_of_voice/agents/__init__.py \
      src/ai_share_of_voice/tools/__init__.py src/ai_share_of_voice/mcp_server/__init__.py \
      src/ai_share_of_voice/automation/__init__.py src/ai_share_of_voice/ui/__init__.py \
      README.md .env.example requirements.txt requirements-dev.txt pyproject.toml CODEOWNERS
printf ".venv/\n__pycache__/\n.env\nruns/\n*.pyc\n.pytest_cache/\n.ruff_cache/\n" > .gitignore

git add -A
git commit -m "chore(repo): scaffold project structure"

# After filling .gitignore, requirements, .env.example, CI workflow:
git add .gitignore requirements*.txt pyproject.toml .env.example
git commit -m "build(deps): add pinned requirements and tooling config"
git add .github/workflows/ci.yml .github/pull_request_template.md CODEOWNERS
git commit -m "ci: add lint and test workflow on pull requests"

gh repo create <org>/ai-share-of-voice --public --source=. --remote=origin --push
```

Protect `main` (requires the CI job to be named `test`; on private repos GitHub Free does not support branch protection, so keep the repo public or use a paid plan):

```bash
gh api -X PUT repos/<org>/ai-share-of-voice/branches/main/protection --input - <<'EOF'
{
  "required_status_checks": {"strict": true, "contexts": ["test"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 1},
  "restrictions": null
}
EOF
```

### 6.2 Everyone — onboarding

```bash
git clone git@github.com:<org>/ai-share-of-voice.git && cd ai-share-of-voice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                  # add YOUR key and YOUR spending cap; never commit .env
pytest -q
```

### 6.3 Daily workflow (any developer)

```bash
git switch main && git pull --ff-only
git switch -c feat/<dept>-<task>          # e.g. feat/stats-stopping-rule

# work in small steps; run before every commit
ruff check . && ruff format --check . && pytest -q

git add <only the files of this step>
git commit -m "<type>(<scope>): <subject>"

git fetch origin && git rebase origin/main
git push -u origin HEAD
gh pr create --fill --reviewer <reviewer-handle>
# after approval and green CI:
gh pr merge --squash --delete-branch
```

Branch naming: `feat/stats-*`, `feat/agents-*`, `feat/platform-*`, `docs/*`, `fix/*`, `test/*`, `chore/*`.

### 6.4 Day-1 contracts (C1) — one PR, all review

```bash
git switch -c feat/schemas-contracts
git add src/ai_share_of_voice/schemas/
git commit -m "feat(schemas): add shared pydantic contracts for state, events and tools"
```
(Each author can push their own file to the same branch, or open one PR per file if preferred.)

## 7. Conventional commit messages by person and phase

Format: `type(scope): imperative subject, lowercase, ≤72 chars`. Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `build`, `perf`. Breaking change: `feat(schemas)!: ...` + `BREAKING CHANGE:` in the body.

### E (Data & Statistics)
```
feat(schemas): add stop decision and stopping config models
feat(stats): add share and wilson interval computation
feat(stats): add cluster bootstrap interval by question
test(stats): cover metrics with fixture answers
feat(stats): add should_stop with precision, marginal and budget rules
test(stats): cover every stopping branch and reason string
feat(tools): add code execution tool for metrics and charts
feat(stats): add share and position charts with matplotlib
feat(ui): add results tab with downloads
docs(readme): add methodology section
```

### J (Agents & Orchestration)
```
feat(schemas): add brief, graph event and analyzed response models
feat(graph): add typed state and conditional continue-or-stop edge
feat(graph): add bounded review loop and human interrupt
test(graph): cover routing, loop limit and interrupt resume
feat(agents): define question generator and mention extractor
feat(agents): define analyst, researcher, visual analyst and writer
feat(tools): add mention extraction tool with pydantic input schema
feat(ui): add activity and approval tab
feat(agents): add skeptic agent with claim-level objections
fix(graph): close run in partial state when budget is exhausted
feat(graph): handle tool failure and missing evidence paths
feat(automation): generate markdown report with sources and run log
feat(automation): append idempotent row to tracking csv after approval
docs(ethics): document non-delegated decisions and risks
docs(readme): add architecture diagram and process justification
```

### An (Integrations & Platform)
```
chore(repo): scaffold project structure
build(deps): add pinned requirements and tooling config
ci: add lint and test workflow on pull requests
feat(schemas): add llm usage, source and vision finding models
feat(app): add mock run emitting simulated graph events
feat(llm): add model client with token and cost logging
feat(llm): enforce spending cap and raise BudgetExceeded
test(llm): cover cap enforcement with simulated usage
feat(mcp): add sources server with search, read and url check
feat(mcp): respect robots.txt and wrap content as untrusted data
test(mcp): add malicious page fixture for injection handling
feat(tools): add mcp client adapter for source researcher
feat(vision): add analyze_image for charts and screenshots
feat(app): add gradio skeleton with tabs and spend bar
feat(scripts): generate 50-answer sample with real client
docs(readme): add configuration and usage sections
```

### Ab (Product & Documentation)
```
docs(data): add competitors with name variants
docs(data): add 30 representative questions
docs(glossary): define mention, recommendation, position, tone and interval
docs(data): add competitor website screenshots with date and url
docs(ui): review brief form texts and error messages
docs(data): add hand-labelled answers for extractor evaluation
docs(labeling): add labeling guide
docs(tests): add manual interface test checklist
docs(readme): add installation section
```
(Use `data` as scope for CSVs; they are content, so `docs`/`chore` — pick one and stay consistent.)

### Integration (day 5–6, mob session)
```
refactor(graph): swap simulated client for real llm client
refactor(graph): swap fake sources for mcp tool
refactor(graph): swap fake vision for real analyze_image
fix(graph): pass validated brief through all nodes
chore(release): freeze interface for delivery
```

## 8. Pull request template (`.github/pull_request_template.md`)

```markdown
## Summary
<!-- 1-3 sentences: what this PR does and why. Reviewers read this first. -->

## Related issue
<!-- A closing keyword moves the issue and its board card to Done on merge. -->
Closes #

## Type of change
<!-- Keep the one that applies; it must match the PR title prefix. -->
- [ ] `feat`: new feature
- [ ] `fix`: bug fix
- [ ] `docs`: documentation only
- [ ] `test`: tests only
- [ ] `refactor`: no behaviour change
- [ ] `ci` / `build` / `chore`: tooling, dependencies, housekeeping
- [ ] Breaking change (shared schema or interface; team notified)

## Changes
<!-- Bullet list of the main changes. -->
-

## How to test
<!-- Commands or steps a reviewer can run. -->
1.

## Screenshots / logs
<!-- UI, charts or run output. Delete if not applicable. -->

## Checklist
- [ ] PR title follows Conventional Commits (`type(scope): subject`)
- [ ] Tests added or updated and passing (`make check`)
- [ ] Documentation updated where needed
- [ ] No secrets or `.env` committed
- [ ] Shared schemas unchanged, or team notified
- [ ] I reviewed my own diff; the PR is under 400 net lines

## Notes for reviewers
<!-- Risks, trade-offs, follow-ups, where feedback is most needed. Delete if empty. -->
```

## 9. Day-by-day summary

| Day | E | J | An | Ab |
| --- | --- | --- | --- | --- |
| 1 | C4, C1, mentor setup | C4, C1 | C4, C1, I1 | P1 |
| 2 | D1 | A1 | I2, I3, I4 | P1/P2 |
| 3 | D1→D2, D3, C0 | A1, A2, A3, C0 | I3, I6, I5, C0 | P2/P3 |
| 4 | D2, D3 | A2, A3 | I3 done, I6, I7 | P2/P3 |
| 5 | D4, C2 | A4, C2 | I4, I5, C2 | P3, P4 |
| 6 | D4 | A5, A6, A7 | integration fixes | P4, P5 |
| 7 | fixes, charts polish | A5, A6, A7 | fixes, CI | P5 |
| 8 | README method, C5 | README arch + ethics, C5 | README config, C5 | P6, P7 |
| 9 | rehearsal | rehearsal | rehearsal | P7 |
| 10 | delivery | delivery | delivery | delivery |
