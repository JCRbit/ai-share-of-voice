# Spec — Local end-to-end integration (E)

Branch: `local/e-integration` (local only, push blocked by config). Base: `feat/stats-results-tab`.
Goal: a working app that runs a real measurement end to end, without waiting for other teammates'
PRs. Nothing here is pushed; pieces that prove useful go upstream later as normal PRs by their owner.

## 1. Constraints

- `src/ai_share_of_voice/schemas/` stays unchanged. New code adapts to the contracts, not the other way.
- Every real component has a simulated twin (`LLM_MODE=simulated`), so `make check` never calls an API.
- Spending cap enforced in one place (the model client); CrewAI agents never get a raw key (see `agents/factory.py`).
- Analysis stays deterministic code (`stats/`, `tools/stats_tool.py`); the model only answers and writes text.
- `.env` is never read by tooling or committed; only `.env.example` documents variable names.

## 2. Current state (2026-11-02)

| Piece | State | Gap |
| --- | --- | --- |
| Graph, state, interrupt, checkpointer, event log | done (main) | nodes are fakes |
| Stats: metrics, stopping rule, stats tool, charts | done | not wired into nodes |
| Mention tool, skeptic checks + skeptic node | done (main) | not wired into a real run |
| Report + tracking CSV | done (main) | fake `finalize` |
| Results tab view | local commit | no UI host |
| Model client (I3) | `origin/feat/integrations-model-client`, Groq | own `Usage`/exception, does not fit `schemas/llm.py` |
| MCP sources (I4) | remote branch, search is mocked | no real search, not an MCP server |
| Vision (I5) | remote branch | sends path as text, no image |
| `app.py` (I6) | missing | — |

## 3. Requirements

R1. `llm_client.py`: `ModelClient` that returns `(schema_instance, schemas.llm.Usage)`, raises
`exceptions.BudgetExceeded`, logs every call, supports `simulated` and `live` modes, and separate
subject / agent / vision models.
R2. Real nodes (`nodes_real.py`): same signatures as `make_fake_nodes`, built from `NodeDeps` + client.
  - questions: load `data/questions_es.csv` or `questions.csv` by `Brief.language`.
  - sampling: ask the subject model each question once per batch; store raw answers in the run folder.
  - metrics: mention tool → `answers.csv` → stats tool → real `should_stop`.
  - research: real search (or fixture list in simulated mode), robots.txt, untrusted wrapping.
  - screenshots: vision on our own convergence/share chart (guaranteed path).
  - write_draft / skeptic: writer via model; skeptic node from `agents/skeptic.py`.
  - finalize: `automation/finalize.py`.
R3. `app.py` (Gradio): Brief tab → background run with event stream and spend bar → approval →
Results tab (`ui/results_tab.py`).
R4. `make check` green; new tests for R1 and R2 in simulated mode.
R5. One live run under 0.50 USD, logged in `runs/`.

## 4. Tasks (one commit each, small)

| # | Task | Done when |
| --- | --- | --- |
| T0 | Merge local stats branches not yet in main; `make check` green | tests pass on this branch |
| T1 | `.env.example` with all variable names; config loader (`config.py`) | loader tests pass, no key needed |
| T2 | `llm_client.py` simulated + live (provider via OpenAI-compatible API) | budget tests pass; one live ping |
| T3 | Real questions + sampling nodes | simulated graph run produces `answers` |
| T4 | Real metrics node (mention tool + stats tool + `should_stop`) | stop reason comes from D2 |
| T5 | Research node (search + fetch + untrusted wrapping) | malicious fixture test passes |
| T6 | Vision on our chart | finding returned with `is_hypothesis=True` |
| T7 | Writer + skeptic + finalize wired | report.md written after approval |
| T8 | `app.py` Gradio | full simulated run clickable in browser |
| T9 | Live run with low cap | report + CSV in `runs/<id>/` |

## 5. Decisions

- D-a Provider: **Groq** (free tier, OpenAI-compatible, has a vision model). Decided 2026-11-02.
  Defaults in `config.py`: subject `llama-3.3-70b-versatile`, agent `openai/gpt-oss-120b`,
  vision `meta-llama/llama-4-scout-17b-16e-instruct`. Team decision 3 is still pending upstream.
- D-b Search backend: `none` (fixtures) by default; `tavily` optional. Open.
- D-c Cap per run: 0.50 USD in `.env.example`. Proposed.

## 6. Log

- T0 (2026-11-02): all local stats/docs branches were already squash-merged into main. 425 tests
  green. On this machine pytest needs `--basetemp` inside the sandbox scratchpad (system Temp denied).
- T1 (2026-11-02): `config.py` + `tests/test_config.py`, `.env.example` lists every variable.
