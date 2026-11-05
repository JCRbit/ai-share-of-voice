# AI Share of Voice

Measure how often an AI assistant mentions and recommends your brand compared with its
competitors, with confidence intervals, traceable answers and a report reviewed before approval.

Project III, Module V (AI Engineering) — AgentSquad: multi-agent system with CrewAI,
LangGraph and MCP.

> Every result is a dated sample of one model and one question panel. It is never presented
> as market share, and no causality is claimed between a web source and a recommendation.

## Current status

The `local/e-integration` branch includes a working local Gradio application in Spanish and
English, with a simulated demo and a live measurement path. It supports question review,
batched sampling, deterministic statistics, business discovery, web research through MCP,
optional chart vision, report drafting, skeptic review and human approval.

Results include brand comparisons, answer examples, downloadable artifacts and trends across
past measurements. The UI stores checkpoints in SQLite so saved measurements can be reopened.
Live runs remain subject to provider quotas and the configured budget; a full-capacity live
validation is still pending. See the [local integration spec](specs/local-integration.md)
for implementation notes and open work, and the [project plan](docs/PROJECT_PLAN.md) for
the wider roadmap.

## Quick start: free demo

**Requirements:** Git and Python 3.12 or 3.13 (`pyproject.toml` requires `>=3.12,<3.14`).
Run the commands from the repository root. No API key is needed for the simulated demo.

### Windows (PowerShell)

```powershell
git clone https://github.com/JCRbit/ai-share-of-voice.git
cd ai-share-of-voice
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py --simulated
```

These commands use the environment's Python directly, so activation is optional.
For an existing checkout, skip cloning; copy `.env.example` only if `.env` does not exist.

### Linux / macOS

```bash
git clone https://github.com/JCRbit/ai-share-of-voice.git
cd ai-share-of-voice
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python app.py --simulated
```

Open the address printed in the terminal, normally `http://127.0.0.1:7860`.
`--simulated` forces simulated model responses and fixture-only research for that process,
even if `.env` is configured for live runs. Demo figures illustrate the workflow; they are
not measurements of a real AI provider.

## Run a measurement in the app

1. Enter a brand, **two to five competitors**, sector and location. Choose the language,
   question count, round limit and precision target.
2. Prepare and review the questions. Edit the panel before starting; questions naming tracked
   brands are rejected. The included demo panels concern food delivery in Madrid.
3. Start the measurement and follow its activity. Sampling stops according to precision,
   marginal return, available budget or the round limit.
4. Read the draft and the skeptic's objections, then approve or reject. Approval is required
   before final reports and the tracking entry are written.
5. Explore Results: mention and recommendation rates, uncertainty, example answers, research,
   discovered businesses and downloads. Use **My measurements** to reopen saved UI runs.

Trend comparisons match brand, sector, city and measured model. Each point identifies its
question set; changes across different or unknown panels are not called clear changes.

## Configuration and live mode

Copy [`.env.example`](.env.example) to `.env` and fill in your own credentials. `.env` and
generated `runs/` files are ignored by Git.

| Variable | Purpose |
| --- | --- |
| `LLM_MODE` | `simulated` (default) or `live` |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` | Agent provider (`groq` or `gemini`), credential and model |
| `SUBJECT_PROVIDER`, `SUBJECT_API_KEY`, `SUBJECT_MODEL` | Model being measured; blank provider inherits the agent provider |
| `VISION_PROVIDER`, `VISION_API_KEY`, `VISION_MODEL` | Chart-reading model; blank provider inherits the agent provider |
| `LLM_BASE_URL`, `SUBJECT_BASE_URL`, `VISION_BASE_URL` | Optional endpoint overrides for each role |
| `SPEND_CAP_USD` | Model-call budget per run; the example file sets `0.50` USD |
| `SEARCH_PROVIDER` | `none` (fixtures), `duckduckgo` or `tavily` |
| `SEARCH_API_KEY` | Required for Tavily |

Subject and vision roles reuse an available key for the same provider unless their own key
is set. Blank model names use [the defaults in `config.py`](src/ai_share_of_voice/config.py).
Pin the subject model explicitly for comparable measurements, and verify model availability
with your provider before a live run. Choose a vision-capable model for the vision role;
the existing Groq vision default is a known integration limitation (see the local spec).
An unavailable vision step can be skipped without invalidating the computed statistics.

To make real model calls, set `LLM_MODE=live`, provide the required keys and run:

```bash
python app.py
```

On Windows without activation, use `.\.venv\Scripts\python.exe app.py`.
`SEARCH_PROVIDER=none` still uses fixture sources in live model mode; choose a web backend
for live research. Search-service quotas and charges are separate from the model-call budget.
The budget uses estimated model costs, so it is not a provider billing guarantee. Begin
with a small panel and few rounds: quota exhaustion or a failed required step can leave a
partial run, with its audit trail preserved.

## Command-line runner

The integration runner uses the same `.env` settings. With `LLM_MODE=simulated` and
`SEARCH_PROVIDER=none`, it can exercise the pipeline without external calls:

```bash
python scripts/real_run.py --questions 5 --batches 2 --language es
```

Generate a panel for another market (one line works in both Bash and PowerShell):

```bash
python scripts/real_run.py --sector "clothing stores" --city Valencia --country España --brand Zara --competitors "Mango,H&M" --generate --questions 5 --batches 2 --language es
```

Reuse a saved panel with `--panel runs/<run_id>/questions.csv`. The four custom brief flags
(`--sector`, `--city`, `--brand`, `--competitors`) must be supplied together. Use
`python scripts/real_run.py --help` for all options.

**The integration runner automatically approves the finalization step.** Use the app for
interactive human approval. CLI runs use in-memory checkpoints and cannot be reopened in
the app like saved UI measurements.

## Outputs and traceability

Artifacts are written locally under `runs/`. Their availability depends on how far the run
progressed and which optional steps ran.

| Path | Contents |
| --- | --- |
| `<run_id>/questions.csv` | Exact question panel used in the run |
| `<run_id>/raw/`, `<run_id>/analyzed/` | Batched model answers and structured analyses |
| `<run_id>/answers.csv` | Pooled input for deterministic statistics |
| `<run_id>/discovered_businesses.csv` | Untracked businesses found in answers |
| `<run_id>/events.jsonl`, `<run_id>/usage.jsonl` | Graph audit events and model-call usage |
| `<run_id>/charts/` | Generated statistics and UI charts |
| `<run_id>/report.md` | Audit report, written after approval |
| `<run_id>/client_report.md` | Client report in the brief's language, written after approval |
| `<run_id>/trend.csv` | Exported trend points with model and question-set identifiers |
| `tracking.csv` | Index of approved measurements |
| `checkpoints.sqlite` | Persistent graph state for UI measurements |

## Architecture

LangGraph controls the workflow, bounded sampling and review loops, failure paths and
approval pause. The measured model answers questions; agent calls analyze answers and
support discovery, research, writing and critique. Python code computes all statistics.
CrewAI agent specifications and factories live in `agents/`; the local integration wires
the graph through `nodes_real.py` and the shared LLM clients.

```text
Reviewed questions → Sampling → Metrics ──→ another batch, if needed
                                  │
                                  └→ Research (MCP) → Chart vision → Draft → Skeptic
                                                                     ↑        │
                                                                     └────────┘
                                                            bounded revisions
                                              → Human approval → Reports + tracking
```

Web content is treated as untrusted input. Research checks access rules and flags potential
prompt injection; chart readings remain hypotheses checked against computed figures.
Rejecting approval closes the run without final publication; required-step failures close
it as partial.

| Location | Responsibility |
| --- | --- |
| `app.py`, `src/ai_share_of_voice/ui/` | Gradio UI, translations, run control and results |
| `graph.py`, `nodes_real.py`, `state.py`, `checkpoint.py` | Workflow, node wiring and persistence |
| `schemas/` | Validated shared contracts |
| `stats/`, `tools/` | Metrics, stopping rules, charts, mentions and source tools |
| `agents/`, `research.py`, `writing.py`, `chart_vision.py` | Agent definitions and review pipeline |
| `mcp_server/`, `automation/` | Source server, reports and tracking exports |
| `data/`, `tests/`, `docs/`, `specs/` | Demo inputs, tests, decisions and implementation notes |

Source paths in this table are relative to `src/ai_share_of_voice/` unless stated otherwise.

## Methodology


The numbers are computed by deterministic, unit-tested code in `src/ai_share_of_voice/stats/`.
No language model writes or runs the analysis.

### Sampling

- A reviewed question panel is sent to the model being measured. Use the demo panels
  (`data/questions_es.csv`, `data/questions.csv`), generate one for a new brief, or reuse a
  saved panel. Questions never name a tracked brand, so we measure whether the model brings
  a brand up on its own. Each run saves the questions it actually asks.
- Questions are asked in batches. Every batch asks every question the same number of times
  (a balanced plan), so every question weighs the same in the results.
- Each answer becomes one row per tracked brand (the brief, or `data/competitors.csv` for the demo): mentioned,
  recommended, position among the tracked brands, and tone. How mentions are counted is in
  [decision 8](docs/decisions.md#8-mention-counting-rules).
- After each batch, metrics are recomputed over all answers so far and the stopping rule
  decides whether another batch is worth its cost.

### Metrics

Every metric is a ratio of sums over answers, so every answer weighs the same.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `mention_rate` | answers that mention the brand | all answers |
| `recommendation_rate` | answers that recommend the brand | all answers |
| `top1_rate` | answers where the brand is the first tracked brand | all answers |
| `share_of_voice` | mentions of the brand | mentions of all tracked brands |
| `recommendation_conversion` | answers that recommend the brand | answers that mention it |
| `net_tone` | positive minus negative mentions (from -1 to 1) | mentions with a tone |

`recommendation_conversion` is undefined for a brand that is never mentioned; it is shown as
"n/a", never as 0.

### Confidence intervals (95%)

Answers to the same question are not independent: the same wording tends to bring up the
same brands. We therefore have less information than the number of answers suggests, so
intervals are computed with the question as the unit.

- **Wilson score interval** for the metrics that count successes among answers. It behaves
  well near 0 and 1 and with small samples, but assumes independent answers, so on its own
  it is too narrow.
- **Cluster bootstrap**: whole questions are resampled with replacement (2,000 resamples,
  fixed seed), the metric is recomputed, and the 2.5th and 97.5th percentiles are kept. This
  keeps the correlation between answers to the same question. It collapses to a single point
  when a brand is always or never mentioned.
- **Reported interval**: for the count metrics, the hull of both (from the lower of the two
  lower limits to the higher of the two upper limits), so it is never narrower than either;
  for `share_of_voice` and `net_tone`, the cluster bootstrap. The hull is a conservative
  reporting rule: its coverage has not been checked by simulation.
- All brands use the same question resamples, so they are compared on the same simulated
  panels. Results with fewer than 10 questions are flagged as `few_questions`, because the
  bootstrap is too optimistic there.

### Stopping rule

`should_stop` watches the half-width of the client brand's reported interval after each
batch. The first rule that holds is applied:

1. **Precision reached**: the half-width is at or below the target, after at least
   `min_batches` batches and with at least 10 questions.
2. **Marginal return**: the half-width shrank by less than `marginal_gain_threshold` in each
   of the last `patience` batches. The half-width shrinks roughly with the square root of the
   number of answers (halving it costs about four times as many), so at some point more
   answers stop buying precision.
3. **Budget exhausted**: the next batch, at the average cost of the batches so far, would go
   over the sampling budget. Sampling may use at most `sampling_budget_share` of the run cap
   and always leaves `agent_reserve_usd` for the agents that run after it.
4. **Max batches** reached.

Statistical reasons come first, so the explanation says why the result is good enough, not
only that a limit was hit. Every decision is logged with its reason. Defaults:
`min_batches` 2, `marginal_gain_threshold` 0.005, `patience` 2, `sampling_budget_share` 0.7,
`agent_reserve_usd` 0.20. The precision target and the spending caps are set in
[decision 3](docs/decisions.md#3-model-provider-and-spending-cap).

### Limits

- A result is a dated sample of one model, in one language, with one question set. It is not
  market share and not what real users are shown.
- Intervals treat the questions as a sample of similar questions. They say nothing about
  questions phrased very differently.
- Intervals cover sampling variability only. Errors in detecting mentions, recommendations or
  tone are not included.
- Mentions are matched by name and variants: paraphrases ("the orange delivery app") and
  untracked brands are excluded from the headline metrics. Discovery lists other businesses
  separately; they do not silently change the comparison set (see
  [decision 7](docs/decisions.md#7-untracked-brands-in-answers)).
- No causality is claimed between a cited web source and a recommendation.
- The cost of the next batch is an estimate (an average), not a guarantee.

## Development checks

With the virtual environment activated and development dependencies installed:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=ai_share_of_voice --cov-report=term-missing
```

On Windows without activation, replace `python` with `.\.venv\Scripts\python.exe`.
If `make` is available, `make install` installs development dependencies and `make check`
runs the same lint, formatting and test checks. Tests use simulated clients and fixtures;
API keys are not required.

## Project documentation

- [Glossary](docs/glossary.md) — product terms and interpretation of results.
- [Decisions](docs/decisions.md) — methodology and team decisions.
- [Project plan](docs/PROJECT_PLAN.md) — phases, owners and tasks.
- [Local integration spec](specs/local-integration.md) — branch scope, implementation log and open items.
- [Contributing](docs/CONTRIBUTING.md) — workflow, review rules and local hooks.
