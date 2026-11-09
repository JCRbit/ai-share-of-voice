# AI Share of Voice

Measures how often a brand appears and is recommended against its competitors in an AI assistant's answers, and doubts its own results before delivering them.

Project III, Module V (AI Engineering) — AgentSquad: multi-agent system with CrewAI, LangGraph and MCP.

> Every result is a dated sample. It is never presented as market share, and no causality is claimed between a web source and a recommendation.

## Status

C1 shared contracts and the A1 LangGraph skeleton are implemented; the graph still uses simulated nodes for several steps. D1 metrics and intervals, the D2 stopping policy, and the D3 statistics tool and chart functions exist at module level, but the real stopping policy and statistics calculations are not yet wired into the end-to-end graph/report path. Supporting modules for mention extraction, agent specifications, failure handling, and reporting are also present, but they are not all connected to a real end-to-end run.

Checkpointing is configurable. A persistent checkpointer is needed to resume a human pause across process restarts. See [`docs/architecture.md`](docs/architecture.md) for module-level status and [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for phases, owners and tasks.


## Installation

These steps install the project and check that it works. They assume a machine with nothing
installed; if a step fails, see [Troubleshooting](#troubleshooting).

### Requirements

- **Python 3.12 or 3.13.** Python 3.14 is not supported yet because CrewAI does not support it, and
  the project refuses to install on it.
- **Git.**
- `make` is optional: every step below has a direct command. GitHub CLI (`gh`) is only needed to work
  with issues and pull requests.

Check which Python versions you have:

```powershell
py -0p                      # Windows: lists every installed version
```

```bash
python3.12 --version        # macOS / Linux
```

### 1. Get the code

```bash
git clone https://github.com/JCRbit/ai-share-of-voice.git
cd ai-share-of-voice
```

### 2. Create the virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

The prompt now starts with `(.venv)`. Check the version inside the environment:

```bash
python --version            # must say 3.12.x or 3.13.x
```

### 3. Install the dependencies

```bash
pip install -r requirements-dev.txt
pip install -e .
pip check                   # must say: No broken requirements found.
```

The first install takes several minutes because CrewAI brings many packages. `pip install -e .`
makes the `ai_share_of_voice` package importable, which `mock_run.py` needs. On macOS, Linux or
any system with `make`, `make install` runs the first command only, so still run `pip install -e .`.

### 4. Create your configuration file

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Leave `LLM_MODE=simulated`. In that mode no API key is needed and nothing is spent. When the team
enables the live mode, your own key and your own spending cap go in `.env`; see the variables listed in
`.env.example`. **Never commit `.env`**; Git already ignores it.

### 5. Install the Git hooks (only if you will contribute)

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

The hooks run `ruff` when you commit, check that the commit message follows
[Conventional Commits](https://www.conventionalcommits.org/), and run the tests when you push.

### 6. Check that everything works

PowerShell:

```powershell
ruff check . ; ruff format --check . ; pytest -q
python mock_run.py
```

macOS / Linux, or any system with `make`:

```bash
make check
python mock_run.py
```

You should see:

- `All checks passed!` from `ruff check`, and no file listed by `ruff format --check`.
- A last line from `pytest` that says `passed` and has no `failed` or `error`.
- `python mock_run.py` prints a series of JSON events and ends with `status=completed`. It runs the
  whole graph with simulated nodes, so it needs no key and spends nothing. The run files are written
  to the `runs/` folder, which Git ignores.

### 7. Run the graph with real stages (optional, spends money)

Task C2 swaps the simulated nodes for real ones one stage at a time, always in this order: `client` (the measured model, the mention analysis, the real statistics and stopping rule), then `mcp` (the sources the answers cite, checked with the sources service), then `vision` (our own chart and the saved screenshots, read by the vision model):

```bash
python scripts/real_run.py                           # stage client, run cap 1.00 USD
python scripts/real_run.py --stages client,mcp
python scripts/real_run.py --stages client,mcp,vision --cap 2
```

It needs `LLM_MODE=live` and your key in `.env`; the vision stage also needs `VISION_MODEL` and its prices. It refuses to run with the simulated client. Every model call counts against the run cap and the person cap and is written to `runs/llm_calls.jsonl`. The run pauses at the approval step and asks for your decision unless `--yes` or `--no` is supplied. The data of the run is in `runs/<run_id>/` (`brief.json`, `raw_answers.jsonl`, `analyzed_answers.jsonl`, `answers.csv`, `metrics.csv`, `charts/`). The question panel is fixed for food delivery in Madrid (`data/questions_es.csv`). The writer is still simulated, so the approval covers a simulated draft and no report is written.

### 8. Open the interface and read the results of a run (optional)

```bash
python app.py
```

Open the address it prints (it listens on `127.0.0.1` only). The **Brief** tab validates the form and starts a simulated run, auto-approved; **Activity** lists its events. The **Results** tab (task D4) reads any run folder in `runs/` that has an `answers.csv`, for example one made by `scripts/real_run.py`:

- a table with every metric of every brand, its estimate and its 95% interval, from the vetted statistics of `stats/`;
- one chart per metric, and the downloads: the report and the tracking CSV once the run was approved, the answers and metrics tables, the charts and the event log.

The run of the Brief tab is simulated and saves no analyzed answers, so its Results say so. Every figure is a dated sample of one model's answers, not a market share, and with fewer than ten questions the tab warns that the intervals are too optimistic. The CSV files are offered as copies in `runs/<run_id>/downloads/` whose text cells cannot start a formula in a spreadsheet; the files the code reads back are not changed. To preview the tab without spending anything, copy `tests/fixtures/sample_answers.csv` to `runs/preview/answers.csv` and reload (three questions, so it shows the warning).

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `pip` refuses to install `crewai`, or says no matching distribution | You are using Python 3.14 | Install Python 3.12, delete `.venv` and repeat step 2 |
| `python --version` shows 3.14 inside the environment | The environment was created with the wrong Python | Delete `.venv` and create it again with `py -3.12 -m venv .venv` |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate again |
| `ModuleNotFoundError: ai_share_of_voice` when running `python mock_run.py` | The package is not installed | Run `pip install -e .` with the environment active |
| `make: command not found` on Windows | `make` is not installed | Use the direct commands of step 6, or run `winget install ezwinports.make` and restart the terminal |
| `pip check` reports conflicts | A leftover environment from another Python or an older install | Delete `.venv` and start again from step 2 |
| A commit is rejected by the hooks | The message does not follow Conventional Commits, or `ruff` reformatted a file | Use a message like `docs: fix a typo`; if `ruff` changed files, run `git add` again and commit again |
| `git push` is rejected by the hooks | The tests fail | Run `pytest -q`, fix the failure and push again |
| The prompt has no `(.venv)` | The environment is not active | Run the activation command of step 2 again |


## Architecture

LangGraph defines the order of work, conditional routes, bounded loops and the human approval step. CrewAI agent specifications define roles, goals, backstories, tools and iteration limits. The graph topology places human approval before finalization. The real report and tracking CSV writer must be injected explicitly; `ui/mock_run.py` uses simulated nodes and does not activate that writer by default. Run events are logged throughout execution, including before approval.

The diagram shows the intended flow, not a fully integrated real run. D1–D3 modules exist, while the graph still uses simulated metrics and stopping behavior. The graph topology defines the Skeptic review loop, and tests exercise the real implementation when it is explicitly injected. However, `ui/mock_run.py` builds the graph with simulated nodes. Defining a CrewAI agent specification does not mean that the agent or its tools execute in the graph. See [`docs/architecture.md`](docs/architecture.md) for the implementation status.

```mermaid
flowchart LR
    A[Brief<br/>validated form] --> B[questions]
    B --> C[sampling<br/>model client: planned integration]
    C --> D[metrics<br/>D1 implemented; graph integration pending]
    D --> E{stop?<br/>D2 policy implemented; integration pending}
    E -- no, another batch --> C
    E -- yes --> F[research<br/>source tools integration pending]
    F --> G[screenshots<br/>vision: planned]
    G --> H[write_draft]
    H --> I[skeptic<br/>simulated in ui/mock_run.py]
    I -- objections, at most 2 rounds --> H
    I -- none, or rounds used up --> J[approval_request]
    J --> K{{human pause<br/>approval_gate}}
    K -- approved --> L["finalize: real writer when explicitly injected"]
    K -- rejected --> M[close_rejected]
    B & C & D & F & G & H & I & J & L -. eligible failure .-> N[close_partial]
```

- How it is built and why (hybrid process, agents, tools, failure paths, status): [`docs/architecture.md`](docs/architecture.md)
- What is not delegated, risks and controls, privacy and auditing: [`docs/ethics.md`](docs/ethics.md)

## Methodology

The statistical functions in `src/ai_share_of_voice/stats/`, the stopping policy in `stats/stopping.py`, and the statistics-tool/chart modules are implemented. However, the current graph still uses simulated core nodes: its metrics node does not yet call the real statistics tool, and its stopping decision does not yet use the real stopping policy. The methodology below describes the intended measurement process; consult [`docs/architecture.md`](docs/architecture.md) before treating every step as an active runtime safeguard.


### Sampling

- A fixed set of questions (`data/questions_es.csv`, `data/questions.csv`) is sent to the model being measured. Questions are intended not to name a brand, so the process can measure whether a brand is brought up without being prompted by name.
- Questions are asked in batches. The balanced plan asks every question the same number of times, so every question weighs the same in the results.
- Each answer can be represented as one row per tracked brand (`data/competitors.csv`): mentioned, recommended, position among tracked brands, and tone. How mentions are counted is in [decision 8](docs/decisions.md#8-mention-counting-rules).
- In the intended real flow, metrics are recomputed over the answers collected so far and the stopping rule decides whether another batch is worth its cost. The metric functions and stopping policy exist, but the current graph uses simulated decisions rather than this complete integrated path.

### Metrics

Each metric is a ratio of sums, but its denominator differs: mention and recommendation rates use answers, share of voice uses tracked mentions, recommendation conversion uses mentions, and net tone uses mentions with a tone. The balanced sampling plan gives every question the same number of answers; it does not make every metric an equal-weight average over answers.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `mention_rate` | answers that mention the brand | all answers |
| `recommendation_rate` | answers that recommend the brand | all answers |
| `top1_rate` | answers where the brand is the first tracked brand | all answers |
| `share_of_voice` | mentions of the brand | mentions of all tracked brands |
| `recommendation_conversion` | answers that recommend the brand | answers that mention it |
| `net_tone` | positive minus negative mentions (from -1 to 1) | mentions with a tone |

`recommendation_conversion` is undefined for a brand that is never mentioned; it is shown as "n/a", never as 0.

### Confidence intervals (95%)

Answers to the same question are not independent: the same wording tends to bring up the same brands. We therefore have less information than the number of answers suggests, so intervals are computed with the question as the unit.

- **Wilson score interval** for binary-success ratios with a defined denominator (for example, mention rate or recommendation conversion). It behaves well near 0 and 1 and with small samples, but assumes independent trials, so on its own it may be too narrow when answers to the same question are correlated.
- **Cluster bootstrap**: whole questions are resampled with replacement (2,000 resamples, fixed seed), the metric is recomputed, and the 2.5th and 97.5th percentiles are kept. This keeps the correlation between answers to the same question. For defined metrics such as mention rate, the interval can collapse to a single point when a brand is always or never mentioned; metrics with no eligible denominator are undefined.
- **Reported interval**: for the count metrics, the hull of both (from the lower of the two lower limits to the higher of the two upper limits), so it is never narrower than either; for `share_of_voice` and `net_tone`, the cluster bootstrap. The hull is a conservative reporting rule: its coverage has not been checked by simulation.
- All brands use the same question resamples, so they are compared on the same simulated panels. Results with fewer than 10 questions are flagged as `few_questions`, because the bootstrap can be too optimistic there.

These intervals quantify uncertainty under the chosen resampling assumptions. They should not be described as calibrated 95% coverage unless that coverage has been validated.

### Stopping rule

The implemented `should_stop` policy in `stats/stopping.py` watches the half-width of the client brand's reported interval after each batch. The current graph still uses a simulated stopping decision; the rules below describe the policy that should be applied when the real policy is connected to the runtime:

1. **Precision reached**: the half-width is at or below the target, after at least `min_batches` batches and with at least 10 questions.
2. **Marginal return**: the half-width shrank by less than `marginal_gain_threshold` in each of the last `patience` batches. The half-width often shrinks roughly with the square root of the number of answers (halving it may cost about four times as many), so at some point more answers stop buying much precision.
3. **Budget exhausted**: the next batch, at the average cost of the batches so far, would go over the sampling budget. Sampling is intended to use at most `sampling_budget_share` of the run cap and leave `agent_reserve_usd` for later agent calls.
4. **Max batches** reached.

Statistical reasons come first, so the explanation says why the result is good enough, not only that a limit was hit. The policy is designed to record each decision with its reason; the current graph's simulated stopping decision does not execute this policy. Defaults: `min_batches` 2, `marginal_gain_threshold` 0.005, `patience` 2, `sampling_budget_share` 0.7, `agent_reserve_usd` 0.20. The precision target and spending caps are set in [decision 3](docs/decisions.md#3-model-provider-and-spending-cap). The average cost of previous batches is an estimate, not a guarantee.

### Limits

- A result is a dated sample of one model, in one language, with one question set. It is not market share and not what real users are shown.
- Intervals treat the questions as a sample of similar questions. They say nothing about questions phrased very differently.
- Intervals cover sampling variability only. Errors in detecting mentions, recommendations or tone are not included.
- Mentions are matched by name and variants: paraphrases ("the orange delivery app") and brands outside `data/competitors.csv` can be missed (see [decision 7](docs/decisions.md#7-untracked-brands-in-answers)).
- No causality is claimed between a cited web source and a recommendation.
- A model API response may differ from what a person sees in an assistant app, which may add web search or other settings.
- Spending limits protect only calls that pass through a client that enforces the cap. CrewAI agents built with `build_budgeted_agents` (`agents/factory.py`) are routed through that client by an adapter (task A2b); agents given any other `llm`, such as a model name string, call the provider by themselves and are not covered. The adapter was written against the CrewAI 1.15 documentation and has not been run with a live CrewAI by its author.

## Sections (owners)
- Architecture, process justification and ethics — J
- Configuration and usage — An

## Contributing

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
