# AI Share of Voice

Measures how often a brand appears and is recommended against its competitors in an AI
assistant's answers, and doubts its own results before delivering them.

Project III, Module V (AI Engineering) — AgentSquad: multi-agent system with CrewAI, LangGraph and MCP.

> Every result is a dated sample. It is never presented as market share, and no causality is
> claimed between a web source and a recommendation.

## Status
C1 shared contracts and the A1 LangGraph skeleton are implemented. The A1 graph uses simulated nodes, bounded loops, SQLite checkpointing, human approval and JSONL audit events. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for phases, owners and tasks.

## Quick start

**Requirements:** Git, Python 3.12 (3.13 also works; CrewAI does not support 3.14 yet) and `make`.

**Linux / macOS**

```bash
git clone https://github.com/JCRbit/ai-share-of-voice.git
cd ai-share-of-voice
python3.12 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env        # add your own key and spending cap
make check
```

**Windows (PowerShell)**

```powershell
winget install ezwinports.make   # skip if you already have make; then restart your terminal/editor
git clone https://github.com/JCRbit/ai-share-of-voice.git
cd ai-share-of-voice
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1       # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
make install
Copy-Item .env.example .env      # add your own key and spending cap
make check
```

`make check` runs lint and tests; if it ends with all tests passing, your setup is ready.

## Methodology

The numbers are computed by deterministic, unit-tested code in `src/ai_share_of_voice/stats/`.
No language model writes or runs the analysis.

### Sampling

- A fixed set of questions (`data/questions_es.csv`, `data/questions.csv`) is sent to the
  model being measured. Questions never name a brand, so we measure whether the model brings
  a brand up on its own.
- Questions are asked in batches. Every batch asks every question the same number of times
  (a balanced plan), so every question weighs the same in the results.
- Each answer becomes one row per tracked brand (`data/competitors.csv`): mentioned,
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
  brands outside `data/competitors.csv` are missed (see
  [decision 7](docs/decisions.md#7-untracked-brands-in-answers)).
- No causality is claimed between a cited web source and a recommendation.
- The cost of the next batch is an estimate (an average), not a guarantee.

## Sections (owners)
- Installation — Ab
- Methodology — E
- Architecture, process justification and ethics — J
- Configuration and usage — An

## Contributing
See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
