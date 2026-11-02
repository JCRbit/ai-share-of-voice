# Ethics and governance

This document explains what the system does not decide on its own, which risks come with autonomous agents and how they are intended to be controlled, how sources and models are used, and how a third party can inspect a run. The technical design and implementation status are described in [architecture.md](architecture.md).

## What the system measures, and what it does not

AI Share of Voice measures how often an AI assistant mentions and recommends a brand compared with its competitors in a dated sample of answers.

- It is **not market share**, sales, or customer preference.
- It does **not** show why an assistant answers the way it does. A source or an image may provide context, but it is not treated as proof of the cause of a recommendation.
- A mention is not necessarily a recommendation.
- Results come from the model endpoint actually queried. They can differ from what a person sees in an assistant's app, which may add web search, personalization, or other settings.
- Every report should state that it represents a dated sample and identify the model, question set, language, and collection period where those details are available.

The system measures responses under a defined configuration. It does not establish how all users experience AI assistants or whether a brand's visibility causes changes in consumer behavior.

## Decisions that are not delegated to the agents

| Decision | Who decides | Control / current limitation |
| --- | --- | --- |
| Approve the report and tracking file | A person | The graph topology has an approval pause before `finalize`. The real `make_finalize_node` requires explicit approval and writes the final report and tracking CSV when injected; the mock-run path uses simulated nodes. This protects those final outputs in the real path, not event logs or every file written during execution. |
| Publish or send anything | A person, outside the system | No agent is specified with a tool to publish or send. External publication and communication remain outside the described workflow. |
| Claim that a source or image caused a recommendation | Nobody: the available evidence cannot establish this | Instructions and Skeptic checks discourage unsupported causal claims. The checks are bounded heuristics and may miss cases. |
| Present visibility as market share | Nobody | The documentation states that results are not market share. Deterministic checks can flag some problematic wording but cannot guarantee that every misleading claim is detected. |
| Treat an image interpretation as an established fact | A person | Visual findings should be treated as hypotheses requiring appropriate verification. The vision integration remains planned and must not be represented as an active safeguard until implemented and tested. |
| Decide whether a source may be used under its terms | A person | The source tools `search_sources`, `read_source`, and `check_url` remain planned. URL availability or a future `robots.txt` check would not, by itself, establish legal permission or compliance with all terms. |
| Spend beyond the configured cap | Configuration owner / operator, subject to runtime enforcement | The `ModelClient` contract defines spending-cap behavior and `BudgetExceeded`. This does not prove that all provider calls are covered. CrewAI calls are not guaranteed to enforce the cap until a suitable integration routes them through the enforcing client and that path is tested. |

Rows that depend on integrations not yet built, including the source tools and vision integration, describe intended design rather than current runtime controls. See [architecture.md](architecture.md).

The distinction between an implemented component and an active end-to-end control is intentional. A class, schema, agent specification, or isolated test does not establish that the production workflow invokes that component in every relevant run.

## Risks of autonomy and intended controls

| Risk | Control / limitation |
| --- | --- |
| **Hallucinated figures** | Deterministic Skeptic checks compare detected figures with available run results. These checks are heuristic and cannot guarantee detection of every incorrect figure, unsupported comparison, or calculation. |
| **Invented sources** | The Skeptic implementation can check source references against recorded results. The source tools `search_sources`, `read_source`, and `check_url` remain planned. A recorded URL or `[S#]` marker is not proof that the source exists, is reliable, or supports the associated claim. |
| **Prompt injection through web content** | Retrieved content must be treated as untrusted input. Source-researcher access is intended to be restricted to the tools it needs; the Skeptic is read-only and has no web access. Agent specifications limit tools, delegation, and iterations, but do not constitute a complete prompt-injection defense. The source-research integration remains planned. |
| **Excessive agency** | The graph controls the intended workflow, and agent specifications constrain tool lists, delegation, and iterations. The human approval step protects final report and tracking CSV writes, not all files, event logging, or every action performed during execution. |
| **Runaway cost** | Intended controls include the model-client spending cap, batch limits, agent iteration limits, a graph recursion limit, and a bounded review loop. The cap applies only to calls routed through a client that actually enforces it. CrewAI integration remains a known gap until the full call path is covered and tested. |
| **Silent failures** | Eligible exceptions from guarded graph nodes are logged and routed to a partial close. Some exceptions are re-raised, and some nodes are not guarded by default. A partial close is therefore not guaranteed for every failure. |
| **Misleading precision** | Statistical intervals accompany estimates where available. The conservative envelope of two interval methods is not claimed to have calibrated 95% coverage. The statistics module can flag results based on few questions, but intervals do not account for every source of measurement error or bias. |
| **Spreadsheet formula injection** | The tracking CSV writer neutralizes values beginning with formula-sensitive characters. This mitigates a known risk but does not guarantee protection against every spreadsheet-specific behavior or downstream processing tool. |

The risk assessment follows the OWASP Top 10 for LLM Applications as a reference, particularly for prompt injection and excessive agency. The controls described here should be understood in light of the implementation status and limitations stated in [architecture.md](architecture.md).

## Sources, terms of use and privacy

- **Model use:** The design calls a model through its configured API endpoint. The actual provider, model, logging, retention, and data-use terms depend on the endpoint and account configuration. The system is not intended to scrape an assistant's web interface.
- **Web sources:** Source retrieval tools remain planned. Public availability does not automatically mean content is accurate or may be reused. Any future `robots.txt` handling would be only one consideration, not a legal determination.
- **Screenshots:** The intended workflow records the date and URL associated with competitor-page screenshots. The vision integration remains planned, so image-based findings must not be described as an active production capability until the relevant path is implemented and verified.
- **Personal information:** The brief is designed to collect business information such as company name, sector, and city rather than personal data. Public pages, model responses, or screenshots may nevertheless contain personal information incidentally. Assess this risk and avoid or remove unnecessary data.
- **Secrets:** API keys should remain out of the repository and be stored in each operator's local `.env` file, which must not be committed. Run files under `runs/` are intended to be ignored by Git; verify the repository's ignore rules and local handling before using real data.
- **Spending:** Each operator should use an appropriately scoped key and configure a suitable cap. The cap is effective only for calls routed through a client that enforces it. Explicitly configuring a model or API key does not, on its own, guarantee that every model call is subject to the cap.

## How a third party can audit a run

1. Inspect `runs/<run_id>/events.jsonl` for events emitted during execution, including node, agent, tool, result, failure reason, and reported cumulative spend where those fields are provided.
2. Inspect the report for figures, source listings, visual findings, and the run log. The log supports review but may be incomplete after abrupt termination. Finalization's success event is emitted after output writing and therefore may not appear in a report created before that event.
3. Inspect `runs/tracking.csv` for recorded figures, stopping reason, and counts where those fields are available.
4. For a partial run, inspect the event log and the final state's `partial_reason` when available.
5. Run the test suite with `pytest` to exercise the documented contracts, approval flow, failure handling, Skeptic behavior, and reporting behavior. Passing tests validate the cases they cover, not all possible runtime behavior.
6. Review the pull-request history for the rationale, implementation decisions, and review of code changes.
7. Compare the active graph implementation with the relevant modules and tests to establish which capabilities were actually invoked. In particular, verify the graph's metrics, stopping, source, vision, and budget-enforcement paths rather than inferring integration from the existence of a module or specification.

The event log and report support traceability. They do not guarantee exact reproduction of outputs from an external model or prove that every cited source supports every claim.

An audit should distinguish simulated results from actual external operations. A logged event is evidence that the application recorded an event; it is not necessarily proof that an external service performed the corresponding operation.

## Limits we state openly

- **Sample size and variability:** The sample is small and dated, and model answers can vary between runs. Conclusions should be limited to the models, questions, language, competitors, configuration, and collection period studied.
- **Mention-extraction accuracy:** The accuracy of the mention extractor is intended to be measured against 30 manually labelled answers. This measurement remains pending until the evaluation has been completed and its results recorded. Do not present the intended evaluation as a measured accuracy result.
- **D1 — metrics:** Brand-metric and confidence-interval functions are implemented as modules. The graph metrics node remains simulated, and the graph/report path does not yet integrate these calculations into the final report. A delivered run must not be described as using the real D1 calculations until the relevant integration is complete and verified.
- **D2 — stopping policy:** The deterministic stopping policy is implemented in `stats/stopping.py`. The graph still uses a simulated stopping decision, so the existence of the policy module does not establish that the delivered workflow actually applies its decision.
- **D3 — statistics and charts:** The statistics-tool and chart functionality is implemented in modules, including a constrained metric menu and chart-generation functionality. The graph/report integration remains incomplete. In particular, do not claim that the final report uses these calculations or convergence charts until the active graph and reporting path invoke them and the integration is tested.
- **Simulated nodes:** Simulated nodes stand in for real integrations until the corresponding production path is implemented and verified. A successful simulated run is not proof of a successful full real run.
- **Skeptic checks:** The real implementation in `agents/skeptic_checks.py` and `agents/skeptic.py` performs deterministic checks and supports optional model-assisted review. The graph topology supports the review loop, and tests exercise the real node when it is explicitly injected. The `mock_run.py` path uses `make_fake_nodes`, so a successful simulated run does not establish that the real Skeptic was used. These checks are bounded and cannot guarantee that every unsupported claim is detected or that a report is correct, unbiased, or free of unsupported claims.
- **Source tools:** `search_sources`, `read_source`, and `check_url` remain planned. Do not claim that the current workflow performs source retrieval, page reading, or URL verification unless those operations have subsequently been implemented, integrated, and tested.
- **Vision integration:** Image-based analysis remains planned. Do not present visual findings as production-verified facts or claim that the workflow currently provides a functioning vision-based evidence check.
- **Spending cap:** The model-client contract defines cap-related behavior, but model calls that bypass the enforcing client may not be covered. End-to-end enforcement must be verified across all applicable model-call paths.
- **Approval scope:** The real finalization path requires human approval before writing the final report and tracking CSV, when `make_finalize_node` is explicitly injected. The `mock_run.py` path uses simulated nodes. Approval does not imply that event persistence or other execution-time file operations wait for approval.
- **Statistical interpretation:** Confidence intervals do not automatically account for extraction errors, classification errors, question-selection bias, or systematic differences between model outputs and real user experiences. The conservative envelope of interval methods is not claimed to provide calibrated 95% coverage.
- **Reproducibility:** Logs and recorded configuration support inspection, but external model responses may not be exactly reproducible because providers, models, and outputs can change.

## Governance principle

The system should report only what its evidence supports, preserve meaningful objections and failures, and make the difference between implemented, integrated, tested, and simulated behavior explicit.

Human approval, deterministic checks, statistical intervals, source records, and spending controls each address different risks. None should be presented as a complete guarantee of correctness or safety.

For component details and implementation status, see [architecture.md](architecture.md). For usage instructions and the measurement definitions, see [README.md](../README.md).
