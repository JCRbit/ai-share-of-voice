# Architecture

AI Share of Voice measures how often an AI assistant mentions and recommends a brand compared with its competitors, and reviews its own results before delivering them. This document explains how the system is built, why the architecture separates workflow control from agent behavior, and which capabilities are implemented versus integrated into the end-to-end execution path. The ethical side, including delegated decisions, risks, privacy, and auditing, is documented in [ethics.md](ethics.md).

A component being implemented or tested at module level does not, by itself, establish that the main LangGraph execution path invokes it. This document distinguishes implementation status, graph integration, and verified end-to-end behavior.

## The graph

LangGraph defines the sequence of steps, conditional routes, bounded loops, human approval pause, and failure routing. The diagram describes the intended end-to-end workflow. It does not imply that every node currently performs a real external operation.

D1 brand metrics, D2 deterministic stopping policy, and D3 statistics-tool and chart modules are implemented. The graph/report integration for these components must be verified separately. The core workflow still uses simulated behavior where real integrations have not been completed.

    ```mermaid
    flowchart LR
        A[Brief<br/>validated form] --> B[questions]
        B --> C[sampling<br/>real model client integration]
        C --> D[metrics<br/>D1 module implemented]
        D --> E{stop?<br/>D2 policy implemented}
        E -- no, another batch --> C
        E -- yes --> F[research<br/>source tools planned]
        F --> G[screenshots<br/>vision planned]
        G --> H[write_draft]
        H --> I[skeptic<br/>deterministic checks and optional model review]
        I -- objections, at most 2 rounds --> H
        I -- no objections or review limit reached --> J[approval_request]
        J --> K{{human pause<br/>approval_gate}}
        K -- approved --> L["finalize: real writer when explicitly injected"]
        K -- rejected --> M[close_rejected]
        B & C & D & F & G & H & I & J & L -. eligible failure .-> N[close_partial]
    ```

- **Bounded loops.** The graph has a batch limit in its run configuration, a review-round limit, and a recursion limit. The deterministic D2 policy exists in `stats/stopping.py`, but it must be wired into the active graph before it can be treated as a runtime stopping safeguard.
- **Approval protects final outputs.** The graph topology routes through human approval before `finalize`. The real node `make_finalize_node` writes the final report and tracking CSV when explicitly injected; `mock_run.py` uses `make_fake_nodes` and does not activate that real writer by default. The JSONL event log is written throughout execution, including before approval. Approval therefore does not prevent every execution-time write.
- **Failure handling has limits.** Eligible exceptions raised by guarded nodes are recorded and routed to `close_partial`. Some exceptions are deliberately re-raised, and the human pause and selected safety-critical nodes are not guarded by default. Such failures may propagate instead of producing a partial close.
- **Integration status matters.** A successful simulated graph run validates the simulated workflow and its tested contracts; it does not establish that real model calls, source retrieval, image analysis, or statistical calculations have been executed end to end.

## Hybrid process: why LangGraph and CrewAI

| Question | LangGraph | CrewAI |
| --- | --- | --- |
| What it decides here | Order, loops, stops, human pause, and failure routing | Agent role, goal, backstory, tools, and iteration limit |
| Why | The brief calls for explicit, auditable control of workflow and loops | Roles and bounded tool lists provide a contract for each agent |

The design does not delegate graph control to a CrewAI manager. A hierarchical crew may choose its next step using a model, which can be harder to audit and may introduce additional model calls. In this design, LangGraph is intended to determine the workflow, while agents perform the work assigned to each step.

Two implementation limits are particularly important:

- **Agent specifications and factory are implemented; full graph integration must be assessed separately.** The seven agent specifications and factory are defined in `agents/specs.py` and `agents/factory.py`. Their existence does not mean that all seven agents are invoked by the active graph. The graph's core workflow still contains simulated behavior where real integrations have not been completed.
- **Model-client enforcement is not automatically inherited by CrewAI.** The factory accepts an explicit `llm`, but this does not guarantee that every model call passes through `ModelClient` or its spending-cap enforcement. The cap protects only calls routed through a client that actually enforces it. A suitable adapter and verified call-path coverage are required before claiming end-to-end budget enforcement.

## Agents

The following table describes the agent specifications and their intended tool and iteration limits. It does not assert that every agent is active in the production graph.

| Agent | Role | Tools (at most three) | Iterations |
| --- | --- | --- | --- |
| `question_generator` | Question Panel Designer | None | 3 |
| `mention_extractor` | Mention Extraction Analyst | `extract_brand_mentions` | 4 |
| `analyst` | Quantitative Analyst | `stats_tool` | 6 |
| `source_researcher` | Source Researcher | `search_sources`, `read_source`, `check_url` | 8 |
| `visual_analyst` | Visual Analyst | `analyze_image` | 4 |
| `skeptic` | Skeptical Reviewer | `read_results` | 6 |
| `writer` | Strategic Report Writer | `report_template`, `read_results` | 5 |

The specifications and their tests enforce constraints including:

- Agents cannot delegate to other agents.
- Only the source researcher is specified to receive web tools.
- The Skeptic has no web access.
- Only the analyst is specified to use the statistics tool.
- Shared tools must be read-only.
- Tool lists and iteration limits are explicitly constrained.

These constraints reduce the potential for excessive agency, but they do not eliminate prompt injection, incorrect reasoning, or unintended behavior. The active runtime path must also respect the specifications.

## Tools

| Requirement | Tool | Status | Notes |
| --- | --- | --- | --- |
| Statistical calculations | Brand metrics and interval functions in `stats/` | Implemented (D1) | Module-level implementation exists; integration with graph metrics and final reporting must be verified separately. |
| Deterministic stopping | `stats/stopping.py` | Implemented (D2) | The policy exists at module level; the graph still uses simulated stopping behavior until integration is complete. |
| Statistics tool and charts | `tools/stats_tool.py`, `stats/charts.py` | Implemented (D3) | Tool and chart functions exist; integration with the agent, graph, and final report remains a separate requirement. |
| Own tool with a validated schema | `extract_brand_mentions` | Implemented | Pydantic input schema; deterministic code performs mention matching, while model-assisted classification may assess recommendation and tone. |
| Skeptic checks | `agents/skeptic_checks.py`, `agents/skeptic.py` | Implemented | Deterministic checks and optional model-assisted review exist. Full graph invocation and end-to-end behavior must be verified separately. |
| MCP source server | `search_sources`, `read_source`, `check_url` | Planned (I4) | Intended to search for sources, read page content, and check URLs. Do not claim these operations are active until the server and runtime integration are implemented and tested. |
| Multimodal analysis | `analyze_image` | Planned (I5) | Intended to analyze screenshots. Findings should be treated as hypotheses, not proof of causation or independently verified facts. |

A URL check, when implemented, will not establish that a source is accurate, authentic, legally reusable, or suitable evidence for a particular claim. Any future `robots.txt` handling must not be confused with a complete legal or terms-of-use assessment.

The presence of a tool schema or agent specification does not prove that the tool has a functional implementation, is registered with the runtime, or is called by the graph.

## State, contracts and traceability

- **Typed contracts.** Pydantic contracts define structures for the brief, questions, answers and their analysis, stopping decisions, graph events, model usage, source tools, and vision findings. Contracts define expected data shapes; they do not by themselves perform the operations those shapes represent.
- **Shared graph state.** `state.py` is intentionally small and holds contracts, counters, and paths. Larger artifacts belong in the run folder where applicable.
- **Event log.** The JSONL log at `runs/<run_id>/events.jsonl` records emitted graph events as execution progresses. Depending on the node, events may include node, agent, tool, result, failure reason, and reported cumulative spend. This supports inspection but does not guarantee completeness after abrupt termination or exact reproducibility of external model responses.
- **Checkpoints.** The graph can be compiled with a checkpointer, such as an in-memory or SQLite implementation, so a paused run can resume. Persistence depends on configuring and managing a suitable checkpointer. `build_graph` accepts `None`, so persistence is not enforced by the function itself.
- **Run isolation.** Run identifiers and output paths should be validated so that a run cannot unintentionally write to another run's files. Audit events and artifacts should be associated with the correct run.
- **Simulated versus real events.** A simulated event must not be described as evidence that an external model, source service, or vision service actually performed an operation.

## Failure paths

| Situation | Exception / condition | `failure_kind` |
| --- | --- | --- |
| A tool fails | `ToolFailure` | `tool_failure` |
| There is nothing usable to report | `NoEvidence` | `no_evidence` |
| The spending cap is reached | `BudgetExceeded` | `budget_exhausted` |
| Another eligible exception is caught by the guard | Other eligible exception | `unexpected` |

The guard around eligible nodes records a `FAILED` event and sets partial-run state. Certain exceptions are re-raised, and the human pause and selected safety-critical nodes are not guarded by default. A partial close is therefore not guaranteed for every possible failure.

A spending-cap failure is handled only if the invoked client actually enforces the cap and raises `BudgetExceeded`. Merely defining this exception or adding a budget field to the event schema does not guarantee that real calls are constrained.

Failure-handling tests validate the cases they exercise. They do not prove that every possible exception, process termination, external timeout, or persistence failure results in a complete audit trail.

## Statistics

### D1 — brand metrics and intervals

The D1 statistics module implements brand metrics and uncertainty calculations, including Wilson intervals and question-level cluster bootstrap intervals.

For applicable count metrics, the reported interval may use the envelope of a Wilson interval and a question-level bootstrap interval. `share_of_voice` and `net_tone` use the cluster bootstrap as defined by the statistics implementation.

The envelope is not presented as a calibrated 95% confidence interval because its coverage has not been established through simulation. Statistical intervals also do not automatically account for mention-extraction errors, recommendation or tone misclassification, question-selection bias, or systematic differences between model outputs and real user experiences.

The existence of these functions does not establish that the graph computes the metrics during a run or that the final report contains the resulting values. That requires graph and report integration.

### D2 — deterministic stopping policy

The D2 policy is implemented in `stats/stopping.py`. Its decision logic considers the configured stopping criteria, including precision, marginal gain, budget, and maximum batch limits.

The graph still uses simulated stopping behavior until the policy is integrated into the active graph. As a result, a simulated stopping decision must not be described as a decision produced by the real D2 policy.

Integration tests should verify that the graph passes the correct accumulated statistics and budget information to the policy, respects its result, and terminates or schedules another batch as intended.

### D3 — statistics tool and charts

The D3 statistics tool and chart functions are implemented in `tools/stats_tool.py` and `stats/charts.py`. The intended tool exposes constrained statistical operations rather than arbitrary model-generated code.

The agent, graph, and reporting integration remains a separate requirement. Until that integration is complete and verified, do not claim that the active workflow uses the D3 tool or includes charts generated by it in the final report.

Tests of individual metric or chart functions establish behavior only for the tested inputs and cases. They do not prove that a production run calls those functions or that the resulting output is correctly represented in the report.

## Skeptic review

The Skeptic implementation includes deterministic checks in `agents/skeptic_checks.py` and the review logic in `agents/skeptic.py`.

The deterministic checks are intended to identify problems such as:

- Figures that cannot be reconciled with the available run record.
- Source references that do not match recorded results.
- Invalid or unsupported `[S#]` markers.
- Unsupported causal language.
- Wording that presents AI visibility as market share.

The Skeptic can also perform optional model-assisted review. Deterministic checks and model-assisted review serve different purposes: explicit rules can catch known patterns consistently, while model review may identify concerns that are not covered by those rules. Both can miss errors, and model review can introduce false positives or false negatives.

The checks are bounded heuristics, not a formal proof that a report is correct, unbiased, or fully supported by evidence. An objection should be retained, addressed, or explicitly qualified rather than silently discarded.

The graph topology supports the Skeptic review loop, and tests explicitly inject `make_skeptic_node` to exercise the real implementation and objection routing. However, `mock_run.py` builds the graph with `make_fake_nodes`, so that entry point uses the simulated Skeptic. Verify each application entry point separately before claiming that the real Skeptic runs end to end.

## Module map

| Module | Purpose |
| --- | --- |
| `schemas/` | Shared typed contracts |
| `state.py`, `graph.py`, `nodes.py` | Shared state, workflow definition, routing, and graph nodes, including simulated core behavior |
| `guard.py`, `errors.py` | Failure classification and partial-close handling |
| `event_log.py`, `checkpoint.py` | Event logging and checkpoint configuration |
| `stats/` | D1 brand metrics and intervals, D2 deterministic stopping policy, and chart functions |
| `tools/` | Mention-extraction tool and D3 statistics tool |
| `agents/` | Agent specifications, CrewAI factory, and Skeptic implementation |
| `automation/` | Report generation, tracking CSV, and finalization |
| `mock_run.py` | Runs the simulated graph end to end |

Module names and responsibilities should be checked against the current repository when this document is updated. The table describes the intended organization and does not imply that every component is wired into the production graph.

## Implementation status

The following table distinguishes implemented modules from end-to-end integration. A task's module-level implementation status should not be interpreted as proof of production integration.

| Piece | Task | Status |
| --- | --- | --- |
| Typed contracts, graph with simulated nodes, event log, and optional checkpointer support | C1, A1 | Implemented |
| Failure paths and partial-close handling | A6 | Implemented, with documented exception boundaries |
| Brand metrics and intervals | D1 | Implemented in `stats/`; graph metrics node and final-report integration pending |
| Mention-extraction tool | A3 | Implemented |
| Seven agent specifications and CrewAI factory | A2 | Implemented; full activation of agents in the graph must be verified |
| Skeptic deterministic checks and optional model-assisted review | A5 | Implemented; complete graph wiring and end-to-end behavior must be verified |
| Report, tracking CSV, and finalization | A7 | Implemented; the real end-to-end path depends on graph integration and explicit approval |
| Deterministic stopping policy | D2 | Implemented in `stats/stopping.py`; graph currently uses simulated stopping behavior, integration pending |
| Statistics tool and chart functions | D3 | Implemented in `tools/stats_tool.py` and `stats/charts.py`; agent, graph, and report integration pending |
| Real model client with spending-cap enforcement | I3 | Must be verified against the current model-client implementation and provider integration; end-to-end cap enforcement is not established until every applicable call path is covered |
| MCP source server and vision integration | I4, I5 | Planned |
| Interface: activity and approval tab, results tab | I6, A4, D4 | Planned unless subsequent implementation has been verified |
| First full real run | C2 | Planned until a complete real run has been executed and its outputs inspected |

Update this table when changes are implemented and merged. In particular, distinguish a tested module from a component integrated into the active graph, and distinguish a simulated end-to-end run from a real run involving external services.

## Brief requirements, mapped

| Requirement | Where | Status / qualification |
| --- | --- | --- |
| At least three agents with role, goal, and tools | `agents/specs.py`, `agents/factory.py` | Agent specifications and factory implemented; verify which agents are invoked by the active graph |
| Typed state, conditional edges, bounded loops, and human pause | `state.py`, `graph.py` | Implemented in the graph structure; persistence depends on checkpointer configuration |
| Error handling for tool failure, no evidence, and exhausted budget | `guard.py`, `errors.py` | Failure-handling paths implemented; budget exhaustion requires an enforcing client to raise `BudgetExceeded` |
| Code-execution/statistics tool | `tools/stats_tool.py` | D3 module implemented; agent, graph, and report integration pending |
| Own tool with Pydantic validation | `extract_brand_mentions` | Implemented |
| MCP source integration | Source-tool contracts and planned server | Planned; do not claim live retrieval until implemented and tested |
| Multimodal analysis | `analyze_image` contract and planned integration | Planned |
| Automation with prior approval | `automation/` and graph approval route | Implemented; approval protects final report and tracking CSV writes, not every execution-time write |
| Traceability and cited sources | `events.jsonl`, report appendix, and Skeptic `[S#]` checks | Logging and deterministic checks implemented; source verification remains limited and cannot guarantee that every source supports every claim |
| Statistical metrics and uncertainty | `stats/` | D1 module implemented; graph and report integration pending |
| Deterministic stopping policy | `stats/stopping.py` | D2 module implemented; graph integration pending |
| Statistics and charts | `tools/stats_tool.py`, `stats/charts.py` | D3 modules implemented; agent, graph, and report integration pending |
| Architecture diagram and process justification | This document | Documented |

## Integration and verification checklist

Before describing the system as a fully integrated real workflow, verify the following:

- [ ] The active graph invokes the real model client for sampling and records the actual responses and usage.
- [ ] The model-call paths, including CrewAI calls, route through the spending-cap enforcement mechanism.
- [ ] The graph computes D1 metrics from actual run data and passes the results to the final report.
- [ ] The graph invokes the D2 stopping policy with the correct statistics and budget state.
- [ ] The active analyst/tool path invokes D3 functions and the resulting statistics or charts are included in the report as intended.
- [ ] The active graph invokes the deterministic Skeptic checks and optional model review as configured.
- [ ] Skeptic objections and review-round limits are preserved and handled correctly.
- [ ] Source retrieval, page reading, and URL checks are implemented, registered, and tested before being described as operational.
- [ ] Vision analysis is implemented and tested before visual findings are presented as actual runtime results.
- [ ] Verify that the real finalization path requires explicit human approval before writing the final report and tracking CSV. The simulated `mock_run.py` path does not activate the real writer by default.
- [ ] Partial failures, rejection, and finalization outcomes are recorded and tested.
- [ ] A full real run has been executed, its artifacts inspected, and its behavior distinguished from simulated runs.

The repository's automated tests and lint checks are useful evidence for the cases they cover, but they do not substitute for verifying the active integration path. Documentation should be updated as these checks become satisfied.

## Related documents

- [README.md](../README.md) — setup, usage, and measurement overview.
- [ethics.md](ethics.md) — governance, autonomy risks, privacy, source limitations, and audit guidance.
