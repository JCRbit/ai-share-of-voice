# AI Share of Voice

Measures how often a brand appears and is recommended against its competitors in an AI
assistant's answers, and doubts its own results before delivering them.

Project III, Module V (AI Engineering) — AgentSquad: multi-agent system with CrewAI, LangGraph and MCP.

> Every result is a dated sample. It is never presented as market share, and no causality is
> claimed between a web source and a recommendation.

## Status
C1 shared contracts and the A1 LangGraph skeleton are implemented. The A1 graph uses simulated nodes, bounded loops, SQLite checkpointing, human approval and JSONL audit events. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for phases, owners and tasks.

## Quick start
```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env        # add your own key and spending cap
make check
```

## Sections (owners)
- Installation — Ab
- Methodology — E
- Architecture, process justification and ethics — J
- Configuration and usage — An

## Contributing
See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
