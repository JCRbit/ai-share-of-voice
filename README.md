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

## Sections (owners)
- Installation — Ab
- Methodology — E
- Architecture, process justification and ethics — J
- Configuration and usage — An

## Contributing
See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
