#!/usr/bin/env bash
# Creates labels, milestones, issues and a GitHub Project (board) for AI Share of Voice.
#
# Usage:
#   gh auth login
#   gh auth refresh -s project            # one-time: grants access to Projects
#   export REPO="<owner>/ai-share-of-voice"
#   export OWNER="<owner>"                # user or organisation that owns the project
#   export USER_E="<github-handle>" USER_J="<github-handle>" \
#          USER_AN="<github-handle>" USER_AB="<github-handle>"
#   bash scripts/setup_github_board.sh
#
# Run it ONCE: it is not idempotent for issues.

set -euo pipefail

: "${REPO:?Set REPO=<owner>/ai-share-of-voice}"
: "${OWNER:?Set OWNER=<owner>}"

declare -A HANDLE=( [E]="${USER_E:-}" [J]="${USER_J:-}" [AN]="${USER_AN:-}" [AB]="${USER_AB:-}" [ALL]="" )

# ---------- Labels ----------
label() { gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force >/dev/null; }
label "dept:data"       "5319e7" "Data and Statistics"
label "dept:agents"     "ff7a45" "Agents and Orchestration"
label "dept:platform"   "2ec4b6" "Integrations and Platform"
label "dept:product"    "f4c95d" "Product and Documentation"
label "dept:common"     "6e7781" "Shared task"
label "critical-path"   "d73a4a" "Delays here delay the whole team"

# ---------- Milestones (phases) ----------
declare -A MS=(
  [P0]="Phase 0 - Foundations (day 1)"
  [P1]="Phase 1 - Modules on mocks (days 2-4)"
  [P2]="Phase 2 - Skeleton (day 3)"
  [P3]="Phase 3 - Real integration (days 5-6)"
  [P4]="Phase 4 - Control and critique (days 6-8)"
  [P5]="Phase 5 - Freeze and docs (days 8-10)"
)
for k in "${!MS[@]}"; do
  gh api "repos/$REPO/milestones" -f title="${MS[$k]}" >/dev/null
done

# ---------- Project board ----------
PROJECT_NUM=$(gh project create --owner "$OWNER" --title "AI Share of Voice Board" --format json --jq '.number')
gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$REPO" >/dev/null
echo "Project #$PROJECT_NUM created"

# ---------- Issues ----------
# Fields: id | title | owner | dept | phase | days | depends on | acceptance criteria
TASKS=$(cat <<'EOF'
C4|Close open decisions and record them in docs/decisions.md|ALL|common|P0|0-1|none|Decided and written down: sector and city, model/provider and spend cap per person, whether AI-visibility monitoring fits as a market study (asked to the training team), how the Cloudista case is included, who is P1 to P4
C1|Agree and merge shared Pydantic contracts in schemas/|ALL|common|P0|1|C4|All contract models merged; each has a docstring and a unit test that instantiates it
I1|Set up repo, environment, CI and protected main|AN|platform|P0|1|none|Fresh clone runs pip install and pytest green; PRs blocked without review and green CI; day-1 spike confirms CrewAI, LangGraph, MCP and Gradio install together on Python 3.12 and versions are pinned
P1|Define sector, city and competitors in data/competitors.csv|AB|product|P1|1-2|none|2 to 5 competitors with name variants; sector and city validated by the team
I2|Create mock_run.py emitting simulated graph events|AN|platform|P1|2|C1|Script streams events in the agreed GraphEvent format
D1|Compute shares and confidence intervals over a test CSV|E|data|P1|2-3|C1|Wilson and cluster-bootstrap intervals implemented; tests with known values
D2|Implement should_stop with tests|E|data|P1|3-4|C1, D1|Every stop reason (precision, marginal return, budget, max batches) covered by a test and returned as text
D3|Build code-execution tool and charts|E|data|P1|3-4|D1|Tool runs vetted pandas/matplotlib functions, validates input and returns chart paths
A1|Build LangGraph graph with fake nodes|J|agents|P1|2-3|C1|Typed state, conditional edge, review loop capped at 2 rounds, interrupt before final action with SQLite checkpointer; every node appends GraphEvents to runs/<id>/events.jsonl; routing tests pass
A2|Define the seven CrewAI agents|J|agents|P1|3-5|C1, I3 signature|Each agent has role, goal, backstory and a bounded tool list
A3|Build mention extractor tool with Pydantic schema|J|agents|P1|3-5|C1|Invalid input is rejected; 5 hand-made examples pass
I3|Build model client with spend cap and logging|AN|platform|P1|2-4|C1|complete(prompt, schema) returns object and usage; raises BudgetExceeded; every call logged with tokens and cost
I4|Build MCP sources server and client adapter|AN|platform|P1|2-5|C1|search_sources, read_source and check_url work; robots.txt respected; page content returned as untrusted data; malicious-page fixture test passes
I5|Build vision agent analyze_image|AN|platform|P1|3-5|C1|Returns a finding for one chart and one screenshot
I6|Build app.py skeleton with tabs and spend bar|AN|platform|P1|3-4|I2|Brief, Activity and Results tabs render driven by mock_run
P2|Write 30 questions in data/questions.csv|AB|product|P1|2-4|P1|30 rows with id, text and intent; reviewed by E
P3|Write glossary and take competitor screenshots|AB|product|P1|2-5|none|docs/glossary.md covers 5 terms; 5 to 8 screenshots with date and URL in index.csv
C0|Day-3 end-to-end skeleton running in CI|ALL|common|P2|3|I2, A1, D1|A CI test runs the whole graph with simulated nodes and client (LLM_MODE=simulated) from brief to approval; mock_run.py still drives the UI
I7|Generate 50-answer real sample CSV|AN|platform|P3|4-5|I3|data/sample_answers.csv produced with the real client under a low cap
C2|First full real run, swapping mocks one at a time|ALL|common|P3|5-6|D2, A1, A2, A3, I3, I4|Real run in order: client, then MCP, then vision; checkpoint with the training team on day 5
D4|Build Results tab|E|data|P3|5-6|C1|Shows shares with intervals, charts and downloads
A4|Build Activity and approval tab|J|agents|P3|5-6|C1, I6|Live agent and tool events; approve and reject buttons resume the graph
P4|Review Brief form texts and error messages|AB|product|P3|5-6|C1, I6|docs/ui_copy_review.md plus one Issue per problem found
A5|Implement Skeptic and its review loop|J|agents|P4|6-7|C2|Skeptic returns claim-level objections; loop stops after 2 rounds; writer and skeptic treat source text as untrusted data and an injection fixture does not change their behaviour; tests pass
A6|Implement error paths|J|agents|P4|6-7|A1, I3|Tool failure, no evidence and exhausted budget all close the run in a partial state with a logged reason
A7|Generate report and tracking CSV after approval|J|agents|P4|6-8|C2|Report cites sources and run log; CSV append is idempotent; nothing runs before approval
P5|Hand-label 30 answers|AB|product|P4|6-7|I7|data/labeled_answers.csv complete; labeling guide written
P6|Run manual interface tests|AB|product|P5|8|Interface freeze|docs/manual_test_checklist.md executed; an Issue per failure
P7|Write and test README Installation section|AB|product|P5|8-9|C2|Works on a clean machine following only the README
D5|Write README Methodology section|E|data|P5|8-9|C2|Explains sampling, intervals, stopping rule and limits
A8|Write README Architecture, process justification and docs/ethics.md|J|agents|P5|8-9|C2|Diagram, hybrid-process justification, non-delegated decisions and risks included
I8|Write README Configuration and usage|AN|platform|P5|8-9|C2|Env variables, spend cap and run instructions documented
C5|Prepare demo script and update presentation slides|ALL|common|P5|8-9|C2|7 to 10 minute script covering use case, live demo, technical defense, MCP learning and ethics reflection; slides updated with real results
C3|Presentation rehearsal and final repo review|ALL|common|P5|9-10|All|7 to 10 minute demo rehearsed; repo checklist passed
EOF
)

while IFS='|' read -r id title who dept phase days deps accept; do
  body=$(printf '**Task ID:** %s\n**Owner:** %s\n**Days:** %s\n**Depends on:** %s\n\n## Acceptance criteria\n%s\n' \
         "$id" "$who" "$days" "$deps" "$accept")

  args=(--repo "$REPO" --title "[$id] $title" --body "$body" \
        --label "dept:$dept" --milestone "${MS[$phase]}")
  [[ "$id" == "I3" || "$id" == "A1" ]] && args+=(--label "critical-path")
  [[ -n "${HANDLE[$who]}" ]] && args+=(--assignee "${HANDLE[$who]}")

  url=$(gh issue create "${args[@]}")
  gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "$url" >/dev/null
  echo "Created $id -> $url"
done <<< "$TASKS"

echo "Done. Open the board: gh project view $PROJECT_NUM --owner $OWNER --web"
