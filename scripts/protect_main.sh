#!/usr/bin/env bash
# Protects `main`: PR mandatory, 1 approval, linear history, squash merge only.
#
# Usage:
#   REPO="<owner>/ai-share-of-voice" bash scripts/protect_main.sh                 # requires CI checks
#   REPO="<owner>/ai-share-of-voice" REQUIRE_CI=false bash scripts/protect_main.sh # no CI available
#
# With REQUIRE_CI=true (default) run it AFTER the first CI run has completed (the checks must exist).
# Note: branch protection on private repos needs a paid GitHub plan.
set -euo pipefail
: "${REPO:?Set REPO=<owner>/ai-share-of-voice}"
REQUIRE_CI="${REQUIRE_CI:-true}"

if [[ "$REQUIRE_CI" == "true" ]]; then
  CHECKS='{ "strict": true, "contexts": ["test", "pr-title"] }'
else
  CHECKS='null'
fi

gh repo edit "$REPO" --enable-squash-merge \
  --enable-merge-commit=false --enable-rebase-merge=false --delete-branch-on-merge

gh api -X PUT "repos/$REPO/branches/main/protection" --input - <<JSON
{
  "required_status_checks": $CHECKS,
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "required_linear_history": true,
  "required_conversation_resolution": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "restrictions": null
}
JSON
echo "main is now protected (REQUIRE_CI=$REQUIRE_CI)."
