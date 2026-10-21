#!/usr/bin/env bash
# Protects `main`: PR mandatory, 1 approval, green CI, linear history.
# Run ONCE, after the first CI run has completed (checks must exist).
# Note: branch protection on private repos needs a paid GitHub plan.
#
# Usage: REPO="<owner>/ai-share-of-voice" bash scripts/protect_main.sh
set -euo pipefail
: "${REPO:?Set REPO=<owner>/ai-share-of-voice}"

gh repo edit "$REPO" --enable-squash-merge \
  --enable-merge-commit=false --enable-rebase-merge=false --delete-branch-on-merge

gh api -X PUT "repos/$REPO/branches/main/protection" --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["test", "pr-title"] },
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
echo "main is now protected."
