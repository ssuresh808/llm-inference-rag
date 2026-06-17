#!/usr/bin/env bash
# Render GitOps deploy for the RAG Blueprint (render.yaml).
#
# Uses the REAL Render.com CLI at ~/.local/bin/render (v2.20.0), bypassing any
# npm "render" template package that may shadow PATH.
set -euo pipefail

RENDER="$HOME/.local/bin/render"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$RENDER" ]]; then
  echo "ERROR: Render CLI not found at $RENDER" >&2
  echo "Install the real CLI (NOT the npm 'render' package): https://render.com/docs/cli" >&2
  exit 1
fi

echo "==> Validating render.yaml against the Render Blueprint schema..."
"$RENDER" blueprints validate render.yaml

echo "==> Syncing the Blueprint to GitHub (Render auto-deploys connected repos on push)..."
git push origin main

cat <<'EOF'

============================================================
 Trigger the deploy with the real Render CLI:
============================================================
  ~/.local/bin/render login
  ~/.local/bin/render services                        # find your <service_id>
  ~/.local/bin/render deploys create <service_id> --wait
============================================================
EOF
