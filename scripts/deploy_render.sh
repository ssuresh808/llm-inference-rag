#!/usr/bin/env bash
# Render GitOps deploy for the RAG Blueprint (render.yaml).
#
# HEADS UP: a tool named `render` (npm "render-cli", a template engine) may
# shadow your PATH — that is NOT the Render.com CLI and has no `blueprints`
# command. Install the real one:  brew tap render-oss/render && brew install render
#
# Render Blueprints deploy via GitOps: pushing render.yaml to the connected repo
# triggers a sync. We validate the YAML locally (no fake `render blueprints
# validate` command exists) and then push.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Validating render.yaml structure locally..."
uv run python - <<'PY'
import sys
import yaml

with open("render.yaml") as fh:
    data = yaml.safe_load(fh)

services = data.get("services", [])
if not services:
    sys.exit("render.yaml has no 'services' — nothing to deploy.")
names = ", ".join(s.get("name", "?") for s in services)
print(f"render.yaml OK: {len(services)} service(s): {names}")
PY

echo "==> Syncing the Blueprint to GitHub (Render auto-deploys connected repos on push)..."
git push origin main

cat <<'EOF'

============================================================
 Finish with the REAL Render CLI (render-oss):
============================================================
  brew tap render-oss/render && brew install render   # if not already installed
  render login

  # One-time: connect this repo's render.yaml as a Blueprint
  render blueprint launch          # interactive blueprint sync

  # Trigger + wait for a deploy of a specific service:
  render deploys create <SERVICE_ID> --wait --confirm

  # Find <SERVICE_ID> with:
  render services
============================================================
EOF
