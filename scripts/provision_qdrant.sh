#!/usr/bin/env bash
# Provision a free-tier Qdrant Cloud cluster from the terminal (no click-ops).
#
# This script does not touch your account — cluster creation needs your own
# Qdrant Cloud credentials, so it prints the exact commands to run and where to
# paste the results (.env.cloud). Review before running anything.
set -euo pipefail

cat <<'EOF'
============================================================
 Qdrant Cloud — free-tier cluster provisioning (qcloud CLI)
============================================================

1) Install the Qdrant Cloud CLI (qcloud):
     macOS (Homebrew):  brew install qdrant/tap/qcloud
     or grab a binary:  https://github.com/qdrant/qdrant-cloud-cli/releases

2) Create an account-level "Cloud API key" in the console
   (Console > Access Management > Cloud API Keys), then authenticate:
     export QDRANT_CLOUD_API_KEY="<your-cloud-api-key>"
     qcloud auth login --api-key "$QDRANT_CLOUD_API_KEY"

3) Create a FREE-tier cluster (1 GB, no credit card):
     qcloud cluster create \
       --name llm-rag-portfolio \
       --cloud aws \
       --region us-east-1 \
       --tier free
     qcloud cluster list        # copy the cluster id + endpoint

4) Mint a database API key scoped to that cluster:
     qcloud database-api-key create --cluster-id <CLUSTER_ID>

5) Paste the endpoint + key into .env.cloud:
     QDRANT_CLOUD_URL=https://<CLUSTER_ID>.<region>.aws.cloud.qdrant.io:6333
     QDRANT_API_KEY=<database-api-key>

6) Seed the cloud collection with the same ingestion pipeline:
     QDRANT_CLOUD_URL=... QDRANT_API_KEY=... uv run python -m scripts.seed_arxiv_db

NOTE: qcloud flags evolve between releases — confirm any command with
      `qcloud <subcommand> --help` before relying on it.
============================================================
EOF
