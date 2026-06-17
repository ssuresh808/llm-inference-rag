#!/usr/bin/env bash
# Provision a free-tier Qdrant Cloud cluster from the terminal (no click-ops).
#
# Uses the natively-compiled Qdrant Cloud CLI at ~/go/bin/qcloud:
#   go install github.com/qdrant/qcloud-cli/cmd/qcloud@latest
#
# Cluster creation needs YOUR credentials, so this script verifies the CLI is
# present and then prints the exact commands to run and where to paste results.
set -euo pipefail

QCLOUD="$HOME/go/bin/qcloud"

if [[ ! -x "$QCLOUD" ]]; then
  echo "ERROR: qcloud CLI not found at $QCLOUD" >&2
  echo "Install it natively (requires Go):" >&2
  echo "  brew install go" >&2
  echo "  go install github.com/qdrant/qcloud-cli/cmd/qcloud@latest" >&2
  exit 1
fi
echo "Found qcloud at $QCLOUD"

cat <<'EOF'

============================================================
 Qdrant Cloud - free-tier cluster provisioning (qcloud)
============================================================

1) Point the CLI at your account. Create a Cloud API key and copy your
   account id from the Qdrant Cloud console first, then:
     ~/go/bin/qcloud context set my-cloud --api-key <KEY> --account-id <ID>

2) Create a FREE-tier cluster and wait for it to become healthy.
   (--package free selects the free booking package; verified via
    `~/go/bin/qcloud cluster create --help`.)
     ~/go/bin/qcloud cluster create \
       --cloud-provider aws \
       --cloud-region us-east-1 \
       --name portfolio-rag \
       --package free \
       --wait
     ~/go/bin/qcloud cluster list        # copy the <CLUSTER_ID> + endpoint URL

3) Mint a database API key scoped to that cluster:
     ~/go/bin/qcloud cluster key create <CLUSTER_ID> --name app-key

4) Paste the endpoint + key into .env.cloud:
     QDRANT_CLOUD_URL=https://<CLUSTER_ID>.<region>.aws.cloud.qdrant.io:6333
     QDRANT_API_KEY=<app-key>

5) Seed the cloud collection with the same ingestion pipeline:
     QDRANT_CLOUD_URL=... QDRANT_API_KEY=... uv run python -m scripts.seed_arxiv_db
============================================================
EOF
