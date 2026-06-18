"""Production smoke test for the deployed RAG backend.

Hits the live FastAPI service to prove the full cloud path works end to end: the
health probe responds, and ``/api/v1/answer`` returns a grounded answer *with*
sources - which only happens if Qdrant Cloud and the hosted LLM are both wired
up and talking to each other.

    uv run python scripts/verify_prod.py https://rag-backend.onrender.com
    # or, with RAG_API_URL set in .env.cloud:
    uv run python scripts/verify_prod.py
"""

import argparse
import sys
from pathlib import Path

import httpx

DEFAULT_QUERY = "How does PagedAttention work?"
ENV_CLOUD = ".env.cloud"


def _read_env_cloud(key: str, path: str = ENV_CLOUD) -> str:
    """Return ``key`` from a dotenv-style ``.env.cloud`` file, or '' if absent.

    Args:
        key: Variable name to look up.
        path: Path to the dotenv file.

    Returns:
        The value, stripped, or an empty string if the file/key is missing.
    """
    file = Path(path)
    if not file.exists():
        return ""
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip()
    return ""


def _resolve_base_url(cli_url: str | None) -> str:
    """Resolve the backend base URL from the CLI arg, else ``.env.cloud``."""
    base = (cli_url or _read_env_cloud("RAG_API_URL")).rstrip("/")
    if not base:
        sys.exit("No backend URL. Pass it as an argument or set RAG_API_URL in .env.cloud.")
    return base


def _check_health(base_url: str) -> None:
    """Assert the liveness probe responds with HTTP 200."""
    response = httpx.get(f"{base_url}/health", timeout=30.0)
    response.raise_for_status()
    print(f"[1/2] GET /health -> {response.status_code} {response.json()}")


def _check_answer(base_url: str, query: str) -> None:
    """Assert ``/api/v1/answer`` returns a non-empty answer and sources."""
    response = httpx.post(f"{base_url}/api/v1/answer", json={"text": query}, timeout=180.0)
    response.raise_for_status()
    data = response.json()

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    if not answer:
        sys.exit("FAIL: response carried no 'answer' (the hosted LLM is not responding).")
    if not sources:
        sys.exit("FAIL: response carried no 'sources' (Qdrant Cloud retrieval is not wired up).")

    print(f"[2/2] POST /api/v1/answer -> {len(answer)} chars, {len(sources)} source(s): {sources}")


def main() -> None:
    """Run the production smoke test against the live backend."""
    parser = argparse.ArgumentParser(description="Smoke-test the deployed RAG backend.")
    parser.add_argument("base_url", nargs="?", help="Live backend base URL.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Test question to send.")
    args = parser.parse_args()

    base_url = _resolve_base_url(args.base_url)
    print(f"Smoke-testing {base_url} ...")
    _check_health(base_url)
    _check_answer(base_url, args.query)

    bar = "=" * 64
    print(f"\n{bar}")
    print("  SMOKE TEST PASSED - backend, Qdrant Cloud, and LLM are live.")
    print(bar)
    streamlit_url = _read_env_cloud("STREAMLIT_URL") or "<your rag-frontend Render URL>"
    print(f"\nNow open the live Streamlit UI in your browser:\n  {streamlit_url}")


if __name__ == "__main__":
    main()
