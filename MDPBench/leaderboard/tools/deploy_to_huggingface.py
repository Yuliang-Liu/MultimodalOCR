#!/usr/bin/env python3
"""Mirror the leaderboard's static files to its Hugging Face Space.

Run in GitHub Actions with a write token in ``HF_TOKEN``. The script uses a
single explicit commit rather than the CLI's upload command, so it never tries
to create or change the Space's SDK configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


SPACE_ID = "Delores-Lin/MDPBench-leaderboard"
SKIP_PARTS = {"__pycache__", ".git"}
SKIP_NAMES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc"}


def is_publishable(path: Path) -> bool:
    return not (
        any(part in SKIP_PARTS for part in path.parts)
        or path.name in SKIP_NAMES
        or path.suffix in SKIP_SUFFIXES
    )


def main() -> None:
    # In GitHub Actions, use the explicit HF_TOKEN secret. Locally, the Hub
    # client falls back to the account token already stored by `hf auth login`.
    token = os.environ.get("HF_TOKEN")

    root = Path(__file__).resolve().parents[1]
    files = sorted(path for path in root.rglob("*") if path.is_file() and is_publishable(path))
    operations = [
        CommitOperationAdd(
            path_in_repo=path.relative_to(root).as_posix(),
            path_or_fileobj=str(path),
        )
        for path in files
    ]

    commit = HfApi(token=token).create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=operations,
        commit_message="Sync MDPBench leaderboard from GitHub",
    )
    print(f"Synced {len(files)} files to {SPACE_ID} at {commit.oid}.")


if __name__ == "__main__":
    main()
