from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .paths import ensure_dir


def get_git_commit(cwd: Path | None = None) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return output.decode("utf-8").strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def snapshot_config(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
