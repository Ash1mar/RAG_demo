from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip a single pair of matching quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Remove inline comments only when preceded by whitespace.
            for idx, ch in enumerate(value):
                if ch == "#" and idx > 0 and value[idx - 1].isspace():
                    value = value[: idx - 1].rstrip()
                    break
        values[key] = value
    return values


def load_app_config() -> None:
    """Load config/app.env unless APP_CONFIG overrides it.

    Values already set in os.environ are preserved to allow runtime overrides.
    """
    root = Path(__file__).resolve().parents[1]
    raw = os.getenv("APP_CONFIG", "config/app.env")
    cfg_path = Path(raw)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        return
    for key, value in _parse_env_file(cfg_path).items():
        if key not in os.environ:
            os.environ[key] = value
