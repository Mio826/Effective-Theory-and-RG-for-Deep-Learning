# dlphys/utils/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists and return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: PathLike, obj: Any, *, indent: int = 2) -> None:
    """
    Save obj to JSON (utf-8).
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: PathLike) -> Any:
    """
    Load JSON (utf-8).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: PathLike, record: Dict[str, Any]) -> None:
    """
    Append one JSON record as a line to a jsonl file.
    """
    p = Path(path)
    ensure_dir(p.parent)
    line = json.dumps(record, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
