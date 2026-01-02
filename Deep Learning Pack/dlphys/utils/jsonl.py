# dlphys/utils/jsonl.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Optional

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (one dict per line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rows: List[Dict[str, Any]] = []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict:
      {"metrics": {"loss": 0.1}, "step": 3}
    -> {"metrics.loss": 0.1, "step": 3}
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def jsonl_to_df(
    path: PathLike,
    *,
    flatten: bool = True,
    sep: str = ".",
    keep_raw: bool = False,
):
    """
    Read JSONL into a pandas DataFrame.
    - flatten=True: flatten nested dicts into columns like "metrics.loss"
    - keep_raw=True: keep the original dict in a column "_raw"
    """
    rows = read_jsonl(path)
    if flatten:
        rows2 = []
        for r in rows:
            fr = flatten_dict(r, sep=sep)
            if keep_raw:
                fr["_raw"] = r
            rows2.append(fr)
        rows = rows2  # type: ignore[assignment]

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for jsonl_to_df. Please install pandas.") from e

    df = pd.DataFrame(rows)
    return df
