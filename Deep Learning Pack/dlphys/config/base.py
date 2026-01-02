# dlphys/config/base.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dlphys.utils.io import ensure_dir, save_json, load_json
from dlphys.utils.time import now_str

PathLike = Union[str, Path]


def _to_path(p: Optional[PathLike]) -> Optional[Path]:
    if p is None:
        return None
    return Path(p).expanduser()


def _serialize(obj: Any) -> Any:
    """Make dataclass dict JSON-safe (Path -> str, etc.)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


@dataclass
class ExperimentConfig:
    """
    v0 experiment config (minimal but complete):
      - reproducibility
      - device/dtype
      - run directory + logging paths
      - basic training hyperparams
      - free-form tags/notes/extra

    You will later extend this with:
      - dataset/model/optimizer names
      - sweep support
      - checkpoints and evaluation
    """

    # ---- Identity / bookkeeping ----
    project_name: str = "dlphys"
    run_name: Optional[str] = None  # if None, auto-generate
    tags: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # ---- Reproducibility ----
    seed: int = 0
    deterministic: bool = False

    # ---- Compute ----
    device: str = "cuda"   # "cpu", "cuda", "cuda:0"
    dtype: str = "float32" # "float32", "float64", "bf16"

    # ---- Paths (relative to project_root unless absolute) ----
    project_root: Optional[str] = None  # if None, resolved at runtime
    runs_dir: str = "runs"              # parent folder for all runs
    run_dir: Optional[str] = None       # if None, runs_dir/run_name
    log_file: str = "train.log"         # inside run_dir by default

    # ---- Basic training hyperparams (placeholders for v0) ----
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 10

    # ---- Extra (for anything you don't want to formalize yet) ----
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- Resolved (filled by resolve_paths; not meant to be set manually) ----
    _resolved: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        """Functional update: returns a new config with overrides."""
        return replace(self, **kwargs)

    def ensure_run_name(self) -> str:
        """Ensure run_name exists; generate if missing."""
        if self.run_name is None or str(self.run_name).strip() == "":
            self.run_name = f"{now_str()}_{self.project_name}"
        return self.run_name

    def resolve_paths(self, *, base_dir: Optional[PathLike] = None, mkdir: bool = True) -> "ExperimentConfig":
        """
        Resolve project_root/run_dir/log_path into absolute paths.
        - base_dir: if provided, used as project_root when self.project_root is None.
                   In notebooks, pass Path.cwd().parent (project root).
        """
        self.ensure_run_name()

        root = _to_path(self.project_root)
        if root is None:
            root = _to_path(base_dir) if base_dir is not None else Path.cwd()
        root = root.resolve()

        runs_parent = (root / self.runs_dir) if not Path(self.runs_dir).is_absolute() else Path(self.runs_dir).resolve()

        if self.run_dir is None or str(self.run_dir).strip() == "":
            run_dir = runs_parent / self.run_name
        else:
            rd = Path(self.run_dir)
            run_dir = rd if rd.is_absolute() else (root / rd)

        run_dir = run_dir.resolve()
        log_path = Path(self.log_file)
        log_path = log_path if log_path.is_absolute() else (run_dir / log_path)
        log_path = log_path.resolve()

        if mkdir:
            ensure_dir(run_dir)
            ensure_dir(log_path.parent)

        self._resolved = {
            "project_root": str(root),
            "runs_parent": str(runs_parent.resolve()),
            "run_dir": str(run_dir),
            "log_path": str(log_path),
        }
        return self

    @property
    def resolved(self) -> Dict[str, str]:
        """Resolved absolute paths after resolve_paths()."""
        return dict(self._resolved)

    @property
    def run_dir_abs(self) -> Path:
        if "run_dir" not in self._resolved:
            raise RuntimeError("Call cfg.resolve_paths(...) first.")
        return Path(self._resolved["run_dir"])

    @property
    def log_path_abs(self) -> Path:
        if "log_path" not in self._resolved:
            raise RuntimeError("Call cfg.resolve_paths(...) first.")
        return Path(self._resolved["log_path"])

    def to_dict(self, *, include_resolved: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        # remove private runtime fields
        d.pop("_resolved", None)
        if include_resolved:
            d["resolved"] = dict(self._resolved)
        return _serialize(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        d.pop("resolved", None)  # ignore runtime data if present
        cfg = cls(**d)
        return cfg

    def save(self, path: PathLike, *, include_resolved: bool = True) -> Path:
        """Save config as JSON."""
        p = Path(path).expanduser()
        ensure_dir(p.parent)
        save_json(p, self.to_dict(include_resolved=include_resolved))
        return p

    @classmethod
    def load(cls, path: PathLike) -> "ExperimentConfig":
        """Load config from JSON."""
        d = load_json(Path(path).expanduser())
        return cls.from_dict(d)
