# dlphys — A Minimal, Research-Friendly Deep Learning Experiment Framework

dlphys is a lightweight PyTorch experiment framework designed for *research workflows*: fast iteration, clean configuration, reproducible runs, and easy “sweep → scan → analyze” loops.

It provides:
- A small but solid training engine (Trainer + callbacks)
- A registry-based component system (model / data / optimizer / init) configured by strings + kwargs
- A run manager that creates run folders, saves configs, and standardizes logging
- Structured metric logging to `metrics.jsonl` for programmatic analysis (`scan_runs`)

This repo is intentionally minimal: it’s meant to be a *research lab bench*, not a full production platform.

---

## Why this exists (research motivation)

This project is built to support experiment-heavy research in deep learning theory, including:
- Tracking/controlling criticality during training (EOC / dynamical isometry proxies)
- Empirical RG-style atlases of pre-activation statistics vs depth/width
- NTK drift measurement and control (lazy ↔ feature-learning regimes)
- Feature Learning Index (FLI) style “phase diagrams” across architectures/tasks

The architecture favors:
- Reproducibility (configs saved per run)
- Sweepability (choose components via config strings)
- Observability (structured metric logs)
- Extensibility (add one file + register one function)

---

## Repository layout (mental model)

Think of this repo as a lab:

- `dlphys/core/`  
  The control room: run directory management and run scanning.
  - `run.py`: creates a run folder, builds model/init/optimizer in the correct order
  - `scan_runs.py`: aggregates many runs into a DataFrame-friendly summary

- `dlphys/config/`  
  The experiment blueprint: configuration objects + registries.
  - `base.py`: `ExperimentConfig` dataclass + path resolution
  - `registry.py`: the central registry system for mapping strings → builder functions

- `dlphys/training/`  
  The experiment executor:
  - `engine.py`: `Trainer.fit()` loop
  - `callbacks.py`: callback API + callback list
  - `logging.py`: JSONL metrics logger callback (`metrics.jsonl`)

- `dlphys/data/`  
  The sample preparation room: datasets and DataLoaders, registered by name.
  - includes a minimal synthetic dataset for smoke tests

- `dlphys/models/`  
  The instruments: neural network architectures, registered by name.

- `dlphys/init/`  
  The calibration bench: init strategies, registered by name.
  - applied *after model build* and *before optimizer build*

- `dlphys/utils/`  
  Tool drawers: seed/device helpers, running stats, JSONL IO, plotting, etc.

Runs are stored under `runs/<run_name>/` and contain:
- `config.json` (snapshot of the config)
- `train.log` (human-readable text logs)
- `metrics.jsonl` (structured metrics, one JSON per line)

---

## Installation

This repo is designed to run as a simple local project.

Recommended:
- Python 3.10+
- PyTorch installed (CPU or GPU)

Example:
1) Create a venv (or conda env)
2) Install PyTorch
3) Add repo root to `PYTHONPATH` or run from notebooks with:

```python
import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
