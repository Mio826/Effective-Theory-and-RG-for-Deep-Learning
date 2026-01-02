# dlphys/utils/loggers.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def _resolve_path(p: PathLike) -> Path:
    return Path(p).expanduser().resolve()


def clear_handlers(logger: logging.Logger) -> None:
    """
    Flush + close + remove all handlers.

    Notebook/Windows tip:
    - FileHandler will keep a lock on the file until closed.
    - Always clear/close before deleting a TemporaryDirectory.
    """
    for h in list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass
        try:
            logger.removeHandler(h)
        except Exception:
            pass


def close_file_handlers(logger: logging.Logger) -> None:
    """Close+remove only FileHandler(s); keep StreamHandler(s)."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            try:
                logger.removeHandler(h)
            except Exception:
                pass


def get_logger(
    name: str = "dlphys",
    *,
    level: int = logging.INFO,
    log_file: Optional[PathLike] = None,
    fmt: str = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt: str = "%H:%M:%S",
    reset: bool = False,
) -> logging.Logger:
    """
    Notebook-friendly logger factory.

    Behavior:
    - By default, reuse the same logger name without duplicating handlers.
    - If reset=True, clears existing handlers first (useful in notebooks).
    - Adds exactly one StreamHandler.
    - Adds at most one FileHandler; replaces it if log_file changes.

    Args:
        name: Logger name (use different names to avoid interference).
        level: Logging level.
        log_file: Optional path to log file.
        fmt/datefmt: Format for handlers.
        reset: If True, clear all handlers on this logger before reconfiguring.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if reset:
        clear_handlers(logger)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ----- ensure a single StreamHandler -----
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    if len(stream_handlers) == 0:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    else:
        # keep the first, remove extras
        for h in stream_handlers[1:]:
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
        stream_handlers[0].setLevel(level)
        stream_handlers[0].setFormatter(formatter)

    # ----- file handler (optional) -----
    if log_file is not None:
        target = _resolve_path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        keep = None
        for h in file_handlers:
            try:
                old = _resolve_path(getattr(h, "baseFilename", ""))
                if old == target:
                    keep = h
                    break
            except Exception:
                pass

        # Remove any file handlers not matching target
        for h in file_handlers:
            if h is keep:
                continue
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

        # Create if missing
        if keep is None:
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            keep.setLevel(level)
            keep.setFormatter(formatter)

    return logger


def set_level(logger: logging.Logger, level: int) -> None:
    """Set logger + all handler levels."""
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
