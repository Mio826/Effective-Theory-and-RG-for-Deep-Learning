# dlphys/utils/plotting.py
from __future__ import annotations

from typing import Optional, Sequence

def plot_metric(
    df,
    y: str = "metrics.loss",
    *,
    x: str = "step",
    title: Optional[str] = None,
    label: Optional[str] = None,
    ax=None,
):
    """
    Plot a metric curve from a DataFrame.

    Example:
      plot_metric(df, y="metrics.loss", x="step")
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if x not in df.columns:
        raise KeyError(f"x='{x}' not found in df.columns: {list(df.columns)}")
    if y not in df.columns:
        raise KeyError(f"y='{y}' not found in df.columns: {list(df.columns)}")

    ax.plot(df[x], df[y], label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()

    return ax
