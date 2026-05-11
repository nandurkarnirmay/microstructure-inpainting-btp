from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_hw(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float().numpy()
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    return np.clip(x, 0.0, 1.0)


def save_comparison_grid(
    tensors: List[torch.Tensor],
    titles: List[str],
    out_path: str | Path,
    nrow: int = 1,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncol = len(tensors)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axs = np.atleast_1d(axs).reshape(-1)
    for ax, t, title in zip(axs, tensors, titles):
        ax.imshow(_to_hw(t), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_training_curves(
    series: dict[str, Iterable[float]],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for name, ys in series.items():
        ys = list(ys)
        plt.plot(range(len(ys)), ys, label=name)
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
