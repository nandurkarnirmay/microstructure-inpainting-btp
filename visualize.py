#!/usr/bin/env python
"""Thin CLI wrapper around visualization.viz for training curves and grids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.viz import plot_training_curves, save_comparison_grid


def main():
    ap = argparse.ArgumentParser(description="Plot curves or save a comparison grid from .pt tensors")
    ap.add_argument("--curves_json", type=str, default=None, help="JSON dict name->list of floats")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.curves_json:
        import json

        with open(args.curves_json, "r", encoding="utf-8") as f:
            series = json.load(f)
        plot_training_curves(series, args.out)
    else:
        ap.error("Provide --curves_json (extend script for multi-tensor grids if needed).")
    print("saved", args.out)


if __name__ == "__main__":
    main()
