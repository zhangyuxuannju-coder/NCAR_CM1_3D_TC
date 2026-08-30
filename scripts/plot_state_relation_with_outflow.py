#!/usr/bin/env python3
"""Replot the 72-h stability/M-budget comparison with radial-outflow overlays."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_stability_budget_ri_link import figure_state_relation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-npz", required=True)
    p.add_argument("--attribution-npz", required=True)
    p.add_argument("--coupling-metrics", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    budget = np.load(args.budget_npz)
    attr = np.load(args.attribution_npz)
    coupling = pd.read_csv(args.coupling_metrics)
    figure_state_relation(budget, attr, coupling, output, hour=72.0)
    print(output)


if __name__ == "__main__":
    main()
