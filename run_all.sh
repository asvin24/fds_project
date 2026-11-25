#!/bin/bash
# Reproduces every number/figure for the report. ~2 minutes on 2 CPU cores. Outputs go to ./results
set -e; cd "$(dirname "$0")"
python3 p1_verify.py            # Part 1: correctness + stability vs 60-digit reference
python3 p2_compare.py 30        # Part 2: poly vs Gaussian, evidence vs CV (30 replicate datasets)
python3 p3_regimes.py 30        # Part 3: 3 DGPs x N x sigma0, interp/extrap, MSE & NLL (30 reps per cell)
python3 p4_rescale.py           # Part 4: rescaling of selected bases, transformed prior
python3 p5_disagree.py 100      # Part 5: evidence-vs-CV disagreement (100 datasets per setting)
