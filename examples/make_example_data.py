#!/usr/bin/env python
"""Generate a tiny synthetic dataset to try the evaluation toolkit offline.

Writes three files next to this script:

  example_predictions.h5   a "model" run   (probabilities, targets, locations)
  example_reference.h5     a weaker "reference" run (same targets/locations)
  example_species_properties.csv  per-species stratification axes
                           (species_id, sampling_effort, relative_prevalence)

The model is engineered to beat the reference for *sparse + infrequent* species
but lose for *dense + frequent* ones, so the stratified report and delta heatmap
show a clear green (model better) -> purple (reference better) gradient. This is
purely synthetic data to demonstrate the toolkit end-to-end without downloading the
real benchmark data — it is not real SDM output.

Run:
    python examples/make_example_data.py
    python scripts/evaluate_predictions.py \\
        --predictions examples/example_predictions.h5 \\
        --reference   examples/example_reference.h5 \\
        --proxies     examples/example_species_properties.csv \\
        --metrics auroc auprc f1 \\
        --output-dir  examples/output
"""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_LOCATIONS = 2000
N_SPECIES = 1500


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(0)

    # Per-species stratification axes (the delta-heatmap / stratify dimensions).
    sampling_effort = rng.lognormal(3.0, 1.0, N_SPECIES)
    taxo_pref = rng.uniform(0.0, 1.0, N_SPECIES)
    se_rank = rankdata(sampling_effort) / N_SPECIES          # 0..1 (sparse..dense)
    tp_rank = rankdata(taxo_pref) / N_SPECIES                # 0..1 (infrequent..frequent)

    # Ground-truth presence/absence; ~20% of entries masked (-1 = out of range).
    prevalence = rng.uniform(0.03, 0.30, N_SPECIES)
    targets = (rng.random((N_LOCATIONS, N_SPECIES)) < prevalence[None, :]).astype(np.float32)
    targets[rng.random((N_LOCATIONS, N_SPECIES)) < 0.20] = -1.0

    # Plausible European lon/lat (unused by the metrics, part of the format).
    locations = np.column_stack([
        rng.uniform(-10.0, 35.0, N_LOCATIONS),
        rng.uniform(34.0, 72.0, N_LOCATIONS),
    ]).astype(np.float32)

    # Probabilities: higher per-species "skill" -> better presence/absence separation.
    signal = np.where(targets == 1, 1.0, -1.0)
    ref_skill = 1.6 + rng.normal(0.0, 0.2, N_SPECIES)             # a decent reference
    # The model beats the reference for sparse + infrequent species but LOSES for
    # dense + frequent ones, so the heatmap shows both green (model wins) and
    # purple (reference wins).
    model_gain = 0.45 * (1 - se_rank) + 0.45 * (1 - tp_rank) - 0.45

    def make_probs(skill):
        logit = 0.5 * skill[None, :] * signal + rng.normal(0.0, 1.0, targets.shape)
        return _sigmoid(logit).astype(np.float32)

    model_probs = make_probs(ref_skill + model_gain)
    reference_probs = make_probs(ref_skill)

    def _write(path, probs):
        with h5py.File(path, "w") as f:
            f.create_dataset("probabilities", data=probs)
            f.create_dataset("targets", data=targets)
            f.create_dataset("locations", data=locations)
        print(f"  wrote {path}  ({probs.shape[0]} locations x {probs.shape[1]} species)")

    _write(os.path.join(out_dir, "example_predictions.h5"), model_probs)
    _write(os.path.join(out_dir, "example_reference.h5"), reference_probs)

    pd.DataFrame({
        "species_id": np.arange(N_SPECIES),
        "sampling_effort": sampling_effort,
        "relative_prevalence": taxo_pref,
    }).to_csv(os.path.join(out_dir, "example_species_properties.csv"), index=False)
    print(f"  wrote {os.path.join(out_dir, 'example_species_properties.csv')}")


if __name__ == "__main__":
    main()
