"""
Unit tests for Stage 5: Residual Diagnostics and Variance Partitioning
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.diagnostics import (
    ResidualDiagnosticsEngine,
    VariancePartitioningEngine,
    run_stage_5_diagnostics,
)


@pytest.fixture
def dummy_diagnostics_setup(tmp_path):
    N = 50
    # Symmetric positive definite matrix with zero diagonal
    rng = np.random.RandomState(42)
    coords = rng.randn(N, 5)
    # Pairwise euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(D, 0.0)

    patristic_path = tmp_path / "patristic_distance_matrix.npy"
    np.save(patristic_path, D)

    # Predictions matrix
    y_true = rng.randn(N)
    pred_rf = y_true + 0.2 * rng.randn(N)
    resid_rf = y_true - pred_rf

    df_preds = pd.DataFrame({
        "canonical_taxon_id": [f"taxon_{i}" for i in range(N)],
        "true_y_log10": y_true,
        "resid__RandomForest__Augmented__Random_CV": resid_rf,
    })
    pred_path = tmp_path / "predictions_matrix.csv"
    df_preds.to_csv(pred_path, index=False)

    return tmp_path, patristic_path, pred_path


def test_residual_diagnostics_engine(dummy_diagnostics_setup):
    tmp_path, patristic_path, pred_path = dummy_diagnostics_setup
    engine = ResidualDiagnosticsEngine(patristic_matrix_path=patristic_path, root_age_ma=50.0)

    df_preds = pd.read_csv(pred_path)
    y_true = df_preds["true_y_log10"].values
    D = np.load(patristic_path)

    k_val = engine.compute_blomberg_k(y_true, D)
    lam_val = engine.compute_pagel_lambda(y_true, D)

    assert isinstance(k_val, float)
    assert not np.isnan(k_val)
    assert 0.0 <= lam_val <= 1.0

    results = engine.evaluate_all_residuals(df_preds)
    assert "true_target" in results
    assert "resid__RandomForest__Augmented__Random_CV" in results["residual_diagnostics"]


def test_stage_5_artifacts_exist():
    data_dir = Path("data/processed")
    res_path = data_dir / "residual_diagnostics.json"
    var_path = data_dir / "variance_partitioning.json"

    assert res_path.exists(), "residual_diagnostics.json was not generated"
    assert var_path.exists(), "variance_partitioning.json was not generated"

    with open(res_path) as f:
        res_data = json.load(f)
    assert "true_target" in res_data
    assert "residual_diagnostics" in res_data
    assert res_data["true_target"]["blomberg_k"] > 2.0

    with open(var_path) as f:
        var_data = json.load(f)
    assert "random_forest_nonlinear_partitioning" in var_data
    assert var_data["random_forest_nonlinear_partitioning"]["pure_phylogenetic"] > 0.8
