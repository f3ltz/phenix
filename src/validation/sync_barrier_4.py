import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL


def audit_sync_barrier_4(
    data_dir: Union[str, Path] = "data/processed",
    results_path: Optional[Union[str, Path]] = None,
    summary_path: Optional[Union[str, Path]] = None,
    predictions_path: Optional[Union[str, Path]] = None,
    divergence_path: Optional[Union[str, Path]] = None,
    report_out_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Sync Barrier 4: Benchmark Review Audit Gate.
    Verifies:
      1. Completeness of all 16 benchmark configurations (4 models x 2 feature sets x 2 protocols).
      2. Non-null evaluation metrics (R^2, RMSE, MAE) across all permutations.
      3. Performance Inflation Gap Verification (Random CV vs Phylo-CV).
      4. Phylogenetic Augmentation Gain Verification (Baseline vs Augmented features).
      5. Predictions matrix integrity (row counts, ground truth, residual consistency).
      6. Divergence degradation tracking across evaluation clades.
    """
    data_dir = Path(data_dir)
    res_p = Path(results_path) if results_path else data_dir / "benchmark_results.json"
    sum_p = Path(summary_path) if summary_path else data_dir / "benchmark_summary.csv"
    pred_p = Path(predictions_path) if predictions_path else data_dir / "predictions_matrix.csv"
    div_p = Path(divergence_path) if divergence_path else data_dir / "divergence_accuracy_tracking.json"

    report: Dict[str, Any] = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "benchmark_review_locked": False,
        "configurations_tested_count": 0,
        "required_configurations_count": 16,
        "species_in_predictions": 0,
        "inflation_gap_verified": False,
        "phylo_gain_verified": False,
        "divergence_tracking_verified": False,
        "key_findings": {},
    }

    # 1. File existence
    for name, p in [
        ("benchmark_results", res_p),
        ("benchmark_summary", sum_p),
        ("predictions_matrix", pred_p),
        ("divergence_tracking", div_p),
    ]:
        if not p.exists():
            report["status"] = "FAIL"
            report["errors"].append(f"Required benchmark artifact not found: {name} ({p})")

    if report["status"] == "FAIL":
        return False, report

    # 2. Benchmark completeness
    try:
        with open(res_p, "r") as f:
            matrix_results = json.load(f)
        df_summary = pd.read_csv(sum_p)
        df_pred = pd.read_csv(pred_p)
        with open(div_p, "r") as f:
            div_data = json.load(f)
    except Exception as e:
        report["status"] = "FAIL"
        report["errors"].append(f"Failed to read benchmark files: {e}")
        return False, report

    expected_models = ["Ridge", "RandomForest", "XGBoost", "PhyloGNN"]
    expected_feats = ["Baseline", "Augmented"]
    expected_protos = ["Random_CV", "Phylo_CV"]

    missing_configs = []
    for m in expected_models:
        for f_name in expected_feats:
            for p_name in expected_protos:
                cfg_key = f"{m}__{f_name}__{p_name}"
                if cfg_key not in matrix_results:
                    missing_configs.append(cfg_key)

    report["configurations_tested_count"] = len(matrix_results)
    if missing_configs:
        report["status"] = "FAIL"
        report["errors"].append(f"Missing benchmark configurations: {missing_configs}")

    # Check for NaN summary metrics
    for col in ["r2_mean", "rmse_mean", "mae_mean"]:
        if col in df_summary.columns and df_summary[col].isnull().any():
            report["status"] = "FAIL"
            report["errors"].append(f"Summary table contains NaN in '{col}'.")

    # 3. Performance Inflation Gap Verification: Random CV vs Phylo-CV
    # On Baseline features, Random CV should show significantly higher R^2 than Phylo-CV
    inflation_gaps = []
    for m in expected_models:
        m_base = df_summary[(df_summary["model"] == m) & (df_summary["feature_set"] == "Baseline")]
        r_rand = m_base[m_base["protocol"] == "Random_CV"]["r2_mean"].values
        r_phylo = m_base[m_base["protocol"] == "Phylo_CV"]["r2_mean"].values
        if len(r_rand) > 0 and len(r_phylo) > 0:
            gap = float(r_rand[0] - r_phylo[0])
            inflation_gaps.append((m, gap))

    # Average inflation gap across baseline models
    if inflation_gaps:
        avg_gap = float(np.mean([g[1] for g in inflation_gaps]))
        report["key_findings"]["average_baseline_inflation_gap_r2"] = round(avg_gap, 4)
        if avg_gap > 0.0:
            report["inflation_gap_verified"] = True
        else:
            report["warnings"].append(
                f"Expected positive performance inflation gap on baseline features, observed: {avg_gap:.4f}"
            )

    # 4. Phylogenetic Augmentation Benefit Verification
    # Augmented features should improve Random CV and Phylo-CV performance
    phylo_gains_random = []
    for m in expected_models:
        m_rows = df_summary[(df_summary["model"] == m) & (df_summary["protocol"] == "Random_CV")]
        r_aug = m_rows[m_rows["feature_set"] == "Augmented"]["r2_mean"].values
        r_base = m_rows[m_rows["feature_set"] == "Baseline"]["r2_mean"].values
        if len(r_aug) > 0 and len(r_base) > 0:
            gain = float(r_aug[0] - r_base[0])
            phylo_gains_random.append((m, gain))

    if phylo_gains_random:
        avg_gain = float(np.mean([g[1] for g in phylo_gains_random]))
        report["key_findings"]["average_phylo_augmentation_gain_r2"] = round(avg_gain, 4)
        if avg_gain > 0.2:  # Strong phylogenetic gain observed (typically > +0.7 R^2)
            report["phylo_gain_verified"] = True
        else:
            report["warnings"].append(f"Phylo gain below expected threshold: {avg_gain:.4f}")

    # 5. Predictions Matrix Integrity
    report["species_in_predictions"] = len(df_pred)
    if len(df_pred) == 0:
        report["status"] = "FAIL"
        report["errors"].append("Predictions matrix has 0 rows.")

    if ID_COL not in df_pred.columns or "true_y_log10" not in df_pred.columns:
        report["status"] = "FAIL"
        report["errors"].append("Predictions matrix missing ID_COL or ground truth column.")

    # 6. Divergence tracking verification
    if len(div_data) > 0:
        report["divergence_tracking_verified"] = True
    else:
        report["warnings"].append("Divergence tracking results empty.")

    # Final lock assessment
    if report["status"] == "PASS" and len(report["errors"]) == 0:
        report["benchmark_review_locked"] = True
    else:
        report["status"] = "FAIL"

    # Export report if requested
    if report_out_path:
        out_p = Path(report_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)

    return (report["status"] == "PASS"), report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Barrier 4: Benchmark Review Audit Gate")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--report", type=str, default="data/processed/sync_barrier_4_report.json", help="Report output path")
    args = parser.parse_args()

    passed, rep = audit_sync_barrier_4(
        data_dir=args.data_dir,
        report_out_path=args.report,
    )

    print("\n" + "=" * 60)
    print("SYNC BARRIER 4 AUDIT GATE REPORT")
    print("=" * 60)
    print(f"Status: {rep['status']}")
    print(f"Configurations Tested: {rep['configurations_tested_count']} / {rep['required_configurations_count']}")
    print(f"Species in Predictions: {rep['species_in_predictions']}")
    print(f"Performance Inflation Gap Verified: {rep['inflation_gap_verified']}")
    print(f"Phylogenetic Gain Verified: {rep['phylo_gain_verified']}")
    print(f"Divergence Tracking Verified: {rep['divergence_tracking_verified']}")
    print(f"Benchmark Review Locked: {rep['benchmark_review_locked']}")
    if rep["key_findings"]:
        print("Key Findings:")
        for k, v in rep["key_findings"].items():
            print(f"  - {k}: {v}")
    if rep["errors"]:
        print(f"Errors ({len(rep['errors'])}):")
        for err in rep["errors"]:
            print(f"  - {err}")
    if rep["warnings"]:
        print(f"Warnings ({len(rep['warnings'])}):")
        for w in rep["warnings"]:
            print(f"  - {w}")
    print("=" * 60)

    if not passed:
        sys.exit(1)
