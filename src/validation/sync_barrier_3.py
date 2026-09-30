import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.validation.cv import RandomKFoldSplitter, PhyloCVSplitter
from src.models import (
    RidgeRegressionPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    PhyloGNNPipeline,
    ValidationEngine,
)


def audit_sync_barrier_3(
    data_dir: Union[str, Path] = "data/processed",
    d_buffer: float = 140.0,
    min_clade_size: int = 50,
    target_trait: str = "adult_body_mass_g",
    report_out_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Sync Barrier 3: Clade Leakage Audit Gate.
    Verifies:
      1. Foundational Phase 1 & 2 artifacts existence and integrity.
      2. Zero sister-taxa / evaluation clade leakage in Phylogenetic CV folds.
      3. Strict buffer zone compliance (min distance between Train and Test >= d_buffer).
      4. Monophyletic integrity of held-out evaluation clades.
      5. Execution compliance across all 4 model architectures (Ridge, RF, XGBoost, GNN).
      6. Completeness of validation engine outputs.
    """
    data_dir = Path(data_dir)

    report: Dict[str, Any] = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "taxa_count": 0,
        "clade_leakage_audit_locked": False,
        "buffer_isolation_verified": False,
        "monophyletic_clades_verified": [],
        "phylo_cv_folds_count": 0,
        "random_cv_folds_count": 0,
        "models_verified": [],
        "smoke_test_metrics": {},
    }

    # 1. Artifact existence check
    req_files = {
        "phenotypic": data_dir / "phenotypic_clean.csv",
        "baseline": data_dir / "features_baseline.csv",
        "augmented": data_dir / "features_augmented.csv",
        "tree": data_dir / "phylo_tree_calibrated.nwk",
        "patristic": data_dir / "patristic_distance_matrix.npy",
        "sync_barrier_2": data_dir / "sync_barrier_2_report.json",
    }

    for name, path in req_files.items():
        if not path.exists():
            report["status"] = "FAIL"
            report["errors"].append(f"Required artifact missing: {name} at {path}")

    if report["status"] == "FAIL":
        return False, report

    # 2. Load data
    try:
        df_pheno = pd.read_csv(req_files["phenotypic"])
        df_base = pd.read_csv(req_files["baseline"])
        D = np.load(req_files["patristic"])

        with open(req_files["tree"], "r") as f:
            tree_root = PhyloNode.from_newick(f.read())
    except Exception as e:
        report["status"] = "FAIL"
        report["errors"].append(f"Error loading Phase 1/2 artifacts: {e}")
        return False, report

    all_taxa = df_pheno[ID_COL].tolist()
    report["taxa_count"] = len(all_taxa)

    if target_trait not in df_pheno.columns:
        report["status"] = "FAIL"
        report["errors"].append(f"Target trait '{target_trait}' not found in phenotypic dataset.")
        return False, report

    # 3. Monophyletic Withholding & Buffer Zone Leakage Audit
    withholding_engine = MonophyleticWithholdingEngine(tree_root=tree_root)
    phylo_cv = PhyloCVSplitter(
        all_taxa=all_taxa,
        patristic_matrix=D,
        withholding_engine=withholding_engine,
        mode="order",
        df_pheno=df_pheno,
        min_clade_size=min_clade_size,
        d_buffer=d_buffer,
    )

    phylo_folds = list(phylo_cv.split())
    report["phylo_cv_folds_count"] = len(phylo_folds)

    if len(phylo_folds) == 0:
        report["status"] = "FAIL"
        report["errors"].append(f"No clades found with min_clade_size >= {min_clade_size}.")
        return False, report

    # Audit each fold for leakage and buffer compliance
    total_taxa_set = set(range(len(all_taxa)))
    all_monophyletic = True
    buffer_compliant = True

    for fold in phylo_folds:
        # Check mutual disjointness
        if not fold.verify_disjointness():
            report["status"] = "FAIL"
            report["errors"].append(
                f"Clade leakage detected in fold '{fold.clade_name}': train, test, or buffer intersect."
            )

        # Check full coverage
        union_set = set(fold.train_indices.tolist()).union(
            set(fold.test_indices.tolist())
        ).union(set(fold.buffer_indices.tolist()))
        if union_set != total_taxa_set:
            report["status"] = "FAIL"
            report["errors"].append(
                f"Taxon coverage gap in fold '{fold.clade_name}': union of train+test+buffer does not equal all taxa."
            )

        # Check buffer distance condition: min D(s in Train, t in Test) >= d_buffer
        if len(fold.train_indices) > 0 and len(fold.test_indices) > 0:
            sub_D = D[np.ix_(fold.train_indices, fold.test_indices)]
            min_dist = float(np.min(sub_D))
            if min_dist < d_buffer - 1e-4:
                buffer_compliant = False
                report["status"] = "FAIL"
                report["errors"].append(
                    f"Buffer violation in fold '{fold.clade_name}': min patristic distance "
                    f"between training set and test clade is {min_dist:.2f} Ma (< d_buffer {d_buffer:.2f} Ma)."
                )

        # Check monophyly
        test_taxa = [all_taxa[i] for i in fold.test_indices]
        is_mono = withholding_engine.validate_monophyly(test_taxa)
        if is_mono:
            report["monophyletic_clades_verified"].append(fold.clade_name)
        else:
            all_monophyletic = False
            report["warnings"].append(
                f"Clade '{fold.clade_name}' is not strictly monophyletic in sampled tree."
            )

    report["buffer_isolation_verified"] = buffer_compliant

    # 4. Random K-Fold Audit
    random_cv = RandomKFoldSplitter(n_splits=10, random_state=42)
    random_folds = list(random_cv.split(all_taxa))
    report["random_cv_folds_count"] = len(random_folds)

    for rf in random_folds:
        if not rf.verify_disjointness():
            report["status"] = "FAIL"
            report["errors"].append(f"Random CV fold {rf.fold_id} has overlapping train/test sets.")

    # 5. Model Architecture Smoke Verification
    # Use first fold for smoke tests across all 4 architectures
    sample_fold = phylo_folds[0]
    feature_cols = [c for c in df_base.columns if c != ID_COL]
    X_mat = df_base[feature_cols].values.astype(np.float32)
    y_vec = df_pheno[target_trait].values.astype(np.float32)

    val_engine = ValidationEngine()

    models_to_test = [
        ("Ridge", lambda: RidgeRegressionPipeline()),
        ("RandomForest", lambda: RandomForestPipeline(n_estimators=10, n_jobs=-1)),
        ("XGBoost", lambda: XGBoostPipeline(n_estimators=10, n_jobs=-1)),
        ("PhyloGNN", lambda: PhyloGNNPipeline(epochs=5)),
    ]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_name, model_fn in models_to_test:
            try:
                m = model_fn()
                res = val_engine.run_fold(m, X_mat, y_vec, sample_fold)
                # Verify outputs
                if np.isnan(res["r2"]) or np.isinf(res["r2"]):
                    report["warnings"].append(f"Model {model_name} produced NaN/Inf R^2 on smoke test.")
                if np.isnan(res["rmse"]) or np.isnan(res["mae"]):
                    report["status"] = "FAIL"
                    report["errors"].append(f"Model {model_name} produced NaN RMSE/MAE.")
                report["models_verified"].append(model_name)
                report["smoke_test_metrics"][model_name] = {
                    "r2": round(res["r2"], 4),
                    "rmse": round(res["rmse"], 4),
                    "mae": round(res["mae"], 4),
                }
            except Exception as e:
                report["status"] = "FAIL"
                report["errors"].append(f"Model smoke test failed for {model_name}: {e}")

    # Final lock evaluation
    if report["status"] == "PASS" and len(report["errors"]) == 0:
        report["clade_leakage_audit_locked"] = True
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
    parser = argparse.ArgumentParser(description="Sync Barrier 3: Clade Leakage Audit Gate")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--d-buffer", type=float, default=140.0, help="Patristic buffer distance (Ma)")
    parser.add_argument("--min-clade-size", type=int, default=50, help="Minimum species in an evaluation clade")
    parser.add_argument("--target", type=str, default="adult_body_mass_g", help="Target phenotypic trait")
    parser.add_argument("--report", type=str, default="data/processed/sync_barrier_3_report.json", help="Report output path")
    args = parser.parse_args()

    passed, rep = audit_sync_barrier_3(
        data_dir=args.data_dir,
        d_buffer=args.d_buffer,
        min_clade_size=args.min_clade_size,
        target_trait=args.target,
        report_out_path=args.report,
    )

    print("\n" + "=" * 60)
    print("SYNC BARRIER 3 AUDIT GATE REPORT")
    print("=" * 60)
    print(f"Status: {rep['status']}")
    print(f"Taxa Locked: {rep['taxa_count']}")
    print(f"Phylo-CV Folds Audited: {rep['phylo_cv_folds_count']}")
    print(f"Random CV Folds Audited: {rep['random_cv_folds_count']}")
    print(f"Buffer Isolation Verified: {rep['buffer_isolation_verified']}")
    print(f"Models Verified: {', '.join(rep['models_verified'])}")
    print(f"Clade Leakage Freeze Locked: {rep['clade_leakage_audit_locked']}")
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
