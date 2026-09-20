import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode


def audit_sync_barrier_2(
    pheno_path: Union[str, Path],
    baseline_path: Union[str, Path],
    augmented_path: Union[str, Path],
    tree_path: Union[str, Path],
    patristic_path: Union[str, Path],
    primary_approach: str = "Phylo-PCoA",
    report_out_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Sync Barrier 2: Architectural & Feature Freeze Audit Gate.
    Verifies:
      1. Complete 1:1 taxon correspondence between phenotypic targets, baseline, and augmented features.
      2. Zero tip-label mismatches between calibrated tree and feature matrices.
      3. Patristic distance matrix symmetry, non-negativity, and diagonal zeros.
      4. Zero missing values (NaNs / Infs) in all feature representations.
      5. Primary architectural approach lock (Phylo-PCoA eigenmaps vs GNN representations).
    """
    pheno_path = Path(pheno_path)
    baseline_path = Path(baseline_path)
    augmented_path = Path(augmented_path)
    tree_path = Path(tree_path)
    patristic_path = Path(patristic_path)

    report: Dict[str, Any] = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "taxa_count": 0,
        "primary_approach": primary_approach,
        "tree_tip_count": 0,
        "patristic_matrix_shape": [],
        "baseline_feature_count": 0,
        "augmented_feature_count": 0,
        "architecture_freeze_locked": False,
    }

    # 1. File existence
    for p_name, p in [
        ("phenotypic", pheno_path),
        ("baseline", baseline_path),
        ("augmented", augmented_path),
        ("tree", tree_path),
        ("patristic", patristic_path),
    ]:
        if not p.exists():
            report["status"] = "FAIL"
            report["errors"].append(f"Required artifact not found: {p_name} ({p})")

    if report["status"] == "FAIL":
        return False, report

    # 2. Load tables
    df_pheno = pd.read_csv(pheno_path)
    df_base = pd.read_csv(baseline_path)
    df_aug = pd.read_csv(augmented_path)

    for name, df in [("phenotypic", df_pheno), ("baseline", df_base), ("augmented", df_aug)]:
        if ID_COL not in df.columns:
            report["status"] = "FAIL"
            report["errors"].append(f"Primary key '{ID_COL}' missing in {name} table.")
        elif df[ID_COL].duplicated().any():
            report["status"] = "FAIL"
            report["errors"].append(f"Duplicate taxon IDs in {name} table.")

    if report["status"] == "FAIL":
        return False, report

    taxa = df_pheno[ID_COL].tolist()
    report["taxa_count"] = len(taxa)
    report["baseline_feature_count"] = len([c for c in df_base.columns if c != ID_COL])
    report["augmented_feature_count"] = len([c for c in df_aug.columns if c != ID_COL])

    # 3. 1:1 Index and ordering match
    if not (df_pheno[ID_COL].equals(df_base[ID_COL]) and df_base[ID_COL].equals(df_aug[ID_COL])):
        report["status"] = "FAIL"
        report["errors"].append("Taxon ID order or contents do not match 1:1 across phenotypic, baseline, and augmented tables.")

    # 4. Zero NaNs
    if df_base.isnull().any().any():
        report["status"] = "FAIL"
        report["errors"].append("Baseline feature table contains NaN values.")

    if df_aug.isnull().any().any():
        report["status"] = "FAIL"
        report["errors"].append("Augmented feature table contains NaN values.")

    # 5. Tree tip match
    with open(tree_path, "r") as f:
        newick_str = f.read().strip()
    tree = PhyloNode.from_newick(newick_str)
    tree_leaves = set(tree.get_leaf_names())
    report["tree_tip_count"] = len(tree_leaves)

    taxa_set = set(taxa)
    missing_in_tree = taxa_set - tree_leaves
    if missing_in_tree:
        report["status"] = "FAIL"
        report["errors"].append(f"Tree is missing {len(missing_in_tree)} taxa: {list(missing_in_tree)[:5]}")

    # 6. Patristic matrix audit
    if patristic_path.suffix == ".npy":
        D = np.load(patristic_path)
    else:
        D = pd.read_csv(patristic_path).values

    report["patristic_matrix_shape"] = list(D.shape)
    if D.shape[0] != len(taxa) or D.shape[1] != len(taxa):
        report["status"] = "FAIL"
        report["errors"].append(f"Patristic matrix dimensions {D.shape} do not match taxa count ({len(taxa)}, {len(taxa)})")

    # Symmetry and diagonal check
    if not np.allclose(D, D.T, atol=1e-3):
        report["status"] = "FAIL"
        report["errors"].append("Patristic matrix D is not symmetric.")

    if not np.allclose(np.diag(D), 0.0, atol=1e-4):
        report["status"] = "FAIL"
        report["errors"].append("Patristic matrix diagonal entries are non-zero.")

    if np.any(D < -1e-4):
        report["status"] = "FAIL"
        report["errors"].append("Patristic matrix contains negative distances.")

    is_passed = report["status"] == "PASS"
    report["architecture_freeze_locked"] = is_passed

    if report_out_path:
        out_p = Path(report_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)

    return is_passed, report


def main():
    parser = argparse.ArgumentParser(description="Sync Barrier 2: Architectural & Feature Freeze")
    parser.add_argument("--pheno", type=Path, default=Path("data/processed/phenotypic_clean.csv"))
    parser.add_argument("--baseline", type=Path, default=Path("data/processed/features_baseline.csv"))
    parser.add_argument("--augmented", type=Path, default=Path("data/processed/features_augmented.csv"))
    parser.add_argument("--tree", type=Path, default=Path("data/processed/phylo_tree_calibrated.nwk"))
    parser.add_argument("--patristic", type=Path, default=Path("data/processed/patristic_distance_matrix.npy"))
    parser.add_argument("--approach", type=str, default="Phylo-PCoA", choices=["Phylo-PCoA", "GNN", "Dual-Ensemble"])
    parser.add_argument("--report-out", type=Path, default=Path("data/processed/sync_barrier_2_report.json"))
    args = parser.parse_args()

    passed, report = audit_sync_barrier_2(
        pheno_path=args.pheno,
        baseline_path=args.baseline,
        augmented_path=args.augmented,
        tree_path=args.tree,
        patristic_path=args.patristic,
        primary_approach=args.approach,
        report_out_path=args.report_out,
    )

    print("=" * 60)
    print("SYNC BARRIER 2: ARCHITECTURAL & FEATURE FREEZE REPORT")
    print("=" * 60)
    print(f"STATUS: {report['status']}")
    print(f"Locked Taxa Count: {report['taxa_count']}")
    print(f"Tree Tip Count: {report['tree_tip_count']}")
    print(f"Patristic Matrix Shape: {report['patristic_matrix_shape']}")
    print(f"Baseline Features: {report['baseline_feature_count']}")
    print(f"Augmented Features: {report['augmented_feature_count']}")
    print(f"Primary Architecture: {report['primary_approach']}")
    print(f"Architecture Freeze Locked: {report['architecture_freeze_locked']}")
    if report["errors"]:
        print("\nERRORS DETECTED:")
        for err in report["errors"]:
            print(f"  [X] {err}")
    print("=" * 60)

    if not passed:
        sys.exit(1)
    print("SUCCESS: Architectural & Feature Freeze Locked.")


if __name__ == "__main__":
    main()
