import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.validation.cv import RandomKFoldSplitter, PhyloCVSplitter, CVFold
from src.models import (
    RidgeRegressionPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    PhyloGNNPipeline,
    ValidationEngine,
)
from src.validation.sync_barrier_3 import audit_sync_barrier_3


class Phase3Pipeline:
    """
    Unified Pipeline Orchestrator for Phase 3: Validation Engine & Model Engineering.
    Coordinates:
      - Person A: Monophyletic taxonomic withholdings, T_cut depth slicing, buffer quarantine zones.
      - Person B: Random 10-Fold CV, Phylogenetic CV (Phylo-CV), and model pipelines (Ridge, RF, XGBoost, GNN).
      - Sync Barrier 3: Clade Leakage Audit Gate.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        d_buffer: float = 140.0,
        min_clade_size: int = 50,
        t_cut_ma: float = 65.0,
        target_trait: str = "adult_body_mass_g",
    ):
        self.data_dir = Path(data_dir)
        self.d_buffer = d_buffer
        self.min_clade_size = min_clade_size
        self.t_cut_ma = t_cut_ma
        self.target_trait = target_trait

    def run(self) -> Dict[str, Any]:
        """Runs end-to-end Phase 3 validation engine and locks Sync Barrier 3."""
        print("\n" + "=" * 70)
        print("PHENIX PHASE 3: VALIDATION ENGINE & MODEL ENGINEERING")
        print("=" * 70)

        # 1. Load Processed Datasets
        print("[Phase 3 Pipeline] Step 1: Loading Phase 1 & Phase 2 frozen artifacts...")
        pheno_path = self.data_dir / "phenotypic_clean.csv"
        baseline_path = self.data_dir / "features_baseline.csv"
        augmented_path = self.data_dir / "features_augmented.csv"
        tree_path = self.data_dir / "phylo_tree_calibrated.nwk"
        patristic_path = self.data_dir / "patristic_distance_matrix.npy"

        for p in [pheno_path, baseline_path, augmented_path, tree_path, patristic_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required artifact missing: {p}")

        df_pheno = pd.read_csv(pheno_path)
        df_base = pd.read_csv(baseline_path)
        df_aug = pd.read_csv(augmented_path)
        D = np.load(patristic_path)

        with open(tree_path, "r") as f:
            tree_root = PhyloNode.from_newick(f.read())

        taxa = df_pheno[ID_COL].tolist()
        N = len(taxa)
        print(f"[Phase 3 Pipeline] Loaded {N} taxa across 27 orders.")

        # 2. Person A: Monophyletic Withholdings & Buffer Zones
        print("\n[Phase 3 Pipeline] Step 2 (Person A): Computing monophyletic withholdings & buffer zones...")
        withholding_engine = MonophyleticWithholdingEngine(tree_root=tree_root)

        order_clades = withholding_engine.get_order_clades(df_pheno, min_clade_size=self.min_clade_size)
        print(f"[Phase 3 Pipeline] Identified {len(order_clades)} orders with >= {self.min_clade_size} species:")
        for order_name, members in order_clades.items():
            print(f"  - {order_name}: {len(members)} species")

        t_cut_clusters = withholding_engine.get_t_cut_clusters(t_cut_ma=self.t_cut_ma)
        print(f"[Phase 3 Pipeline] Sliced calibrated tree at T_cut = {self.t_cut_ma} Ma -> {len(t_cut_clusters)} lineages.")

        # 3. Person B: CV Splitters & Fold Definitions
        print("\n[Phase 3 Pipeline] Step 3 (Person B): Generating CV splitters & fold schemas...")
        # Random 10-Fold
        random_cv = RandomKFoldSplitter(n_splits=10, random_state=42)
        random_folds = list(random_cv.split(taxa))
        random_folds_dict = {
            f"fold_{f.fold_id}": {
                "train_taxa": [taxa[i] for i in f.train_indices],
                "test_taxa": [taxa[i] for i in f.test_indices],
            }
            for f in random_folds
        }
        random_folds_path = self.data_dir / "cv_random_folds.json"
        with open(random_folds_path, "w") as f:
            json.dump(random_folds_dict, f, indent=2)
        print(f"[Phase 3 Pipeline] Exported Random 10-Fold CV splits to {random_folds_path.name}")

        # Phylo-CV (Order holdout with buffer exclusion)
        phylo_cv = PhyloCVSplitter(
            all_taxa=taxa,
            patristic_matrix=D,
            withholding_engine=withholding_engine,
            mode="order",
            df_pheno=df_pheno,
            min_clade_size=self.min_clade_size,
            d_buffer=self.d_buffer,
        )
        phylo_folds = list(phylo_cv.split())
        phylo_folds_dict = {
            f.clade_name: {
                "train_count": f.n_train,
                "test_count": f.n_test,
                "buffer_count": f.n_buffer,
                "test_taxa": [taxa[i] for i in f.test_indices],
                "buffer_taxa": [taxa[i] for i in f.buffer_indices],
            }
            for f in phylo_folds
        }
        phylo_folds_path = self.data_dir / "cv_phylo_folds.json"
        with open(phylo_folds_path, "w") as f:
            json.dump(phylo_folds_dict, f, indent=2)
        print(f"[Phase 3 Pipeline] Exported Phylo-CV splits (d_buffer={self.d_buffer} Ma) to {phylo_folds_path.name}")

        # 4. Person B: Model Verification Smoke Tests
        print("\n[Phase 3 Pipeline] Step 4 (Person B): Verifying 4 model families on sample fold...")
        feature_cols = [c for c in df_base.columns if c != ID_COL]
        X = df_base[feature_cols].values.astype(np.float32)
        y = df_pheno[self.target_trait].values.astype(np.float32)

        sample_fold = phylo_folds[0]
        val_engine = ValidationEngine()

        models = [
            ("Ridge Regression", RidgeRegressionPipeline()),
            ("Random Forest", RandomForestPipeline(n_estimators=20, n_jobs=-1)),
            ("XGBoost Regressor", XGBoostPipeline(n_estimators=20, n_jobs=-1)),
            ("PhyloGNN (PyG)", PhyloGNNPipeline(epochs=10)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m_name, m_inst in models:
                res = val_engine.run_fold(m_inst, X, y, sample_fold)
                print(
                    f"  [Model Check] {m_name:<20} | Test Clade: {sample_fold.clade_name} | "
                    f"R^2={res['r2']:.4f} | RMSE={res['rmse']:.4f} | MAE={res['mae']:.4f}"
                )

        # 5. SYNC BARRIER 3: Clade Leakage Audit Gate
        print("\n[Phase 3 Pipeline] Step 5: Executing SYNC BARRIER 3 Audit Gate...")
        report_path = self.data_dir / "sync_barrier_3_report.json"
        passed, report = audit_sync_barrier_3(
            data_dir=self.data_dir,
            d_buffer=self.d_buffer,
            min_clade_size=self.min_clade_size,
            target_trait=self.target_trait,
            report_out_path=report_path,
        )

        if not passed:
            raise RuntimeError(f"Sync Barrier 3 Audit Failed: {report['errors']}")

        print("\n" + "=" * 70)
        print("[Phase 3 Pipeline] SUCCESS: SYNC BARRIER 3 AUDIT GATE PASSED.")
        print(f"[Phase 3 Pipeline] Clade Leakage Audit Locked: {report['clade_leakage_audit_locked']}")
        print(f"[Phase 3 Pipeline] Buffer Isolation Verified: {report['buffer_isolation_verified']}")
        print(f"[Phase 3 Pipeline] Verified Orders: {len(report['monophyletic_clades_verified'])}")
        print(f"[Phase 3 Pipeline] Report written to: {report_path}")
        print("=" * 70)

        return {
            "status": "PASS",
            "report": report,
            "artifacts": {
                "cv_random_folds": str(random_folds_path),
                "cv_phylo_folds": str(phylo_folds_path),
                "sync_barrier_3_report": str(report_path),
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHENIX Phase 3 Pipeline Runner")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--d-buffer", type=float, default=140.0, help="Patristic buffer distance (Ma)")
    parser.add_argument("--min-clade-size", type=int, default=50, help="Minimum species in an evaluation clade")
    parser.add_argument("--t-cut", type=float, default=65.0, help="Tree cut depth (Ma)")
    parser.add_argument("--target", type=str, default="adult_body_mass_g", help="Target phenotypic trait")
    args = parser.parse_args()

    pipeline = Phase3Pipeline(
        data_dir=Path(args.data_dir),
        d_buffer=args.d_buffer,
        min_clade_size=args.min_clade_size,
        t_cut_ma=args.t_cut,
        target_trait=args.target,
    )
    pipeline.run()
