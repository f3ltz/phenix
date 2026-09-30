import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.features.graph import PhyloGraphConverter
from src.validation.cv import RandomKFoldSplitter, PhyloCVSplitter, CVFold
from src.phylogenetics.divergence import PhyloDivergenceTracker
from src.models.experiment import BenchmarkMatrixRunner
from src.validation.sync_barrier_4 import audit_sync_barrier_4


class Phase4Pipeline:
    """
    Unified Pipeline Orchestrator for Phase 4: Experimental Benchmarking & Extrapolation.
    Coordinates:
      - Person A: Divergence tracking vs accuracy decay analysis across held-out clades.
      - Person B: 16-configuration benchmark matrix (4 models x 2 feature sets x 2 protocols),
                  Performance inflation gap calculation, out-of-fold predictions matrix.
      - Sync Barrier 4: Benchmark Review Audit Gate.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        target_trait: str = "adult_body_mass_g",
        d_buffer: float = 140.0,
        min_clade_size: int = 50,
        n_random_splits: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.target_trait = target_trait
        self.d_buffer = d_buffer
        self.min_clade_size = min_clade_size
        self.n_random_splits = n_random_splits

    def run(self) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("PHENIX PHASE 4: EXPERIMENTAL BENCHMARKING & EXTRAPOLATION")
        print("=" * 70)

        # 1. Load Processed Datasets & Trees
        print("[Phase 4 Pipeline] Step 1: Loading verified Phase 1, 2 & 3 frozen artifacts...")
        pheno_path = self.data_dir / "phenotypic_clean.csv"
        baseline_path = self.data_dir / "features_baseline.csv"
        augmented_path = self.data_dir / "features_augmented.csv"
        tree_path = self.data_dir / "phylo_tree_calibrated.nwk"
        patristic_path = self.data_dir / "patristic_distance_matrix.npy"

        for p in [pheno_path, baseline_path, augmented_path, tree_path, patristic_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required artifact not found: {p}")

        df_pheno = pd.read_csv(pheno_path)
        df_base = pd.read_csv(baseline_path)
        df_aug = pd.read_csv(augmented_path)
        D = np.load(patristic_path)

        with open(tree_path, "r") as f:
            tree_root = PhyloNode.from_newick(f.read())

        taxa = df_pheno[ID_COL].tolist()
        N = len(taxa)
        print(f"[Phase 4 Pipeline] Loaded {N} taxa across 27 mammalian orders.")

        # Build PyG graph dict
        print("[Phase 4 Pipeline] Converting phylogenetic tree topology to graph structure...")
        graph_converter = PhyloGraphConverter()
        tree_graph_dict = graph_converter.tree_to_pyg_graph(tree_root, df_base)

        # 2. Person A: Divergence Tracking Setup
        print("\n[Phase 4 Pipeline] Step 2 (Person A): Initializing evolutionary divergence tracker...")
        divergence_tracker = PhyloDivergenceTracker(patristic_matrix=D, all_taxa=taxa)

        # 3. Person B: Setup Validation Generators
        print("\n[Phase 4 Pipeline] Step 3 (Person B): Initializing CV fold generators...")
        random_cv = RandomKFoldSplitter(n_splits=self.n_random_splits, random_state=42)
        random_folds = list(random_cv.split(taxa))

        withholding_engine = MonophyleticWithholdingEngine(tree_root=tree_root)
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
        print(f"[Phase 4 Pipeline] Generated {len(random_folds)} Random CV folds and {len(phylo_folds)} Phylo-CV folds.")

        # 4. Person B: Execute Complete 16-Permutation Benchmark Matrix
        print("\n[Phase 4 Pipeline] Step 4 (Person B): Executing 16-permutation benchmark matrix...")
        runner = BenchmarkMatrixRunner(
            df_pheno=df_pheno,
            df_baseline=df_base,
            df_augmented=df_aug,
            random_folds=random_folds,
            phylo_folds=phylo_folds,
            target_trait=self.target_trait,
            tree_graph_dict=tree_graph_dict,
            divergence_tracker=divergence_tracker,
        )

        benchmark_outputs = runner.run_benchmark(verbose=True)
        matrix_results = benchmark_outputs["matrix_results"]
        df_summary = benchmark_outputs["summary_df"]
        df_pred = benchmark_outputs["predictions_df"]
        div_tracking = benchmark_outputs["divergence_tracking"]

        # 5. Export Benchmark Artifacts
        print("\n[Phase 4 Pipeline] Step 5: Exporting Phase 4 benchmark artifacts...")
        results_path = self.data_dir / "benchmark_results.json"
        summary_path = self.data_dir / "benchmark_summary.csv"
        pred_path = self.data_dir / "predictions_matrix.csv"
        div_path = self.data_dir / "divergence_accuracy_tracking.json"

        with open(results_path, "w") as f:
            json.dump(matrix_results, f, indent=2)
        print(f"  -> Exported benchmark matrix results to {results_path.name}")

        df_summary.to_csv(summary_path, index=False)
        print(f"  -> Exported benchmark summary table to {summary_path.name}")

        df_pred.to_csv(pred_path, index=False)
        print(f"  -> Exported out-of-fold predictions matrix ({len(df_pred)} species) to {pred_path.name}")

        with open(div_path, "w") as f:
            json.dump(div_tracking, f, indent=2)
        print(f"  -> Exported divergence vs accuracy tracking to {div_path.name}")

        # Print Benchmark Highlights
        print("\n" + "-" * 70)
        print("BENCHMARK MATRIX HIGHLIGHTS (Summary Table Preview):")
        print("-" * 70)
        display_cols = ["model", "feature_set", "protocol", "r2_mean", "rmse_mean", "inflation_gap_r2", "phylo_gain_r2"]
        print(df_summary[display_cols].to_string(index=False))
        print("-" * 70)

        # 6. SYNC BARRIER 4: Benchmark Review Audit Gate
        print("\n[Phase 4 Pipeline] Step 6: Executing SYNC BARRIER 4 Audit Gate...")
        report_path = self.data_dir / "sync_barrier_4_report.json"
        passed, report = audit_sync_barrier_4(
            data_dir=self.data_dir,
            results_path=results_path,
            summary_path=summary_path,
            predictions_path=pred_path,
            divergence_path=div_path,
            report_out_path=report_path,
        )

        if not passed:
            raise RuntimeError(f"Sync Barrier 4 Audit Gate Failed: {report['errors']}")

        print("\n" + "=" * 70)
        print("[Phase 4 Pipeline] SUCCESS: SYNC BARRIER 4 AUDIT GATE PASSED.")
        print(f"[Phase 4 Pipeline] Configurations Evaluated: {report['configurations_tested_count']}")
        print(f"[Phase 4 Pipeline] Performance Inflation Gap Verified: {report['inflation_gap_verified']}")
        print(f"[Phase 4 Pipeline] Phylogenetic Augmentation Gain Verified: {report['phylo_gain_verified']}")
        print(f"[Phase 4 Pipeline] Benchmark Review Locked: {report['benchmark_review_locked']}")
        print(f"[Phase 4 Pipeline] Report written to: {report_path}")
        print("=" * 70)

        return {
            "status": "PASS",
            "report": report,
            "artifacts": {
                "benchmark_results": str(results_path),
                "benchmark_summary": str(summary_path),
                "predictions_matrix": str(pred_path),
                "divergence_tracking": str(div_path),
                "sync_barrier_4_report": str(report_path),
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHENIX Phase 4 Pipeline Runner")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--target", type=str, default="adult_body_mass_g", help="Target phenotypic trait")
    parser.add_argument("--d-buffer", type=float, default=140.0, help="Patristic buffer distance (Ma)")
    parser.add_argument("--min-clade-size", type=int, default=50, help="Minimum species in an evaluation clade")
    parser.add_argument("--random-splits", type=int, default=10, help="Number of random CV folds")
    args = parser.parse_args()

    pipeline = Phase4Pipeline(
        data_dir=Path(args.data_dir),
        target_trait=args.target,
        d_buffer=args.d_buffer,
        min_clade_size=args.min_clade_size,
        n_random_splits=args.random_splits,
    )
    pipeline.run()
