import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.data.pipeline import IngestionPipeline
from src.phylogenetics.otol import OToLClient, PhyloNode
from src.phylogenetics.calibration import TreeCalibrator
from src.phylogenetics.patristic import PatristicMatrixEngine
from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.phylogenetics.divergence import PhyloDivergenceTracker
from src.features.baseline import BaselineFeatureBuilder
from src.features.whitening import CholeskyWhitener
from src.features.graph import PhyloGraphConverter
from src.features.augmented import PhyloAugmentedFeatureBuilder
from src.validation.cv import RandomKFoldSplitter, PhyloCVSplitter
from src.models import (
    RidgeRegressionPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    PhyloGNNPipeline,
    ValidationEngine,
)
from src.models.experiment import BenchmarkMatrixRunner
from src.validation.sync_barrier_1 import audit_sync_barrier_1
from src.validation.sync_barrier_2 import audit_sync_barrier_2
from src.validation.sync_barrier_3 import audit_sync_barrier_3
from src.validation.sync_barrier_4 import audit_sync_barrier_4


class PhenixPipeline:
    """
    Unified Master Pipeline for PHENIX:
      Stage 1: Multimodal Data Ingestion & Taxonomic Alignment
      Stage 2: Evolutionary Geometry & Feature Transformations
      Stage 3: Macroevolutionary Validation Engine & Buffer Quarantine
      Stage 4: Experimental Benchmarking Matrix & Inflation Gap Analysis
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        target_trait: str = "adult_body_mass_g",
        d_buffer: float = 140.0,
        min_clade_size: int = 50,
        t_cut_ma: float = 65.0,
        pcoa_components: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.target_trait = target_trait
        self.d_buffer = d_buffer
        self.min_clade_size = min_clade_size
        self.t_cut_ma = t_cut_ma
        self.pcoa_components = pcoa_components

        # Underlying engines
        self.otol_client = OToLClient()
        self.calibrator = TreeCalibrator()
        self.patristic_engine = PatristicMatrixEngine()
        self.baseline_builder = BaselineFeatureBuilder()
        self.whitener = CholeskyWhitener()
        self.graph_converter = PhyloGraphConverter()
        self.augmented_builder = PhyloAugmentedFeatureBuilder()

    def run_ingest(self) -> Dict[str, Any]:
        """Stage 1: Ingests PanTHERIA, WorldClim rasters, and ESM-2 genomics."""
        print("\n" + "=" * 70)
        print("STAGE 1: MULTIMODAL DATA INGESTION & TAXONOMIC ALIGNMENT")
        print("=" * 70)
        pipeline = IngestionPipeline(output_dir=self.data_dir)
        outputs = pipeline.run_full_pantheria_pipeline()
        return outputs

    def run_transform(
        self,
        pheno_df: Optional[pd.DataFrame] = None,
        env_df: Optional[pd.DataFrame] = None,
        gen_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Stage 2: Calibrates timetree, calculates patristic matrix, PCoA, whitening, and graphs."""
        print("\n" + "=" * 70)
        print("STAGE 2: EVOLUTIONARY GEOMETRIES & FEATURE TRANSFORMATIONS")
        print("=" * 70)

        if pheno_df is None:
            pheno_df = pd.read_csv(self.data_dir / "phenotypic_clean.csv")
        if env_df is None:
            env_df = pd.read_csv(self.data_dir / "environmental_clean.csv")
        if gen_df is None:
            gen_df = pd.read_csv(self.data_dir / "genomic_clean.csv")

        target_taxa = pheno_df[ID_COL].tolist()
        N = len(target_taxa)
        print(f"[Stage 2] Processing {N} taxa across 27 mammalian orders.")

        # Divergence calibration
        print("[Stage 2] Reconstructing consensus topology and applying BLADJ calibration (177.0 Ma)...")
        tree_root = self.otol_client.build_mammal_backbone_topology(pheno_df)
        calibrated_tree = self.calibrator.calibrate_tree(tree_root)

        tree_path = self.data_dir / "phylo_tree_calibrated.nwk"
        with open(tree_path, "w") as f:
            f.write(calibrated_tree.to_newick(include_branch_lengths=True) + ";")

        # Patristic matrix D & PCoA eigenmaps
        print("[Stage 2] Computing patristic evolutionary distance matrix D (3,268 x 3,268)...")
        D = self.patristic_engine.compute_patristic_distance_matrix(calibrated_tree, target_taxa)
        patristic_path = self.data_dir / "patristic_distance_matrix.npy"
        np.save(patristic_path, D)

        print(f"[Stage 2] Computing Gower's double-centering & extracting {self.pcoa_components} Phylo-PCoA eigenmaps...")
        B = self.patristic_engine.compute_double_centered_matrix(D)
        df_pcoa, _ = self.patristic_engine.compute_phylo_pcoa_eigenmaps(
            B, target_taxa, n_components=self.pcoa_components
        )
        pcoa_path = self.data_dir / "features_phylo_pcoa.csv"
        df_pcoa.to_csv(pcoa_path, index=False)

        # Baseline & Whitening
        print("[Stage 2] Assembling Baseline features (85 vars) & applying Cholesky whitening...")
        df_baseline = self.baseline_builder.construct_baseline_features(
            df_env=env_df, df_genomic=gen_df, target_taxa=target_taxa
        )
        baseline_path = self.data_dir / "features_baseline.csv"
        df_baseline.to_csv(baseline_path, index=False)

        whitened_mat = self.whitener.fit_transform(
            df_baseline[[c for c in df_baseline.columns if c != ID_COL]].values
        )
        df_whitened = pd.DataFrame(whitened_mat, columns=[c for c in df_baseline.columns if c != ID_COL])
        df_whitened.insert(0, ID_COL, target_taxa)
        whitened_path = self.data_dir / "features_whitened.csv"
        df_whitened.to_csv(whitened_path, index=False)

        # PyG Graph & Augmented features
        print("[Stage 2] Building PyTorch Geometric graph G=(V, E) & Augmented feature set (117 vars)...")
        graph_dict = self.graph_converter.tree_to_pyg_graph(calibrated_tree, df_baseline)
        graph_path = self.data_dir / "phylo_graph.json"
        with open(graph_path, "w") as f:
            json.dump({
                "num_nodes": graph_dict["num_nodes"],
                "num_tips": graph_dict["num_tips"],
                "num_features": graph_dict["num_features"],
                "edge_index_shape": list(graph_dict["edge_index"].shape),
                "edge_attr_shape": list(graph_dict["edge_attr"].shape),
                "taxa": graph_dict["taxa"],
            }, f, indent=2)

        df_augmented = self.augmented_builder.assemble_augmented_features(
            df_baseline=df_baseline, df_phylo=df_pcoa
        )
        augmented_path = self.data_dir / "features_augmented.csv"
        df_augmented.to_csv(augmented_path, index=False)

        # Audit gate
        print("[Stage 2] Auditing Feature & Geometry Freeze Gate...")
        report_path = self.data_dir / "sync_barrier_2_report.json"
        passed, report = audit_sync_barrier_2(
            pheno_path=self.data_dir / "phenotypic_clean.csv",
            baseline_path=baseline_path,
            augmented_path=augmented_path,
            tree_path=tree_path,
            patristic_path=patristic_path,
            report_out_path=report_path,
        )
        if not passed:
            raise RuntimeError(f"Stage 2 Audit Failed: {report['errors']}")

        return {"status": "SUCCESS", "report": report}

    def run_validate(self) -> Dict[str, Any]:
        """Stage 3: Generates monophyletic withholdings, buffer zones, and CV generators."""
        print("\n" + "=" * 70)
        print("STAGE 3: MACROEVOLUTIONARY VALIDATION ENGINE & BUFFER QUARANTINE")
        print("=" * 70)

        df_pheno = pd.read_csv(self.data_dir / "phenotypic_clean.csv")
        D = np.load(self.data_dir / "patristic_distance_matrix.npy")
        with open(self.data_dir / "phylo_tree_calibrated.nwk", "r") as f:
            tree_root = PhyloNode.from_newick(f.read())

        taxa = df_pheno[ID_COL].tolist()

        # Withholding engine
        engine = MonophyleticWithholdingEngine(tree_root=tree_root)
        order_clades = engine.get_order_clades(df_pheno, min_clade_size=self.min_clade_size)
        t_cut_clusters = engine.get_t_cut_clusters(t_cut_ma=self.t_cut_ma)
        print(f"[Stage 3] Identified {len(order_clades)} orders (N >= {self.min_clade_size}) and {len(t_cut_clusters)} lineages at T_cut={self.t_cut_ma} Ma.")

        # Cross-validation generators
        random_cv = RandomKFoldSplitter(n_splits=10, random_state=42)
        random_folds = list(random_cv.split(taxa))
        with open(self.data_dir / "cv_random_folds.json", "w") as f:
            json.dump({f"fold_{f.fold_id}": {"train": [taxa[i] for i in f.train_indices], "test": [taxa[i] for i in f.test_indices]} for f in random_folds}, f, indent=2)

        phylo_cv = PhyloCVSplitter(
            all_taxa=taxa,
            patristic_matrix=D,
            withholding_engine=engine,
            mode="order",
            df_pheno=df_pheno,
            min_clade_size=self.min_clade_size,
            d_buffer=self.d_buffer,
        )
        phylo_folds = list(phylo_cv.split())
        with open(self.data_dir / "cv_phylo_folds.json", "w") as f:
            json.dump({f.clade_name: {"train_count": f.n_train, "test_count": f.n_test, "buffer_count": f.n_buffer, "test_taxa": [taxa[i] for i in f.test_indices], "buffer_taxa": [taxa[i] for i in f.buffer_indices]} for f in phylo_folds}, f, indent=2)

        # Audit gate
        print("[Stage 3] Auditing Clade Leakage & Buffer Isolation Gate...")
        report_path = self.data_dir / "sync_barrier_3_report.json"
        passed, report = audit_sync_barrier_3(
            data_dir=self.data_dir,
            d_buffer=self.d_buffer,
            min_clade_size=self.min_clade_size,
            target_trait=self.target_trait,
            report_out_path=report_path,
        )
        if not passed:
            raise RuntimeError(f"Stage 3 Audit Failed: {report['errors']}")

        return {"status": "SUCCESS", "report": report}

    def run_benchmark(self, random_splits: int = 5) -> Dict[str, Any]:
        """Stage 4: Executes 16-permutation benchmark matrix, divergence tracking, and inflation gap."""
        print("\n" + "=" * 70)
        print("STAGE 4: EXPERIMENTAL BENCHMARKING & PERFORMANCE INFLATION ANALYSIS")
        print("=" * 70)

        df_pheno = pd.read_csv(self.data_dir / "phenotypic_clean.csv")
        df_base = pd.read_csv(self.data_dir / "features_baseline.csv")
        df_aug = pd.read_csv(self.data_dir / "features_augmented.csv")
        D = np.load(self.data_dir / "patristic_distance_matrix.npy")
        with open(self.data_dir / "phylo_tree_calibrated.nwk", "r") as f:
            tree_root = PhyloNode.from_newick(f.read())

        taxa = df_pheno[ID_COL].tolist()
        tree_graph_dict = self.graph_converter.tree_to_pyg_graph(tree_root, df_base)
        divergence_tracker = PhyloDivergenceTracker(patristic_matrix=D, all_taxa=taxa)

        # Folds
        random_cv = RandomKFoldSplitter(n_splits=random_splits, random_state=42)
        random_folds = list(random_cv.split(taxa))

        engine = MonophyleticWithholdingEngine(tree_root=tree_root)
        phylo_cv = PhyloCVSplitter(
            all_taxa=taxa,
            patristic_matrix=D,
            withholding_engine=engine,
            mode="order",
            df_pheno=df_pheno,
            min_clade_size=self.min_clade_size,
            d_buffer=self.d_buffer,
        )
        phylo_folds = list(phylo_cv.split())

        # Matrix Runner
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

        # Export artifacts
        with open(self.data_dir / "benchmark_results.json", "w") as f:
            json.dump(benchmark_outputs["matrix_results"], f, indent=2)
        benchmark_outputs["summary_df"].to_csv(self.data_dir / "benchmark_summary.csv", index=False)
        benchmark_outputs["predictions_df"].to_csv(self.data_dir / "predictions_matrix.csv", index=False)
        with open(self.data_dir / "divergence_accuracy_tracking.json", "w") as f:
            json.dump(benchmark_outputs["divergence_tracking"], f, indent=2)

        # Audit gate
        print("[Stage 4] Auditing Benchmark Review Gate...")
        report_path = self.data_dir / "sync_barrier_4_report.json"
        passed, report = audit_sync_barrier_4(
            data_dir=self.data_dir,
            report_out_path=report_path,
        )
        if not passed:
            raise RuntimeError(f"Stage 4 Audit Failed: {report['errors']}")

        return {"status": "SUCCESS", "report": report}

    def run_all(self, random_splits: int = 5) -> Dict[str, Any]:
        """Runs the entire pipeline end-to-end."""
        print("\n" + "#" * 70)
        print("PHENIX: EXECUTING FULL PIPELINE END-TO-END")
        print("#" * 70)

        if not (self.data_dir / "phenotypic_clean.csv").exists():
            self.run_ingest()

        self.run_transform()
        self.run_validate()
        return self.run_benchmark(random_splits=random_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHENIX Unified Master Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "ingest", "transform", "validate", "benchmark"],
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--target", type=str, default="adult_body_mass_g", help="Target phenotypic trait")
    parser.add_argument("--d-buffer", type=float, default=140.0, help="Patristic buffer distance (Ma)")
    parser.add_argument("--min-clade-size", type=int, default=50, help="Minimum species in an evaluation clade")
    parser.add_argument("--t-cut", type=float, default=65.0, help="Tree cut depth (Ma)")
    parser.add_argument("--random-splits", type=int, default=5, help="Number of random CV folds")
    args = parser.parse_args()

    pipeline = PhenixPipeline(
        data_dir=Path(args.data_dir),
        target_trait=args.target,
        d_buffer=args.d_buffer,
        min_clade_size=args.min_clade_size,
        t_cut_ma=args.t_cut,
    )

    if args.stage == "ingest":
        pipeline.run_ingest()
    elif args.stage == "transform":
        pipeline.run_transform()
    elif args.stage == "validate":
        pipeline.run_validate()
    elif args.stage == "benchmark":
        pipeline.run_benchmark(random_splits=args.random_splits)
    elif args.stage == "all":
        pipeline.run_all(random_splits=args.random_splits)
