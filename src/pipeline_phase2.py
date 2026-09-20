import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.phylogenetics.otol import OToLClient, PhyloNode
from src.phylogenetics.calibration import TreeCalibrator
from src.phylogenetics.patristic import PatristicMatrixEngine
from src.features.baseline import BaselineFeatureBuilder
from src.features.whitening import CholeskyWhitener
from src.features.graph import PhyloGraphConverter
from src.features.augmented import PhyloAugmentedFeatureBuilder
from src.validation.sync_barrier_2 import audit_sync_barrier_2


class Phase2Pipeline:
    """
    Unified Pipeline Orchestrator for Phase 2: Tree Calibration & Feature Transformation.
    Integrates:
      Person A: OToL Topology, TimeTree BLADJ Calibration, Patristic Matrix D, Phylo-PCoA Eigenmaps.
      Person B: Baseline Feature Set, Cholesky Whitening, PyG Tree Graph, Augmented Feature Set.
      Sync Barrier 2: Architectural & Feature Freeze Audit Gate.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        pcoa_components: int = 32,
        primary_approach: str = "Phylo-PCoA",
    ):
        self.data_dir = Path(data_dir)
        self.pcoa_components = pcoa_components
        self.primary_approach = primary_approach

        self.otol_client = OToLClient()
        self.calibrator = TreeCalibrator()
        self.patristic_engine = PatristicMatrixEngine()
        self.baseline_builder = BaselineFeatureBuilder()
        self.whitener = CholeskyWhitener()
        self.graph_converter = PhyloGraphConverter()
        self.augmented_builder = PhyloAugmentedFeatureBuilder()

    def run(
        self,
        pheno_df: Optional[pd.DataFrame] = None,
        env_df: Optional[pd.DataFrame] = None,
        gen_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Runs end-to-end Phase 2 transformations and exports all artifacts."""
        print("[Phase 2 Pipeline] Step 1: Loading clean Phase 1 tables...")
        if pheno_df is None:
            pheno_df = pd.read_csv(self.data_dir / "phenotypic_clean.csv")
        if env_df is None:
            env_df = pd.read_csv(self.data_dir / "environmental_clean.csv")
        if gen_df is None:
            gen_df = pd.read_csv(self.data_dir / "genomic_clean.csv")

        target_taxa = pheno_df[ID_COL].tolist()
        N = len(target_taxa)
        print(f"[Phase 2 Pipeline] Processing {N} locked mammalian taxa.")

        # --- Person A: Phylogenetics & Tree Calibration ---
        print("[Phase 2 Pipeline] Person A: Building mammalian consensus tree topology...")
        tree_root = self.otol_client.build_mammal_backbone_topology(pheno_df)

        print("[Phase 2 Pipeline] Person A: Calibrating divergence times via TimeTree BLADJ algorithm...")
        calibrated_tree = self.calibrator.calibrate_tree(tree_root)

        # Save calibrated Newick tree
        newick_str = calibrated_tree.to_newick(include_branch_lengths=True) + ";"
        tree_path = self.data_dir / "phylo_tree_calibrated.nwk"
        with open(tree_path, "w") as f:
            f.write(newick_str)

        print("[Phase 2 Pipeline] Person A: Computing patristic evolutionary distance matrix D...")
        D = self.patristic_engine.compute_patristic_distance_matrix(calibrated_tree, target_taxa)
        patristic_path = self.data_dir / "patristic_distance_matrix.npy"
        np.save(patristic_path, D)

        print("[Phase 2 Pipeline] Person A: Computing double-centered matrix B & Phylo-PCoA eigenmaps...")
        B = self.patristic_engine.compute_double_centered_matrix(D)
        df_pcoa, evals = self.patristic_engine.compute_phylo_pcoa_eigenmaps(
            B, target_taxa, n_components=self.pcoa_components
        )
        pcoa_path = self.data_dir / "features_phylo_pcoa.csv"
        df_pcoa.to_csv(pcoa_path, index=False)

        # --- Person B: Feature Transformations & Graph Conversion ---
        print("[Phase 2 Pipeline] Person B: Constructing Baseline Feature Set (Genomic + Environmental)...")
        df_baseline = self.baseline_builder.construct_baseline_features(
            df_env=env_df, df_genomic=gen_df, target_taxa=target_taxa
        )
        baseline_path = self.data_dir / "features_baseline.csv"
        df_baseline.to_csv(baseline_path, index=False)

        print("[Phase 2 Pipeline] Person B: Applying Cholesky whitening X* = L^-1 X...")
        df_whitened, cov_whitened = self.whitener.fit_transform(df_baseline)
        whitened_path = self.data_dir / "features_whitened.csv"
        df_whitened.to_csv(whitened_path, index=False)

        print("[Phase 2 Pipeline] Person B: Converting tree topology to PyTorch Geometric graph G=(V,E)...")
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

        print("[Phase 2 Pipeline] Person B: Assembling Phylogenetic-Augmented Feature Set...")
        df_augmented = self.augmented_builder.assemble_augmented_features(
            df_baseline=df_baseline, df_phylo=df_pcoa
        )
        augmented_path = self.data_dir / "features_augmented.csv"
        df_augmented.to_csv(augmented_path, index=False)

        # --- SYNC BARRIER 2: Architectural & Feature Freeze ---
        print("[Phase 2 Pipeline] SYNC BARRIER 2: Executing Architectural & Feature Freeze Audit Gate...")
        report_path = self.data_dir / "sync_barrier_2_report.json"
        passed, report = audit_sync_barrier_2(
            pheno_path=self.data_dir / "phenotypic_clean.csv",
            baseline_path=baseline_path,
            augmented_path=augmented_path,
            tree_path=tree_path,
            patristic_path=patristic_path,
            primary_approach=self.primary_approach,
            report_out_path=report_path,
        )

        if not passed:
            raise RuntimeError(f"Sync Barrier 2 Audit Gate Failed: {report['errors']}")

        print("[Phase 2 Pipeline] SUCCESS: Sync Barrier 2 PASS. Phase 2 Complete.")
        return {
            "status": "SUCCESS",
            "report": report,
            "paths": {
                "tree": tree_path,
                "patristic": patristic_path,
                "baseline": baseline_path,
                "whitened": whitened_path,
                "augmented": augmented_path,
                "graph": graph_path,
                "report": report_path,
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline: Tree Calibration & Feature Transformation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pcoa-components", type=int, default=32)
    parser.add_argument("--approach", type=str, default="Phylo-PCoA", choices=["Phylo-PCoA", "GNN", "Dual-Ensemble"])
    args = parser.parse_args()

    pipeline = Phase2Pipeline(
        data_dir=args.data_dir,
        pcoa_components=args.pcoa_components,
        primary_approach=args.approach,
    )
    result = pipeline.run()
    rep = result["report"]
    print("=" * 60)
    print("PHASE 2 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status: {rep['status']}")
    print(f"Locked Taxa: {rep['taxa_count']}")
    print(f"Calibrated Tree Tips: {rep['tree_tip_count']}")
    print(f"Baseline Features (Env + Gen): {rep['baseline_feature_count']}")
    print(f"Augmented Features (Baseline + Phylo): {rep['augmented_feature_count']}")
    print(f"Primary Architecture Locked: {rep['primary_approach']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
