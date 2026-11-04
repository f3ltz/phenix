"""
Eco-Evolutionary Forecaster for PHENIX
Compares Machine Learning trait forecasts against an Ornstein-Uhlenbeck (OU) REFERENCE
trajectory under future climate scenarios.

The OU reference is not observed data: its direction and size are set by the assumed
Bergmann coefficient (bergmann_coef, log10 g per degree C) together with alpha_selection and
years_forward_ma. Results are therefore a sensitivity analysis of the ML models against that
assumption, not a validation against ground truth. Note also that the SSP presets describe
warming by ~2070, while years_forward_ma is in millions of years.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.features.baseline import BaselineFeatureBuilder
from src.features.augmented import PhyloAugmentedFeatureBuilder
from src.models import (
    RandomForestPipeline,
    XGBoostPipeline,
    RidgeRegressionPipeline,
    compute_regression_metrics,
)
from src.phylogenetics.otol import PhyloNode
from src.simulation.evolution import MacroevolutionarySimulator
from src.simulation.scenarios import ClimateScenarioGenerator, ProteomicMutationGenerator


class EcoEvolutionaryForecaster:
    """
    Executes forward eco-evolutionary forecasting experiments:
      1. Perturbs climate rasters across IPCC scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5).
      2. In silico mutates BUSCO protein sequences along molecular clock timescales.
      3. Predicts future phenotypes using trained Baseline and Augmented models.
      4. Benchmarks ML forecasts against Ornstein-Uhlenbeck adaptive simulations (Bergmann's rule).
      5. Quantifies the Inertia Drag Factor (IDF) and Coordinate Locking Ratio (CLR).
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        target_trait: str = "adult_body_mass_g",
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.target_trait = target_trait
        self.random_state = random_state

        self.climate_gen = ClimateScenarioGenerator()
        self.genomic_gen = ProteomicMutationGenerator(random_state=random_state)
        self.simulator = MacroevolutionarySimulator(random_state=random_state)
        self.baseline_builder = BaselineFeatureBuilder()
        self.augmented_builder = PhyloAugmentedFeatureBuilder()

    def run_forecasting_experiment(
        self,
        scenarios: Optional[List[str]] = None,
        years_forward_ma: float = 0.5,
        alpha_selection: float = 0.15,
        sigma2_drift: float = 0.02,
        bergmann_coef: float = -0.015,
        n_replicates: int = 200,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the full forecasting and simulation benchmarking matrix across climate scenarios.
        """
        if scenarios is None:
            scenarios = ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]

        if verbose:
            print("\n" + "=" * 70)
            print("PHENIX: ECO-EVOLUTIONARY FORECASTING & SIMULATION BENCHMARK")
            print("=" * 70)

        # 1. Load locked processed datasets
        df_pheno = pd.read_csv(self.data_dir / "phenotypic_clean.csv")
        df_env = pd.read_csv(self.data_dir / "environmental_clean.csv")
        df_genomic = pd.read_csv(self.data_dir / "genomic_clean.csv")
        df_baseline = pd.read_csv(self.data_dir / "features_baseline.csv")
        df_augmented = pd.read_csv(self.data_dir / "features_augmented.csv")
        df_phylo = pd.read_csv(self.data_dir / "features_phylo_pcoa.csv")

        with open(self.data_dir / "phylo_tree_calibrated.nwk", "r") as f:
            tree_root = PhyloNode.from_newick(f.read())

        taxa = df_pheno[ID_COL].tolist()
        N = len(taxa)
        y_raw = df_pheno[self.target_trait].values.astype(np.float32)
        y_log_current = np.log10(np.maximum(y_raw, 1e-6))
        current_traits_dict = {taxa[i]: float(y_log_current[i]) for i in range(N)}

        # Feature column masks
        base_cols = [c for c in df_baseline.columns if c != ID_COL]
        aug_cols = [c for c in df_augmented.columns if c != ID_COL]

        X_base_cur = df_baseline[base_cols].values.astype(np.float32)
        X_aug_cur = df_augmented[aug_cols].values.astype(np.float32)

        if verbose:
            print(f"[Forecaster] Training baseline and augmented models across {N} mammalian species...")

        # 2. Train historical model representations
        # Baseline models (Env + Gen, no phylogeny)
        rf_base = RandomForestPipeline(n_estimators=50, max_depth=15, random_state=self.random_state)
        xgb_base = XGBoostPipeline(n_estimators=50, max_depth=6, learning_rate=0.08, random_state=self.random_state)
        ridge_base = RidgeRegressionPipeline(log_transform_target=True)

        rf_base.fit(X_base_cur, y_raw)
        xgb_base.fit(X_base_cur, y_raw)
        ridge_base.fit(X_base_cur, y_raw)

        # Augmented models (Env + Gen + 32 Phylo-PCoA)
        rf_aug = RandomForestPipeline(n_estimators=50, max_depth=15, random_state=self.random_state)
        xgb_aug = XGBoostPipeline(n_estimators=50, max_depth=6, learning_rate=0.08, random_state=self.random_state)
        ridge_aug = RidgeRegressionPipeline(log_transform_target=True)

        rf_aug.fit(X_aug_cur, y_raw)
        xgb_aug.fit(X_aug_cur, y_raw)
        ridge_aug.fit(X_aug_cur, y_raw)

        # Baseline & Augmented historical predictions
        y_pred_cur_rf_base = rf_base.predict(X_base_cur)
        y_pred_cur_rf_aug = rf_aug.predict(X_aug_cur)
        y_pred_cur_xgb_base = xgb_base.predict(X_base_cur)
        y_pred_cur_xgb_aug = xgb_aug.predict(X_aug_cur)

        scenario_results: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []

        # 3. Simulate future scenarios
        for scen in scenarios:
            preset = self.climate_gen.SSP_PRESETS[scen]
            delta_t = preset["delta_temp_c"]

            if verbose:
                print(f"\n[Forecasting Scenario: {scen}] (Delta T: +{delta_t:.1f}°C, Precip: x{preset['precip_factor']:.2f})")

            # Perturb environment & genomics
            df_env_future = self.climate_gen.get_preset_scenario(df_env, scen)
            df_gen_future = self.genomic_gen.apply_genomic_drift(df_genomic, years_forward_ma=years_forward_ma)

            # Reconstruct feature tables
            df_base_future = self.baseline_builder.construct_baseline_features(
                df_env=df_env_future, df_genomic=df_gen_future, target_taxa=taxa
            )
            df_aug_future = self.augmented_builder.assemble_augmented_features(
                df_baseline=df_base_future, df_phylo=df_phylo
            )

            X_base_fut = df_base_future[base_cols].values.astype(np.float32)
            X_aug_fut = df_aug_future[aug_cols].values.astype(np.float32)

            # ML Projections under future climate & drifted genomics
            pred_rf_base_fut = rf_base.predict(X_base_fut)
            pred_rf_aug_fut = rf_aug.predict(X_aug_fut)
            pred_xgb_base_fut = xgb_base.predict(X_base_fut)
            pred_xgb_aug_fut = xgb_aug.predict(X_aug_fut)

            # 4. Ground-Truth Macroevolutionary Simulation (Ornstein-Uhlenbeck Adaptive Selection)
            # Under Bergmann's rule, warming shifts optimal body mass downward
            shifted_optima_arr = self.simulator.compute_bergmann_optima(
                bio1_temperatures_c=df_env["bio1"].values,
                base_body_mass_log10=y_log_current,
                temperature_coefficient=bergmann_coef,
                delta_temp_c=delta_t,
            )
            future_optima_dict = {taxa[i]: float(shifted_optima_arr[i]) for i in range(N)}

            # Expected OU trajectory (noise-free): E[X(dt)] = theta + (x0 - theta) * exp(-alpha * dt).
            # A single stochastic draw is dominated by drift noise at these effect sizes, so the
            # reference is the expectation; replicate draws only quantify the noise spread.
            decay = np.exp(-alpha_selection * years_forward_ma)
            y_ou_fut = shifted_optima_arr + (y_log_current - shifted_optima_arr) * decay

            replicate_shifts = []
            for _ in range(n_replicates):
                ou_sim_future = self.simulator.forward_simulate_tips(
                    current_traits=current_traits_dict,
                    future_optima=future_optima_dict,
                    years_forward_ma=years_forward_ma,
                    alpha=alpha_selection,
                    sigma2=sigma2_drift,
                )
                replicate_shifts.append(np.mean([ou_sim_future[t] for t in taxa]) - np.mean(y_log_current))
            ou_shift_replicate_sd = float(np.std(replicate_shifts))

            # 5. Compute Diagnostic Shifts and Metrics
            # OU reference shift (expected value under the assumed Bergmann coefficient)
            mean_shift_ou = float(np.mean(y_ou_fut - y_log_current))

            # ML predicted shifts
            shift_rf_base = float(np.mean(pred_rf_base_fut - y_pred_cur_rf_base))
            shift_rf_aug = float(np.mean(pred_rf_aug_fut - y_pred_cur_rf_aug))
            shift_xgb_base = float(np.mean(pred_xgb_base_fut - y_pred_cur_xgb_base))
            shift_xgb_aug = float(np.mean(pred_xgb_aug_fut - y_pred_cur_xgb_aug))

            # Inertia Drag Factor (IDF): |ML shift| / |OU shift|
            # IDF ~ 1: Perfect adaptive tracking; IDF << 1: Heavy coordinate drag/inertia locking
            denom_ou = abs(mean_shift_ou) if abs(mean_shift_ou) > 1e-6 else 1e-6
            idf_rf_aug = abs(shift_rf_aug) / denom_ou
            idf_xgb_aug = abs(shift_xgb_aug) / denom_ou
            idf_rf_base = abs(shift_rf_base) / denom_ou
            idf_xgb_base = abs(shift_xgb_base) / denom_ou

            # Coordinate Locking Ratio (CLR): |Base Shift| / |Augmented Shift|
            # Measures how many times more responsive the unanchored model is compared to the coordinate-anchored model
            denom_rf = abs(shift_rf_aug) if abs(shift_rf_aug) > 1e-6 else 1e-6
            denom_xgb = abs(shift_xgb_aug) if abs(shift_xgb_aug) > 1e-6 else 1e-6
            clr_rf = abs(shift_rf_base) / denom_rf
            clr_xgb = abs(shift_xgb_base) / denom_xgb

            # Adaptive Tracking Error (ATE) vs the expected OU reference trajectory
            ate_rf_base = float(np.sqrt(np.mean((pred_rf_base_fut - y_ou_fut) ** 2)))
            ate_rf_aug = float(np.sqrt(np.mean((pred_rf_aug_fut - y_ou_fut) ** 2)))
            ate_xgb_base = float(np.sqrt(np.mean((pred_xgb_base_fut - y_ou_fut) ** 2)))
            ate_xgb_aug = float(np.sqrt(np.mean((pred_xgb_aug_fut - y_ou_fut) ** 2)))

            if verbose:
                print(
                    f"  • OU Reference Shift (assumed):      {mean_shift_ou:+.4f} log10(g) "
                    f"(single-draw noise SD: {ou_shift_replicate_sd:.4f})"
                )
                print(f"  • RF Augmented Shift (With Phylo):   {shift_rf_aug:+.4f} log10(g) (IDF: {idf_rf_aug:.3f})")
                print(f"  • RF Baseline Shift (No Phylo):      {shift_rf_base:+.4f} log10(g) (CLR: {clr_rf:.1f}x)")
                print(f"  • XGB Augmented Shift (With Phylo):  {shift_xgb_aug:+.4f} log10(g) (IDF: {idf_xgb_aug:.3f})")
                print(f"  • XGB Baseline Shift (No Phylo):     {shift_xgb_base:+.4f} log10(g) (CLR: {clr_xgb:.1f}x)")
                print(f"  • Augmented Adaptive Tracking Error: RF={ate_rf_aug:.4f}, XGB={ate_xgb_aug:.4f}")

            scenario_data = {
                "scenario": scen,
                "delta_temp_c": delta_t,
                "precip_factor": preset["precip_factor"],
                "mean_shift_ou_reference": mean_shift_ou,
                "ou_shift_replicate_sd": ou_shift_replicate_sd,
                "shifts": {
                    "rf_augmented": shift_rf_aug,
                    "rf_baseline": shift_rf_base,
                    "xgb_augmented": shift_xgb_aug,
                    "xgb_baseline": shift_xgb_base,
                },
                "inertia_drag_factors": {
                    "rf_augmented": idf_rf_aug,
                    "xgb_augmented": idf_xgb_aug,
                    "rf_baseline": idf_rf_base,
                    "xgb_baseline": idf_xgb_base,
                },
                "coordinate_locking_ratios": {
                    "rf": clr_rf,
                    "xgb": clr_xgb,
                },
                "adaptive_tracking_errors": {
                    "rf_augmented": ate_rf_aug,
                    "rf_baseline": ate_rf_base,
                    "xgb_augmented": ate_xgb_aug,
                    "xgb_baseline": ate_xgb_base,
                },
            }
            scenario_results[scen] = scenario_data

            summary_rows.append({
                "Scenario": scen,
                "Delta_T_Celsius": delta_t,
                "OU_Reference_Shift": mean_shift_ou,
                "OU_Shift_Replicate_SD": ou_shift_replicate_sd,
                "RF_Augmented_Shift": shift_rf_aug,
                "RF_Baseline_Shift": shift_rf_base,
                "XGB_Augmented_Shift": shift_xgb_aug,
                "XGB_Baseline_Shift": shift_xgb_base,
                "RF_Inertia_Drag_Factor": idf_rf_aug,
                "XGB_Inertia_Drag_Factor": idf_xgb_aug,
                "RF_Coordinate_Lock_Ratio": clr_rf,
                "XGB_Coordinate_Lock_Ratio": clr_xgb,
                "RF_Augmented_ATE": ate_rf_aug,
                "XGB_Augmented_ATE": ate_xgb_aug,
            })

        df_summary = pd.DataFrame(summary_rows)

        # 6. Export results
        results_payload = {
            "target_trait": self.target_trait,
            "species_count": N,
            "years_forward_ma": years_forward_ma,
            "alpha_selection": alpha_selection,
            "sigma2_drift": sigma2_drift,
            "bergmann_coef": bergmann_coef,
            "n_replicates": n_replicates,
            "reference_note": (
                "OU reference shifts are expected values under the assumed bergmann_coef, "
                "alpha_selection and years_forward_ma; they are not observed data."
            ),
            "scenarios": scenario_results,
        }

        out_json = self.data_dir / "simulation_forecast_results.json"
        with open(out_json, "w") as f:
            json.dump(results_payload, f, indent=2)

        out_csv = self.data_dir / "simulation_summary.csv"
        df_summary.to_csv(out_csv, index=False)

        if verbose:
            print(f"\n[Forecaster] Exported simulation artifacts to:")
            print(f"  - {out_json}")
            print(f"  - {out_csv}")

        return {
            "results_json": results_payload,
            "summary_df": df_summary,
        }
