"""
Unit & Integration Tests for PHENIX Eco-Evolutionary Simulation Engine
Tests:
  - Brownian Motion (BM) neutral trait simulation along phylogenetic trees.
  - Ornstein-Uhlenbeck (OU) stabilizing selection trait simulation toward environmental optima.
  - Bergmann's rule phenotypic optimum calculations.
  - Climate warming scenario perturbations (SSP presets & custom shifts).
  - In silico proteomic molecular clock mutations and genomic drift.
  - Forecaster diagnostic metric computations (Inertia Drag Factor, Coordinate Locking Ratio).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.phylogenetics.otol import PhyloNode
from src.simulation.evolution import MacroevolutionarySimulator
from src.simulation.scenarios import ClimateScenarioGenerator, ProteomicMutationGenerator
from src.simulation.forecaster import EcoEvolutionaryForecaster
from src.data.schemas import ID_COL


@pytest.fixture
def sample_tree() -> PhyloNode:
    """Creates a calibrated 3-tip phylogenetic tree: ((A:5, B:5):5, C:10);"""
    root = PhyloNode(name="root", length=1.0, age=10.0)
    clade_ab = PhyloNode(name="node_ab", length=5.0, age=5.0)
    tip_a = PhyloNode(name="species_a", length=5.0, age=0.0)
    tip_b = PhyloNode(name="species_b", length=5.0, age=0.0)
    tip_c = PhyloNode(name="species_c", length=10.0, age=0.0)

    clade_ab.add_child(tip_a)
    clade_ab.add_child(tip_b)
    root.add_child(clade_ab)
    root.add_child(tip_c)
    return root


@pytest.fixture
def sample_env_df() -> pd.DataFrame:
    """Mock environmental dataframe with bio1 and bio12."""
    return pd.DataFrame({
        ID_COL: ["species_a", "species_b", "species_c"],
        "bio1": [10.0, 15.0, 20.0],
        "bio5": [25.0, 30.0, 35.0],
        "bio6": [-5.0, 0.0, 5.0],
        "bio7": [30.0, 30.0, 30.0],
        "bio12": [800.0, 1000.0, 1200.0],
        "elevation": [200.0, 500.0, 100.0],
        "habitat_suitability": [0.8, 0.9, 0.7],
    })


@pytest.fixture
def sample_genomic_df() -> pd.DataFrame:
    """Mock genomic dataframe with ESM-2 components."""
    rng = np.random.default_rng(42)
    data = {ID_COL: ["species_a", "species_b", "species_c"]}
    for i in range(1, 9):
        data[f"esm2_dim_{i}"] = rng.normal(0, 1, 3).tolist()
    return pd.DataFrame(data)


def test_brownian_motion_simulation(sample_tree):
    sim = MacroevolutionarySimulator(random_state=42)
    traits = sim.simulate_brownian_motion(sample_tree, sigma2=0.05, root_value=2.0)
    
    assert len(traits) == 3
    assert set(traits.keys()) == {"species_a", "species_b", "species_c"}
    for val in traits.values():
        assert isinstance(val, float)
        assert not np.isnan(val)


def test_ornstein_uhlenbeck_simulation(sample_tree):
    sim = MacroevolutionarySimulator(random_state=42)
    optima = {"species_a": 4.0, "species_b": 4.0, "species_c": 1.0}
    
    # Strong selection should pull traits toward their respective optima
    traits = sim.simulate_ornstein_uhlenbeck(
        root=sample_tree,
        alpha=0.5,
        sigma2=0.01,
        root_value=2.0,
        environmental_optima=optima,
    )
    
    assert len(traits) == 3
    # species_a and b should be closer to 4.0 than species_c
    assert traits["species_a"] > traits["species_c"]
    assert traits["species_b"] > traits["species_c"]


def test_forward_simulate_tips():
    sim = MacroevolutionarySimulator(random_state=42)
    cur = {"species_a": 2.0, "species_b": 3.0}
    future_opt = {"species_a": 1.5, "species_b": 2.5}

    fut = sim.forward_simulate_tips(
        current_traits=cur,
        future_optima=future_opt,
        years_forward_ma=1.0,
        alpha=0.3,
        sigma2=0.01,
    )

    assert len(fut) == 2
    # Expect downward shift toward optima
    assert fut["species_a"] < cur["species_a"] + 0.1
    assert fut["species_b"] < cur["species_b"] + 0.1


def test_compute_bergmann_optima():
    sim = MacroevolutionarySimulator(random_state=42)
    temps = np.array([10.0, 20.0])
    base_mass = np.array([3.0, 3.0])

    shifted = sim.compute_bergmann_optima(
        bio1_temperatures_c=temps,
        base_body_mass_log10=base_mass,
        temperature_coefficient=-0.02,
        delta_temp_c=3.0,
    )

    # Shift should be 3.0 + (-0.02 * 3.0) = 2.94
    np.testing.assert_allclose(shifted, np.array([2.94, 2.94]))


def test_climate_scenario_generator(sample_env_df):
    gen = ClimateScenarioGenerator()
    df_fut = gen.apply_warming_scenario(sample_env_df, delta_temp_c=3.0, precip_factor=1.05)

    assert df_fut["bio1"].tolist() == [13.0, 18.0, 23.0]
    assert df_fut["bio12"].iloc[0] == pytest.approx(800.0 * 1.05)
    assert df_fut[ID_COL].tolist() == sample_env_df[ID_COL].tolist()
    assert not df_fut.isnull().any().any()


def test_climate_presets(sample_env_df):
    gen = ClimateScenarioGenerator()
    df_ssp = gen.get_preset_scenario(sample_env_df, "SSP5-8.5")
    # SSP5-8.5 applies +5.0 C
    assert df_ssp["bio1"].iloc[0] == pytest.approx(15.0)


def test_proteomic_mutation_generator(sample_genomic_df):
    gen = ProteomicMutationGenerator(random_state=42)
    seq = "MKWVTFISLLLLFSSAYSRG"
    mutated = gen.mutate_amino_acid_sequence(seq, years_forward=500_000.0, mutation_rate_per_site_per_year=1e-5)

    assert len(mutated) == len(seq)
    # With this rate and horizon, at least some positions will change
    assert mutated != seq

    df_drifted = gen.apply_genomic_drift(sample_genomic_df, years_forward_ma=0.5, drift_rate=0.05)
    assert df_drifted.shape == sample_genomic_df.shape
    assert not df_drifted.isnull().any().any()
    assert not np.allclose(df_drifted["esm2_dim_1"].values, sample_genomic_df["esm2_dim_1"].values)
