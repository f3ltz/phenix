"""
Unit Tests for PHENIX Visualization & Dashboard Graphics Engine
Verifies all formatting, charts, and interactive rendering utilities.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.graphics import (
    render_species_morphology_card,
    get_species_evolutionary_note,
    plot_size_comparison_silhouette,
    plot_mass_scale_gauge,
    plot_multi_trait_radar,
    plot_allometric_scaling_curve,
    plot_animal_forecast_real_units,
    plot_forecasting_comparison_horizontal,
    get_shrinkage_analogy,
    format_trait_value,
    TRAIT_CONFIGS,
    ORDER_ICONS,
)


@pytest.fixture
def sample_pheno_df():
    data_dir = Path("data/processed")
    if (data_dir / "phenotypic_clean.csv").exists():
        return pd.read_csv(data_dir / "phenotypic_clean.csv")
    
    # Fallback minimal synthetic dataframe
    return pd.DataFrame([
        {
            "canonical_taxon_id": "SP001",
            "species": "Panthera leo",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Panthera",
            "adult_body_mass_g": 158623.9,
            "gestation_length_d": 110.0,
            "litter_size": 2.5,
            "weaning_age_d": 90.0,
            "home_range_km2": 150.0,
            "trophic_level": 3.0,
            "max_longevity_m": 300.0,
        },
        {
            "canonical_taxon_id": "SP002",
            "species": "Artibeus cinereus",
            "order": "Chiroptera",
            "family": "Phyllostomidae",
            "genus": "Artibeus",
            "adult_body_mass_g": 12.7,
            "gestation_length_d": 115.0,
            "litter_size": 1.0,
            "weaning_age_d": 45.0,
            "home_range_km2": 0.05,
            "trophic_level": 1.0,
            "max_longevity_m": 120.0,
        },
    ])


def test_format_trait_value():
    assert "tonnes" in format_trait_value("adult_body_mass_g", 3_000_000.0)
    assert "kg" in format_trait_value("adult_body_mass_g", 150_000.0)
    assert "g" in format_trait_value("adult_body_mass_g", 12.7)
    assert "days" in format_trait_value("gestation_length_d", 110.0)
    assert "offspring" in format_trait_value("litter_size", 3.2)
    assert "km²" in format_trait_value("home_range_km2", 150.0)
    assert "ha" in format_trait_value("home_range_km2", 0.05)
    assert format_trait_value("adult_body_mass_g", float("nan")) == "N/A"
    assert format_trait_value("adult_body_mass_g", -5.0) == "N/A"


def test_get_shrinkage_analogy():
    analogy_large = get_shrinkage_analogy("Loxodonta africana", 3_500_000.0, -140_000.0, "adult_body_mass_g")
    assert analogy_large is not None
    assert "human" in analogy_large.lower() or "tissue" in analogy_large.lower()

    analogy_bat = get_shrinkage_analogy("Artibeus cinereus", 12.7, -1.0, "adult_body_mass_g")
    assert analogy_bat is not None
    assert "mice" in analogy_bat.lower() or "tissue" in analogy_bat.lower()


def test_get_species_evolutionary_note():
    note = get_species_evolutionary_note("Panthera leo", "Carnivora", 158623.9)
    assert isinstance(note, dict)
    assert "title" in note
    assert "description" in note
    assert "scale_text" in note
    assert len(note["title"]) > 0


def test_plot_size_comparison_silhouette():
    fig1 = plot_size_comparison_silhouette(158623.9, "Panthera leo")
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # Reversed arguments test
    fig2 = plot_size_comparison_silhouette("Panthera leo", 158623.9)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_plot_mass_scale_gauge():
    fig = plot_mass_scale_gauge(158623.9, "Panthera leo")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_multi_trait_radar(sample_pheno_df):
    sp_id = sample_pheno_df.iloc[0]["canonical_taxon_id"]
    fig = plot_multi_trait_radar(sample_pheno_df, sp_id)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_allometric_scaling_curve(sample_pheno_df):
    sp_id = sample_pheno_df.iloc[0]["canonical_taxon_id"]
    fig = plot_allometric_scaling_curve(sample_pheno_df, sp_id, "gestation_length_d", "Gestation")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_animal_forecast_real_units():
    # Without unanchored models
    fig = plot_animal_forecast_real_units(
        species_name="Artibeus cinereus",
        trait_col="adult_body_mass_g",
        trait_label="Adult Body Mass",
        base_val=12.7,
        pred_ou=11.8,
        pred_rf_aug=15.5,
        pred_xgb_aug=20.3,
        pred_rf_base=125.0,
        pred_xgb_base=193.0,
        pred_piml=12.0,
        show_unanchored=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # With unanchored models
    fig_unanchored = plot_animal_forecast_real_units(
        species_name="Artibeus cinereus",
        trait_col="adult_body_mass_g",
        trait_label="Adult Body Mass",
        base_val=12.7,
        pred_ou=11.8,
        pred_rf_aug=15.5,
        pred_xgb_aug=20.3,
        pred_rf_base=125.0,
        pred_xgb_base=193.0,
        pred_piml=12.0,
        show_unanchored=True,
    )
    assert isinstance(fig_unanchored, plt.Figure)
    plt.close(fig_unanchored)


def test_plot_forecasting_comparison_horizontal():
    fig = plot_forecasting_comparison_horizontal(
        delta_t=3.0,
        shift_ou=-0.03,
        shift_rf_aug=-0.005,
        shift_xgb_aug=-0.008,
        shift_rf_base=0.15,
        shift_xgb_base=0.20,
        shift_piml=-0.02,
        trait_label="Adult Body Mass",
        show_unanchored=True,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
