"""
Extracts real WorldClim v2.1 GeoTIFF raster values for all 3,268 mammalian species,
saving the 100% real empirical bioclimatic table to data/processed/environmental_clean.csv.
"""

from pathlib import Path
import pandas as pd
from src.data.worldclim import WorldClimExtractor
from src.data.schemas import normalize_taxon_id

def build_real_environmental():
    print("Loading PanTHERIA species occurrence coordinates...")
    pheno = pd.read_csv("data/processed/phenotypic_clean.csv")
    pan = pd.read_csv("data/raw/phenotypic/pantheria.csv")
    pan["canonical_taxon_id"] = pan["species_original_name"].apply(normalize_taxon_id)

    # Match real coordinates
    merged = pd.merge(
        pheno[["canonical_taxon_id"]],
        pan[["canonical_taxon_id", "gr_midrangelat_dd", "gr_midrangelong_dd"]],
        on="canonical_taxon_id",
        how="left"
    )

    occ_df = pd.DataFrame({
        "canonical_taxon_id": merged["canonical_taxon_id"],
        "latitude": merged["gr_midrangelat_dd"],
        "longitude": merged["gr_midrangelong_dd"]
    })

    print("Extracting pixel-level data from 20 WorldClim v2.1 GeoTIFF rasters in data/raw/environmental...")
    extractor = WorldClimExtractor(raster_dir="data/raw/environmental")
    env_df = extractor.extract_species_bioclim(
        occ_df,
        aggregation="median",
        include_habitat=True,
        fill_missing=True
    )

    out_path = Path("data/processed/environmental_clean.csv")
    # Preserve exact column structure: canonical_taxon_id, bio1..bio19, elevation, habitat_suitability
    numeric_cols = [f"bio{i}" for i in range(1, 20)] + ["elevation", "habitat_suitability"]
    final_df = env_df[["canonical_taxon_id"] + numeric_cols].copy()
    final_df.to_csv(out_path, index=False)
    print(f"SUCCESS: Saved real WorldClim v2.1 table to {out_path} ({final_df.shape})")
    print(f"Zero NaNs: {final_df.isna().sum().sum() == 0}")
    print("Sample values:")
    print(final_df.head(3))

if __name__ == "__main__":
    build_real_environmental()
