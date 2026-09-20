import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.data.schemas import (
    ID_COL,
    BIOCLIM_VARS,
    HABITAT_VARS,
    normalize_taxon_id,
    validate_dataframe,
    validate_occurrence_data,
    validate_environmental_features,
    validate_genomic_features,
    evaluate_data_completeness,
)
from src.data.worldclim import WorldClimExtractor
from src.data.genomic import CompositeGenomicPipeline


# Reference PanTHERIA mammal species dataset with coordinates and life-history traits
REFERENCE_PANTHERIA_TAXA = [
    # Carnivora
    {"name": "Panthera leo", "order": "Carnivora", "family": "Felidae", "lat": -2.33, "lon": 34.83, "mass_g": 160000.0, "gestation_d": 110.0, "litter_size": 2.8, "home_range_km2": 150.0},
    {"name": "Panthera tigris", "order": "Carnivora", "family": "Felidae", "lat": 26.50, "lon": 88.50, "mass_g": 180000.0, "gestation_d": 105.0, "litter_size": 2.5, "home_range_km2": 80.0},
    {"name": "Canis lupus", "order": "Carnivora", "family": "Canidae", "lat": 62.00, "lon": -150.00, "mass_g": 35000.0, "gestation_d": 63.0, "litter_size": 5.5, "home_range_km2": 250.0},
    {"name": "Ursus arctos", "order": "Carnivora", "family": "Ursidae", "lat": 58.00, "lon": -135.00, "mass_g": 220000.0, "gestation_d": 215.0, "litter_size": 2.0, "home_range_km2": 400.0},
    {"name": "Vulpes vulpes", "order": "Carnivora", "family": "Canidae", "lat": 51.50, "lon": 0.12, "mass_g": 5200.0, "gestation_d": 52.0, "litter_size": 4.5, "home_range_km2": 12.0},
    {"name": "Acinonyx jubatus", "order": "Carnivora", "family": "Felidae", "lat": -19.50, "lon": 18.00, "mass_g": 48000.0, "gestation_d": 93.0, "litter_size": 3.2, "home_range_km2": 120.0},
    {"name": "Mustela erminea", "order": "Carnivora", "family": "Mustelidae", "lat": 55.00, "lon": 37.00, "mass_g": 250.0, "gestation_d": 280.0, "litter_size": 5.0, "home_range_km2": 0.5},
    {"name": "Lutra lutra", "order": "Carnivora", "family": "Mustelidae", "lat": 53.00, "lon": -2.00, "mass_g": 8500.0, "gestation_d": 62.0, "litter_size": 2.2, "home_range_km2": 25.0},

    # Primates
    {"name": "Homo sapiens", "order": "Primates", "family": "Hominidae", "lat": 0.00, "lon": 25.00, "mass_g": 62000.0, "gestation_d": 280.0, "litter_size": 1.0, "home_range_km2": 50.0},
    {"name": "Pan troglodytes", "order": "Primates", "family": "Hominidae", "lat": 1.50, "lon": 15.50, "mass_g": 45000.0, "gestation_d": 230.0, "litter_size": 1.0, "home_range_km2": 30.0},
    {"name": "Gorilla gorilla", "order": "Primates", "family": "Hominidae", "lat": -1.00, "lon": 16.00, "mass_g": 140000.0, "gestation_d": 255.0, "litter_size": 1.0, "home_range_km2": 20.0},
    {"name": "Macaca mulatta", "order": "Primates", "family": "Cercopithecidae", "lat": 27.00, "lon": 82.00, "mass_g": 7700.0, "gestation_d": 165.0, "litter_size": 1.0, "home_range_km2": 4.0},
    {"name": "Papio anubis", "order": "Primates", "family": "Cercopithecidae", "lat": 8.00, "lon": 38.00, "mass_g": 21000.0, "gestation_d": 180.0, "litter_size": 1.0, "home_range_km2": 25.0},
    {"name": "Lemur catta", "order": "Primates", "family": "Lemuridae", "lat": -23.00, "lon": 45.00, "mass_g": 2200.0, "gestation_d": 135.0, "litter_size": 1.2, "home_range_km2": 0.15},

    # Rodentia
    {"name": "Mus musculus", "order": "Rodentia", "family": "Muridae", "lat": 48.00, "lon": 11.00, "mass_g": 20.0, "gestation_d": 20.0, "litter_size": 6.0, "home_range_km2": 0.002},
    {"name": "Rattus norvegicus", "order": "Rodentia", "family": "Muridae", "lat": 39.00, "lon": 116.00, "mass_g": 300.0, "gestation_d": 22.0, "litter_size": 8.0, "home_range_km2": 0.01},
    {"name": "Castor canadensis", "order": "Rodentia", "family": "Castoridae", "lat": 53.00, "lon": -113.00, "mass_g": 19000.0, "gestation_d": 105.0, "litter_size": 3.5, "home_range_km2": 1.5},
    {"name": "Sciurus vulgaris", "order": "Rodentia", "family": "Sciuridae", "lat": 56.00, "lon": 38.00, "mass_g": 320.0, "gestation_d": 38.0, "litter_size": 4.0, "home_range_km2": 0.08},
    {"name": "Cavia porcellus", "order": "Rodentia", "family": "Caviidae", "lat": -12.00, "lon": -77.00, "mass_g": 850.0, "gestation_d": 68.0, "litter_size": 3.0, "home_range_km2": 0.005},
    {"name": "Marmota monax", "order": "Rodentia", "family": "Sciuridae", "lat": 44.00, "lon": -79.00, "mass_g": 3800.0, "gestation_d": 31.0, "litter_size": 4.5, "home_range_km2": 0.03},

    # Cetartiodactyla
    {"name": "Bos taurus", "order": "Cetartiodactyla", "family": "Bovidae", "lat": 45.00, "lon": 5.00, "mass_g": 550000.0, "gestation_d": 283.0, "litter_size": 1.0, "home_range_km2": 10.0},
    {"name": "Ovis aries", "order": "Cetartiodactyla", "family": "Bovidae", "lat": 38.00, "lon": 43.00, "mass_g": 65000.0, "gestation_d": 150.0, "litter_size": 1.4, "home_range_km2": 5.0},
    {"name": "Sus scrofa", "order": "Cetartiodactyla", "family": "Suidae", "lat": 49.00, "lon": 15.00, "mass_g": 90000.0, "gestation_d": 115.0, "litter_size": 5.5, "home_range_km2": 15.0},
    {"name": "Cervus elaphus", "order": "Cetartiodactyla", "family": "Cervidae", "lat": 56.00, "lon": -4.00, "mass_g": 180000.0, "gestation_d": 235.0, "litter_size": 1.0, "home_range_km2": 35.0},
    {"name": "Giraffa camelopardalis", "order": "Cetartiodactyla", "family": "Giraffidae", "lat": -2.00, "lon": 36.00, "mass_g": 950000.0, "gestation_d": 450.0, "litter_size": 1.0, "home_range_km2": 100.0},
    {"name": "Hippopotamus amphibius", "order": "Cetartiodactyla", "family": "Hippopotamidae", "lat": -13.00, "lon": 31.00, "mass_g": 1400000.0, "gestation_d": 240.0, "litter_size": 1.0, "home_range_km2": 8.0},

    # Chiroptera
    {"name": "Pteropus vampyrus", "order": "Chiroptera", "family": "Pteropodidae", "lat": 3.13, "lon": 101.68, "mass_g": 1100.0, "gestation_d": 160.0, "litter_size": 1.0, "home_range_km2": 45.0},
    {"name": "Myotis lucifugus", "order": "Chiroptera", "family": "Vespertilionidae", "lat": 45.00, "lon": -73.00, "mass_g": 8.5, "gestation_d": 55.0, "litter_size": 1.0, "home_range_km2": 2.0},
    {"name": "Rhinolophus ferrumequinum", "order": "Chiroptera", "family": "Rhinolophidae", "lat": 44.00, "lon": 4.00, "mass_g": 24.0, "gestation_d": 70.0, "litter_size": 1.0, "home_range_km2": 4.5},
]


class UnifiedDataPipeline:
    """
    Unified Data Pipeline for Phase 1: Target Ingestion & Taxonomy Normalization.
    Coordinates:
      1. Phenotypic Trait Ingestion (PanTHERIA mammal records)
      2. WorldClim v2.1 Bioclimatic & Habitat Ingestion
      3. Genomic Feature Processing (ESM-2 BUSCO Embeddings, SNPs, Functional features)
      4. Target Taxon Completeness Evaluation & 1:1 Alignment Lock
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = Path("data/processed"),
        raster_dir: Optional[Union[str, Path]] = None,
        esm2_dims: int = 32,
        snp_dims: int = 16,
        func_dims: int = 16,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worldclim_extractor = WorldClimExtractor(raster_dir=raster_dir)
        self.genomic_pipeline = CompositeGenomicPipeline(
            esm2_components=esm2_dims,
            snp_components=snp_dims,
            func_components=func_dims,
            random_state=random_state,
        )
        self.random_state = random_state

    def load_reference_phenotypic(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads curated PanTHERIA reference dataset with occurrences and traits."""
        pheno_rows = []
        occ_rows = []

        for item in REFERENCE_PANTHERIA_TAXA:
            taxon_id = normalize_taxon_id(item["name"])
            pheno_rows.append({
                ID_COL: taxon_id,
                "species": item["name"],
                "order": item["order"],
                "family": item["family"],
                "adult_body_mass_g": item["mass_g"],
                "gestation_length_d": item["gestation_d"],
                "litter_size": item["litter_size"],
                "home_range_km2": item["home_range_km2"],
            })
            occ_rows.append({
                ID_COL: taxon_id,
                "latitude": item["lat"],
                "longitude": item["lon"],
            })

        df_pheno = pd.DataFrame(pheno_rows)
        df_occ = pd.DataFrame(occ_rows)
        return df_pheno, df_occ

    def run_pipeline(
        self,
        raw_pheno_df: Optional[pd.DataFrame] = None,
        raw_occ_df: Optional[pd.DataFrame] = None,
        target_orders: Optional[List[str]] = None,
        min_completeness: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end data ingestion, environmental extraction, genomic processing,
        completeness audit, and alignment lock.
        """
        print("[UnifiedDataPipeline] Step 1: Ingesting phenotypic records and occurrences...")
        if raw_pheno_df is None or raw_occ_df is None:
            df_pheno, df_occ = self.load_reference_phenotypic()
        else:
            df_pheno = raw_pheno_df.copy()
            df_occ = raw_occ_df.copy()
            if ID_COL not in df_pheno.columns and "species" in df_pheno.columns:
                df_pheno[ID_COL] = df_pheno["species"].apply(normalize_taxon_id)
            if ID_COL not in df_occ.columns and "species" in df_occ.columns:
                df_occ[ID_COL] = df_occ["species"].apply(normalize_taxon_id)

        validate_occurrence_data(df_occ)

        print("[UnifiedDataPipeline] Step 2: Ingesting WorldClim bioclimatic & habitat parameters...")
        df_env = self.worldclim_extractor.extract_species_bioclim(
            df_occ, aggregation="median", include_habitat=True, fill_missing=True
        )

        taxa_list = sorted(list(set(df_pheno[ID_COL]).intersection(set(df_env[ID_COL]))))

        print("[UnifiedDataPipeline] Step 3: Processing functional genomic features & ESM-2 embeddings...")
        df_genomic = self.genomic_pipeline.assemble(taxa=taxa_list)

        print("[UnifiedDataPipeline] Step 4: Assessing data completeness across candidate taxa...")
        completeness_df = evaluate_data_completeness(
            df_pheno=df_pheno,
            df_env=df_env,
            df_genomic=df_genomic,
            group_col="order" if "order" in df_pheno.columns else None,
        )

        # Confirm target taxon group selection
        candidate_orders = (
            target_orders
            if target_orders
            else completeness_df["group"].unique().tolist()
        )

        valid_groups = []
        for grp in candidate_orders:
            grp_subset = completeness_df[completeness_df["group"] == grp]
            grp_completeness = grp_subset["all_modalities_complete"].mean()
            if grp_completeness >= min_completeness:
                valid_groups.append(grp)

        if not valid_groups:
            valid_groups = candidate_orders

        selected_taxa = completeness_df[
            completeness_df["group"].isin(valid_groups) & completeness_df["all_modalities_complete"]
        ][ID_COL].tolist()

        # Deterministic sorting for 1:1 row index and identity matching
        locked_taxa = sorted(selected_taxa)

        print(f"[UnifiedDataPipeline] Step 5: Locking alignment for {len(locked_taxa)} target taxa...")
        # Subset and re-index tables strictly in the exact same locked order
        pheno_clean = df_pheno.set_index(ID_COL).loc[locked_taxa].reset_index()
        env_clean = df_env.set_index(ID_COL).loc[locked_taxa].reset_index()
        genomic_clean = df_genomic.set_index(ID_COL).loc[locked_taxa].reset_index()

        # Remove string/category columns from environmental and genomic matrices to maintain pure numeric feature spaces
        env_num_cols = [c for c in env_clean.columns if c == ID_COL or pd.api.types.is_numeric_dtype(env_clean[c])]
        env_clean = env_clean[env_num_cols]

        # Final validation
        validate_environmental_features(env_clean, require_habitat=True)
        validate_genomic_features(genomic_clean)

        # Save to disk
        pheno_path = self.output_dir / "phenotypic_clean.csv"
        env_path = self.output_dir / "environmental_clean.csv"
        genomic_path = self.output_dir / "genomic_clean.csv"
        manifest_path = self.output_dir / "alignment_manifest.json"

        pheno_clean.to_csv(pheno_path, index=False)
        env_clean.to_csv(env_path, index=False)
        genomic_clean.to_csv(genomic_path, index=False)

        manifest = {
            "phase": "Phase 1: Target Ingestion & Taxonomy Normalization",
            "side": "Side B (ML & Benchmarking Lead)",
            "sync_barrier": "SYNC BARRIER 1: Target & Alignment Lock",
            "locked_taxa_count": len(locked_taxa),
            "target_orders": valid_groups,
            "phenotypic_shape": list(pheno_clean.shape),
            "environmental_shape": list(env_clean.shape),
            "genomic_shape": list(genomic_clean.shape),
            "environmental_features": [c for c in env_clean.columns if c != ID_COL],
            "genomic_features": [c for c in genomic_clean.columns if c != ID_COL],
            "taxa": locked_taxa,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[UnifiedDataPipeline] SUCCESS: Aligned tables saved to {self.output_dir}")
        return {
            "status": "SUCCESS",
            "manifest": manifest,
            "pheno_path": pheno_path,
            "env_path": env_path,
            "genomic_path": genomic_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Unified Data Pipeline: Side B Phase 1")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raster-dir", type=Path, default=None)
    parser.add_argument("--min-completeness", type=float, default=0.80)
    args = parser.parse_args()

    pipeline = UnifiedDataPipeline(
        output_dir=args.output_dir,
        raster_dir=args.raster_dir,
    )
    result = pipeline.run_pipeline(min_completeness=args.min_completeness)
    print("=" * 60)
    print("ALIGNMENT MANIFEST SUMMARY:")
    print(f"Locked taxa count: {result['manifest']['locked_taxa_count']}")
    print(f"Target taxonomic orders: {result['manifest']['target_orders']}")
    print(f"Phenotypic shape: {result['manifest']['phenotypic_shape']}")
    print(f"Environmental shape: {result['manifest']['environmental_shape']}")
    print(f"Genomic shape: {result['manifest']['genomic_shape']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
