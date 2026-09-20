import unittest
import pandas as pd
import numpy as np
from src.data.schemas import ID_COL, BIOCLIM_VARS
from src.data.worldclim import (
    WorldClimExtractor,
    approximate_bioclim_from_coords,
    classify_biome,
)


class TestWorldClim(unittest.TestCase):
    def test_approximate_bioclim_from_coords_equator(self):
        profile = approximate_bioclim_from_coords(lat=0.0, lon=25.0, elevation=400.0)
        for var in BIOCLIM_VARS:
            self.assertIn(var, profile)
            self.assertFalse(np.isnan(profile[var]))
        self.assertIn("elevation", profile)
        # Equatorial annual mean temp should be warm (~25-28C)
        self.assertGreater(profile["bio1"], 20.0)
        # Annual precipitation should be high (>1000mm)
        self.assertGreater(profile["bio12"], 1000.0)

    def test_approximate_bioclim_from_coords_polar(self):
        profile = approximate_bioclim_from_coords(lat=70.0, lon=0.0)
        # Polar annual mean temp should be cold (<0C)
        self.assertLess(profile["bio1"], 5.0)

    def test_classify_biome(self):
        biome_rainforest = classify_biome(bio1=26.0, bio12=2200.0, elevation=200.0)
        self.assertEqual(biome_rainforest, "Tropical & Subtropical Moist Broadleaf Forests")

        biome_tundra = classify_biome(bio1=-5.0, bio12=300.0, elevation=100.0)
        self.assertEqual(biome_tundra, "Tundra")

        biome_desert = classify_biome(bio1=28.0, bio12=150.0, elevation=200.0)
        self.assertEqual(biome_desert, "Deserts & Xeric Shrublands")

        biome_montane = classify_biome(bio1=8.0, bio12=900.0, elevation=3500.0)
        self.assertEqual(biome_montane, "Montane Grasslands & Shrublands")

    def test_worldclim_extractor_aggregation(self):
        # Two occurrences for species_a, one for species_b
        occ_df = pd.DataFrame({
            ID_COL: ["species_a", "species_a", "species_b"],
            "latitude": [10.0, 12.0, -25.0],
            "longitude": [20.0, 22.0, 45.0],
        })
        extractor = WorldClimExtractor()
        env_df = extractor.extract_species_bioclim(occ_df, aggregation="median", include_habitat=True)

        self.assertEqual(len(env_df), 2)
        self.assertIn("species_a", env_df[ID_COL].values)
        self.assertIn("species_b", env_df[ID_COL].values)
        for var in BIOCLIM_VARS:
            self.assertIn(var, env_df.columns)
            self.assertFalse(env_df[var].isnull().any())
        self.assertIn("elevation", env_df.columns)
        self.assertIn("biome_class", env_df.columns)
        self.assertIn("habitat_suitability", env_df.columns)


if __name__ == "__main__":
    unittest.main()
