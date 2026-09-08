import unittest
import pandas as pd
from src.data.schemas import normalize_taxon_id, validate_occurrence_data, validate_dataframe, ID_COL

class TestSchemas(unittest.TestCase):
    def test_normalize_taxon_id(self):
        self.assertEqual(normalize_taxon_id("Panthera leo"), "panthera_leo")
        self.assertEqual(normalize_taxon_id("Quercus-robur "), "quercus_robur")
        self.assertEqual(normalize_taxon_id("Canis lupus (Grey Wolf)"), "canis_lupus_grey_wolf")

    def test_validate_occurrence_data_valid(self):
        df = pd.DataFrame({
            ID_COL: ["taxon_a", "taxon_b"],
            "latitude": [12.5, -45.0],
            "longitude": [77.5, 120.0]
        })
        validate_occurrence_data(df)

    def test_validate_occurrence_data_invalid_coords(self):
        df = pd.DataFrame({
            ID_COL: ["taxon_a"],
            "latitude": [95.0],
            "longitude": [77.5]
        })
        with self.assertRaises(ValueError):
            validate_occurrence_data(df)

    def test_validate_occurrence_data_missing_id(self):
        df = pd.DataFrame({
            "species": ["taxon_a"],
            "latitude": [10.0],
            "longitude": [20.0]
        })
        with self.assertRaises(ValueError):
            validate_occurrence_data(df)

if __name__ == "__main__":
    unittest.main()
