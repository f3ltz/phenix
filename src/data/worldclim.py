import os
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS, validate_occurrence_data

WORLDCLIM_URL_2_5M = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"

def download_worldclim_data(output_dir: Union[str, Path], resolution: str = "2.5m") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    zip_path = out_path / f"worldclim_{resolution}_bio.zip"
    
    if not zip_path.exists():
        print(f"Downloading WorldClim ({resolution}) to {zip_path}...")
        urllib.request.urlretrieve(WORLDCLIM_URL_2_5M, zip_path)
    
    extracted_flag = out_path / ".extracted"
    if not extracted_flag.exists():
        print(f"Extracting WorldClim rasters...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_path)
        extracted_flag.touch()
    
    return out_path

class WorldClimExtractor:
    def __init__(self, raster_dir: Union[str, Path]):
        self.raster_dir = Path(raster_dir)
        self.raster_paths = self._find_rasters()

    def _find_rasters(self) -> Dict[str, Path]:
        found = {}
        for var in BIOCLIM_VARS:
            patterns = [
                f"*_{var}.tif",
                f"*_{var}_*.tif",
                f"*{var}.bil",
            ]
            matches = []
            for pat in patterns:
                matches.extend(list(self.raster_dir.glob(pat)))
                matches.extend(list(self.raster_dir.glob(pat.upper())))
            
            if matches:
                found[var] = matches[0]
        return found

    def sample_point(self, lat: float, lon: float, raster_path: Path) -> Optional[float]:
        try:
            import rasterio
            with rasterio.open(raster_path) as src:
                coords = [(lon, lat)]
                val = list(src.sample(coords))[0][0]
                if src.nodata is not None and val == src.nodata:
                    return np.nan
                return float(val)
        except ImportError:
            raise ImportError("rasterio is required to sample GeoTIFF rasters. Please install rasterio.")

    def extract_species_bioclim(
        self,
        occurrence_df: pd.DataFrame,
        aggregation: str = "median",
        fill_missing: bool = True
    ) -> pd.DataFrame:
        validate_occurrence_data(occurrence_df)
        
        extracted_rows = []
        for _, row in occurrence_df.iterrows():
            taxon = row[ID_COL]
            lat = row["latitude"]
            lon = row["longitude"]
            rec = {ID_COL: taxon}
            for var in BIOCLIM_VARS:
                if var in self.raster_paths:
                    rec[var] = self.sample_point(lat, lon, self.raster_paths[var])
                else:
                    rec[var] = np.nan
            extracted_rows.append(rec)
        
        raw_env_df = pd.DataFrame(extracted_rows)
        
        if aggregation == "median":
            aggregated = raw_env_df.groupby(ID_COL).median(numeric_only=True).reset_index()
        elif aggregation == "mean":
            aggregated = raw_env_df.groupby(ID_COL).mean(numeric_only=True).reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        if fill_missing:
            for var in BIOCLIM_VARS:
                if var in aggregated.columns and aggregated[var].isnull().any():
                    col_mean = aggregated[var].dropna().mean()
                    val_to_fill = col_mean if not np.isnan(col_mean) else 0.0
                    aggregated[var] = aggregated[var].fillna(val_to_fill)

        return aggregated
