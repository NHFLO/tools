"""Utility functions for geographic data processing, validation, and conversion."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


class GeoToolsError(Exception):
    """Base exception for geo_tools errors."""



def validate_geometry(geometry) -> tuple[bool, str | None]:
    """Validate a geometry object."""
    if geometry is None:
        return False, "Null geometry"
    if not geometry.is_valid:
        return False, "Invalid geometry"
    if geometry.is_empty:
        return False, "Empty geometry"
    return True, None


def validate_crs(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:28992") -> tuple[bool, str | None]:
    """Validate CRS of a GeoDataFrame."""
    if gdf.crs is None:
        return False, "Missing CRS information"
    if gdf.crs.to_string() == target_crs:
        return True, None
    return False, f"Invalid CRS: {gdf.crs.to_string()}"


def round_bounds(bounds: dict[str, float], rounding_interval: int = 1000) -> dict[str, float]:
    """Validate and round extent bounds to specified interval."""
    required_keys = {"minx", "miny", "maxx", "maxy"}
    if not all(key in bounds for key in required_keys):
        msg = "Bounds dictionary missing required keys"
        raise GeoToolsError(msg)

    return {
        "minx": math.floor(bounds["minx"] / rounding_interval) * rounding_interval,
        "miny": math.floor(bounds["miny"] / rounding_interval) * rounding_interval,
        "maxx": math.ceil(bounds["maxx"] / rounding_interval) * rounding_interval,
        "maxy": math.ceil(bounds["maxy"] / rounding_interval) * rounding_interval,
    }


def optimize_dataframe(df: pd.DataFrame, exclude_columns: list[str] | None = None) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    use_cat_threshold = 0.1
    exclude_columns = exclude_columns or []
    result = df.copy()

    # Remove duplicate columns
    for dup_name in result.columns[result.columns.duplicated()]:
        a = result.loc[:, dup_name].values
        if ~np.all(a == a[:, [0]]):
            msg = f"Duplicate column '{dup_name}' has different values"
            raise ValueError(msg)

    result = result.loc[:, ~result.columns.duplicated()]

    # remove nonsense columns
    if len(result) > 1:
        result = result.loc[:, ~result.isnull().all(axis=0)]

        # remove columns with predictable indices
        mask = np.arange(len(result))[:, None] == result
        result = result.loc[:, ~mask.all(axis=0)]

        mask = np.arange(1, len(result) + 1)[:, None] == result
        result = result.loc[:, ~mask.all(axis=0)]

    for col in result.columns:
        if col in exclude_columns:
            continue
        col_type = result.dtypes[col]
        if col_type == "object":
            if result[col].nunique() / len(result) < use_cat_threshold:
                result[col] = result[col].astype("category")
        elif col_type == "float64":
            result[col] = pd.to_numeric(result[col], downcast="float")
        elif col_type == "int64":
            result[col] = pd.to_numeric(result[col], downcast="integer")
    return result


def calculate_folder_bounds(folder_path: str | Path, rounding_interval: int = 1000) -> dict[str, float]:
    """Calculate total bounds for all GeoJSON files in a folder."""
    folder_path = Path(folder_path)
    all_bounds = []

    geojson_files = list(folder_path.glob("*.geojson"))
    if not geojson_files:
        msg = f"No GeoJSON files found in {folder_path}"
        raise GeoToolsError(msg)

    for file_path in geojson_files:
        gdf = gpd.read_file(file_path)
        if not gdf.empty:
            all_bounds.append(gdf.total_bounds)

    bounds = np.array(all_bounds)
    total_bounds = {
        "minx": bounds[:, 0].min(),
        "miny": bounds[:, 1].min(),
        "maxx": bounds[:, 2].max(),
        "maxy": bounds[:, 3].max(),
    }

    return round_bounds(total_bounds, rounding_interval)

def read_tabular(
    self,
    input_path: Path,
    x_column: str | None,
    y_column: str | None,
    wkt_column: str | None,
) -> gpd.GeoDataFrame:
    """Read tabular data with geometry information.

    Supported file formats: CSV, Excel, ODS.

    Parameters
    ----------
    input_path : Path
        Path to the input file.
    x_column : str, optional
        Column name with x-coordinate information.
    y_column : str, optional
        Column name with y-coordinate information.
    wkt_column : str, optional
        Column name with WKT geometry information.

    Returns
    -------
    gpd.GeoDataFrame
    GeoDataFrame with geometry information.
    """
    # Read the file based on its extension
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    elif input_path.suffix == '.ods':
        df = pd.read_excel(input_path, engine='odf')
    else:
        msg = f"Unsupported file format: {input_path.suffix}"
        raise ValueError(msg)

    # Create geometry from coordinates or WKT
    if x_column and y_column:
        geometry = gpd.points_from_xy(df[x_column], df[y_column], crs=self.target_crs)
        df.drop(columns=[x_column, y_column], inplace=True)
    elif wkt_column:
        geometry = gpd.GeoSeries.from_wkt(df[wkt_column], crs=self.target_crs)
        df.drop(columns=[wkt_column], inplace=True)
    else:
        msg = "No geometry information provided for tabular data"
        raise ValueError(msg)
    return gpd.GeoDataFrame(df, geometry=geometry, crs=self.target_crs)
