"""
Geographic Data Processing functions.

Main module for converting, formatting, and validating geographic data files.
"""

from pathlib import Path
from shutil import copy2
from typing import ClassVar

import geopandas as gpd
import pandas as pd

from nhflotools.geoconverter.utils import (
    calculate_folder_bounds,
    optimize_dataframe,
    read_tabular,
    validate_crs,
    validate_geometry,
)


class GeoProcessingError(Exception):
    """Base exception for geographic processing errors."""


class GeoConverter:
    """Convert and process geographic data files."""

    SUPPORTED_FORMATS: ClassVar[dict[str, str]] = {
        ".shp": "Shapefile",
        ".gpkg": "GeoPackage",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".csv": "CSV",
        ".xlsx": "Excel",
    }

    def __init__(self, target_crs: str = "EPSG:28992", rounding_interval: int = 1000):
        """Initialize converter with settings."""
        self.target_crs = target_crs
        self.rounding_interval = rounding_interval

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
        wkt_column: str | None = None,
    ) -> Path:
        """Convert a single file to GeoJSON format."""
        input_path = Path(input_path)
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            msg = f"Unsupported format: {input_path.suffix}"
            raise GeoProcessingError(msg)

        if output_path is None:
            output_path = input_path.with_suffix(".geojson")
        output_path = Path(output_path)

        # Read input file
        if input_path.suffix.lower() in {'.csv', '.xls', '.xlsx', '.ods'}:
            gdf = read_tabular(
                input_path,
                x_column,
                y_column,
                wkt_column
            )
        else:
            # For all other formats, use standard GeoPandas reading
            gdf = gpd.read_file(input_path)

        # Process GeoDataFrame
        gdf = gdf.to_crs(self.target_crs)

        # Format and optimize
        gdf.geometry = gdf.geometry.make_valid()
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf.dropna(subset=["geometry"])
        gdf.geometry = gdf.geometry.simplify(tolerance=0.01, preserve_topology=True)
        gdf = optimize_dataframe(gdf, exclude_columns=["geometry"])

        # Save output
        gdf.to_file(output_path, driver="GeoJSON")
        return output_path

    def convert_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path | None = None
    ) -> dict[str, list[Path]]:
        """
        Convert all supported files in a folder structure to GeoJSON format and copy all other files to maintain complete folder structure.

        Parameters
        ----------
        input_folder : Union[str, Path]
            Root folder containing files to convert
        output_folder : Optional[Union[str, Path]]
            Root folder for output files. If None, creates 'converted' in input folder

        Returns
        -------
        Dict[str, List[Path]]
            Dictionary with lists of converted, copied, and failed files
        """
        input_folder = Path(input_folder)
        if output_folder is None:
            output_folder = input_folder / 'converted'
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        results = {
            'converted': [],
            'copied': [],
            'failed': []
        }

        # Find all files recursively
        convertible_files = []
        for ext in self.SUPPORTED_FORMATS:
            convertible_files.extend(input_folder.rglob(f'*{ext}'))

        # Find all other files
        all_files = set(input_folder.rglob('*'))
        other_files = {
            f for f in all_files
            if f.is_file() and f not in convertible_files
        }

        for input_file in convertible_files:
            try:
                # Calculate relative path to maintain folder structure
                rel_path = input_file.relative_to(input_folder)
                output_path = output_folder / rel_path.with_suffix('.geojson')
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert file
                converted_path = self.convert_file(input_file, output_path)
                results['converted'].append(converted_path)

            except Exception:  # noqa: BLE001
                results['failed'].append(input_file)

        # Copy other files
        for input_file in other_files:
            try:
                rel_path = input_file.relative_to(input_folder)
                output_path = output_folder / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                copy2(input_file, output_path)
                results['copied'].append(output_path)

            except Exception:  # noqa: BLE001
                results['failed'].append(input_file)

        return results


def validate_folder(folder_path: str | Path, rounding_interval: int = 1000) -> dict[str, list[str]]:
    """Validate all GeoJSON files in a folder."""
    folder_path = Path(folder_path)
    results = {"errors": [], "warnings": [], "info": []}

    try:
        bounds = calculate_folder_bounds(folder_path, rounding_interval)
        results["info"].append(f"Folder bounds: {bounds}")

        for file_path in folder_path.glob("*.geojson"):
            try:
                gdf = gpd.read_file(file_path)

                # Validate CRS
                is_valid, error = validate_crs(gdf)
                if not is_valid:
                    results["errors"].append(f"{file_path.name}: {error}")

                # Validate geometries
                invalid_geoms = []
                for idx, geom in enumerate(gdf.geometry):
                    is_valid, error = validate_geometry(geom)
                    if not is_valid:
                        invalid_geoms.append(f"Row {idx}: {error}")

                if invalid_geoms:
                    results["errors"].extend([f"{file_path.name}: {error}" for error in invalid_geoms])

                # Check bounds
                if bounds:
                    file_bounds = gdf.total_bounds
                    if (
                        file_bounds[0] < bounds["minx"]
                        or file_bounds[1] < bounds["miny"]
                        or file_bounds[2] > bounds["maxx"]
                        or file_bounds[3] > bounds["maxy"]
                    ):
                        results["errors"].append(f"{file_path.name}: Geometries outside bounds")

            except (pd.errors.ParserError, gpd.errors.GeoPandasError, ValueError, OSError) as e:
                results["errors"].append(f"{file_path.name}: {e!s}")

    except (pd.errors.ParserError, gpd.errors.GeoPandasError, ValueError, OSError) as e:
        results["errors"].append(f"Folder validation error: {e!s}")

    return results


def main():
    """Demonstrate usage of the geographic processing suite."""
    converter = GeoConverter()

    # Convert folder with all files
    results = converter.convert_folder("input_folder", "output_folder")

    # Print results

    if results['converted']:
        print("\nConverted files:")  # noqa: T201
        for path in results['converted']:
            print(f"- {path}")  # noqa: T201

    if results['copied']:
        print("\nCopied files:")  # noqa: T201
        for path in results['copied']:
            print(f"- {path}")  # noqa: T201

    if results['failed']:
        print("\nFailed files:")  # noqa: T201
        for path in results['failed']:
            print(f"- {path}")  # noqa: T201

if __name__ == "__main__":
    main()
