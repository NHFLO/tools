"""
Geographic Data Processing functions.

Main module for converting, formatting, and validating geographic data files.
"""

from pathlib import Path
from shutil import copy2
from typing import ClassVar

import geopandas as gpd
import numpy as np
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
        # ".json": "GeoJSON",
        # ".csv": "CSV",
        # ".xlsx": "Excel",
        # ".xls": "Excel",
        # ".ods": "OpenDocument Spreadsheet",
    }

    def __init__(self, target_crs: str = "EPSG:28992", rounding_interval: int = 1000):
        """Initialize converter with settings."""
        self.target_crs = target_crs
        self.rounding_interval = rounding_interval

    def convert_file(
        self,
        *,
        gdf: gpd.GeoDataFrame | None = None,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
        wkt_column: str | None = None,
        coordinate_precision: int = 2,
        overwrite_with_target_crs: bool = True,
    ) -> Path:
        """Convert a single file to GeoJSON format."""
        if gdf is None == input_path is None:
            msg = "Either GeoDataFrame or input path must be provided"
            raise GeoProcessingError(msg)

        if gdf is not None:
            gdf = gpd.GeoDataFrame(gdf).copy()
        else:
            input_path = Path(input_path)
            if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                msg = f"Unsupported format: {input_path.suffix}"
                raise GeoProcessingError(msg)

            # Read input file
            if input_path.suffix.lower() in {".csv", ".xls", ".xlsx", ".ods"}:
                gdf = read_tabular(input_path, x_column, y_column, wkt_column)
            else:
                # For all other formats, use standard GeoPandas reading
                if len(gpd.list_layers(input_path)) != 1:
                    msg = "Multiple layers in file not supported"
                    raise GeoProcessingError(msg)
                gdf = gpd.read_file(input_path)
                if gdf.crs is None:
                    gdf.crs = self.target_crs

        # Process GeoDataFrame
        _gdf = gdf.to_crs(self.target_crs)
        if ~np.isfinite(_gdf.geometry.total_bounds).all():
            if overwrite_with_target_crs:
                gdf.crs = self.target_crs
            else:
                msg = "CRS conversion failed"
                raise GeoProcessingError(msg)
        else:
            gdf = _gdf

        # Format and optimize geometries
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf.dropna(subset=["geometry"])
        geometry = gdf.make_valid()
        geometry = geometry.set_precision(10**-coordinate_precision, mode="valid_output")
        geometry = geometry.simplify(tolerance=10**-coordinate_precision, preserve_topology=True)
        gdf.set_geometry(geometry, inplace=True)

        # Optimize DataFrame
        gdf = optimize_dataframe(gdf, exclude_columns=["geometry"])

        # Save output
        if output_path is None:
            output_path = input_path.with_suffix(".geojson")
        output_path = Path(output_path)
        gdf.to_file(output_path, driver="GeoJSON", coordinate_precision=coordinate_precision + 1, write_bbox="yes")
        return output_path

    def convert_folder(
        self,
        *,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
        coordinate_precision: int = 2,
        overwrite_with_target_crs: bool = True,
    ) -> dict[str, list[Path]]:
        """
        Convert all supported files in a folder structure to GeoJSON format and copy all other files to maintain complete folder structure.

        Parameters
        ----------
        input_folder : Union[str, Path]
            Root folder containing files to convert
        output_folder : Optional[Union[str, Path]]
            Root folder for output files. If None, creates 'converted' in input folder
        c

        Returns
        -------
        Dict[str, List[Path]]
            Dictionary with lists of converted, copied, and failed files
        """
        input_folder = Path(input_folder)
        if output_folder is None:
            output_folder = input_folder / "converted"
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        results = {"converted": [], "copied": [], "failed": []}

        # Find all files recursively
        convertible_files = []
        for ext in self.SUPPORTED_FORMATS:
            convertible_files.extend(input_folder.rglob(f"*{ext}"))

        excluded_shape_extensions = {
            ".sbn",
            ".sbx",
            ".shx",
            ".dbf",
            ".prj",
            ".shp.xml",
            ".cpg",
            ".qix",
            ".qpj",
            ".par",
            ".ung",
            ".qmd",
        }
        excl_convertable_shape_files = {f.with_suffix(s) for f in convertible_files for s in excluded_shape_extensions}

        # Find all other files
        all_files = set(input_folder.rglob("*"))
        other_files = {
            f for f in all_files if f.is_file() and f not in convertible_files and f not in excl_convertable_shape_files
        }

        for input_file in convertible_files:
            try:
                # Calculate relative path to maintain folder structure
                rel_path = input_file.relative_to(input_folder)
                output_path = output_folder / rel_path.with_suffix(".geojson")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert file
                converted_path = self.convert_file(
                    input_path=input_file,
                    output_path=output_path,
                    coordinate_precision=coordinate_precision,
                    overwrite_with_target_crs=overwrite_with_target_crs,
                )
                results["converted"].append(converted_path)

            except Exception:  # noqa: BLE001
                results["failed"].append(input_file)
                raise ValueError  # Reraise exception for debugging  # noqa: B904

        # Copy other files
        for input_file in other_files:
            try:
                rel_path = input_file.relative_to(input_folder)
                output_path = output_folder / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                copy2(input_file, output_path)
                results["copied"].append(output_path)

            except Exception:  # noqa: BLE001
                results["failed"].append(input_file)

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


def print_results(results):
    """Print the conversion result to standard output."""
    if results["converted"]:
        print("\nConverted files:")  # noqa: T201
        for path in results["converted"]:
            print(f"- {path}")  # noqa: T201

    if results["copied"]:
        print("\nCopied files:")  # noqa: T201
        for path in results["copied"]:
            print(f"- {path}")  # noqa: T201

    if results["failed"]:
        print("\nFailed files:")  # noqa: T201
        for path in results["failed"]:
            print(f"- {path}")  # noqa: T201


def main():
    """Demonstrate usage of the geographic processing suite."""
    converter = GeoConverter()

    # Convert folder with all files
    mockup_path = Path("/Users/bdestombe/Projects/NHFLO/data/src/nhflodata/data/mockup")
    input_folder = mockup_path / "doorsnedes_nh/v1.0.0"
    output_folder = mockup_path / "doorsnedes_nh/v2.0.0"
    results = converter.convert_folder(
        input_folder=input_folder, output_folder=output_folder, coordinate_precision=1, overwrite_with_target_crs=True
    )
    print_results(results)

    input_folder = mockup_path / "bodemlagen_pwn_nhdz/v1.0.0"
    output_folder = mockup_path / "bodemlagen_pwn_nhdz/v2.0.0"
    results = converter.convert_folder(input_folder=input_folder, output_folder=output_folder, coordinate_precision=1)
    print_results(results)

    # Convert folder with all files
    input_folder = mockup_path / "bodemlagen_pwn_bergen/v1.0.0"
    output_folder = mockup_path / "bodemlagen_pwn_bergen/v2.0.0"
    results = converter.convert_folder(
        input_folder=input_folder, output_folder=output_folder, coordinate_precision=1, overwrite_with_target_crs=True
    )
    print_results(results)

    input_folder = mockup_path / "bodemlagen_pwn_2024/temp"
    output_folder = mockup_path / "bodemlagen_pwn_2024/v1.0.0"
    results = converter.convert_folder(
        input_folder=input_folder, output_folder=output_folder, coordinate_precision=1, overwrite_with_target_crs=True
    )
    print_results(results)


if __name__ == "__main__":
    main()
