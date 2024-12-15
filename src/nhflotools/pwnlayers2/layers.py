"""Module containing functions to retrieve PWN bodemlagen."""

from contextlib import redirect_stdout
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pykrige.ok
import xarray as xr
from flopy.utils.gridintersect import GridIntersect
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def get_pwn_aquitard_data(ds_regis: xr.Dataset, data_dir: Path, ix: GridIntersect, transition_length: float) -> dict:
    """
    Interpolate the thickness of the aquitard layers and the top of the aquitard layers using Kriging.

    The thickness of the aquitard layers is interpolated using the points in the file
    `dikte_aquitard/D{layer_name}/D{layer_name}_interpolation_points.geojson`.
    The top of the aquitard layers is interpolated using the points in the file
    `top_aquitard/T{layer_name}/T{layer_name}_interpolation_points.geojson`.
    The mask of the aquitard layers is defined in the file
    `dikte_aquitard/D{layer_name}/D{layer_name}_mask_combined.geojson`.

    Parameters
    ----------
    ds_regis : xr.Dataset
        The REGIS modellayer that contains the vertex grid.
    data_dir : Path
        The directory containing the data. Contains folders `dikte_aquitard` and `top_aquitard`.
    ix : flopy.utils.GridIntersect
        The index of the model grid.
    transition_length : float
        The length of the transition zone in meters.

    Returns
    -------
    dict
        A dictionary containing the interpolated values of the aquitard layers.
    """
    layer_names = ["S11", "S12", "S13", "S21", "S22", "S31", "S32"]
    data = {}

    for name in layer_names:
        # Compute where the layer is _not_ present
        logger.info(f"Interpolating aquitard layer {name} data and its transition zone")
        fp_mask = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_mask_combined.geojson"
        gdf_mask = gpd.read_file(fp_mask)

        multipolygon = unary_union(gdf_mask.geometry)
        ids = ix.intersect(multipolygon, contains_centroid=False, min_area_fraction=0.5).cellids.astype(int)
        data[f"{name}_mask"] = np.zeros(ds_regis.sizes["icell2d"], dtype=bool)
        data[f"{name}_mask"][ids] = True

        # Compute where the layer transitions to REGIS
        multipolygon_transition = multipolygon.buffer(transition_length).difference(multipolygon)
        ids_trans = ix.intersect(
            multipolygon_transition, contains_centroid=False, min_area_fraction=0.5
        ).cellids.astype(int)
        data[f"{name}_transition"] = np.zeros(ds_regis.sizes["icell2d"], dtype=bool)
        data[f"{name}_transition"][ids_trans] = True

        # Interpolate thickness points using Kriging
        fp_pts = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)

        with redirect_stdout(logging.StreamHandler(logger)):
            ok = pykrige.ok.OrdinaryKriging(
                gdf_pts.geometry.x.values,
                gdf_pts.geometry.y.values,
                gdf_pts.value.values,
                variogram_model="linear",
                verbose=logger.level <= logging.DEBUG,
                enable_plotting=logger.level <= logging.DEBUG,
            )
        xq = ds_regis.x.values[~data[f"{name}_mask"]]
        yq = ds_regis.y.values[~data[f"{name}_mask"]]
        kriging_result = ok.execute("points", xq, yq)
        data[f"D{name}_value"] = np.where(~data[f"{name}_mask"], kriging_result[0], 0.)
        data[f"D{name}_value_unc"] = np.where(~data[f"{name}_mask"], kriging_result[1], np.nan)

        # Interpolate top aquitard points using Kriging
        fp_pts = data_dir / "top_aquitard" / f"T{name}" / f"T{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)

        with redirect_stdout(logging.StreamHandler(logger)):
            ok = pykrige.ok.OrdinaryKriging(
                gdf_pts.geometry.x.values,
                gdf_pts.geometry.y.values,
                gdf_pts.value.values,
                variogram_model="linear",
                verbose=logger.level <= logging.DEBUG,
                enable_plotting=logger.level <= logging.DEBUG,
            )
        kriging_result = ok.execute("points", xq, yq)
        data[f"T{name}_value"] = np.where(~data[f"{name}_mask"], kriging_result[0], np.nan)
        data[f"T{name}_value_unc"] = np.where(~data[f"{name}_mask"], kriging_result[1], np.nan)

    return data
