"""Module containing functions to retrieve PWN bodemlagen."""

import logging
from pathlib import Path

import geopandas as gpd
import nlmod
import numpy as np
import pykrige.ok
import xarray as xr
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def get_pwn_aquitard_data(data_dir: Path, ds_regis: xr.Dataset, ix: nlmod.Index, transition_length: float) -> dict:
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
    data_dir : Path
        The directory containing the data. Contains folders `dikte_aquitard` and `top_aquitard`.
    ds_regis : xr.Dataset
        The REGIS modellayer that contains the vertex grid.
    ix : nlmod.Index
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

        # Interpolate thickness points using Krieging
        fp_pts = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)
        ok = pykrige.ok.OrdinaryKriging(
            gdf_pts.geometry.x.values,
            gdf_pts.geometry.y.values,
            gdf_pts.value.values,
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,
        )
        xq = ds_regis.x.values[~data[f"{name}_mask"]]
        yq = ds_regis.y.values[~data[f"{name}_mask"]]
        data[f"D{name}_value"] = np.zeros(ds_regis.sizes["icell2d"])
        data[f"D{name}_value"][~data[f"{name}_mask"]] = ok.execute("points", xq, yq)[0]

        # Interpolate top aquitard points using Krieging
        fp_pts = data_dir / "top_aquitard" / f"T{name}" / f"T{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)
        ok = pykrige.ok.OrdinaryKriging(
            gdf_pts.geometry.x.values,
            gdf_pts.geometry.y.values,
            gdf_pts.value.values,
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,
        )
        data[f"T{name}_value"] = np.zeros(ds_regis.sizes["icell2d"])
        data[f"T{name}_value"][~data[f"D{name}_mask"]] = ok.execute("points", xq, yq)[0]
    return data
