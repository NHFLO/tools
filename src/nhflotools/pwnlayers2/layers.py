"""Module containing functions to retrieve PWN bodemlagen."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pykrige.ok
import xarray as xr
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridintersect import GridIntersect
from nlmod.dims.grid import modelgrid_from_ds
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def get_pwn_aquitard_data(ds_regis: xr.Dataset, ix: GridIntersect, modelgrid: VertexGrid, data_dir: Path, transition_length: float) -> dict:
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
    ix : flopy.utils.GridIntersect
        The index of the model grid.
    modelgrid : flopy.discretization.VertexGrid
        The model grid.
    data_dir : Path
        The directory containing the data. Contains folders `dikte_aquitard` and `top_aquitard`.
    transition_length : float
        The length of the transition zone in meters.

    Returns
    -------
    dict
        A dictionary containing the interpolated values of the aquitard layers.
    """
    verbose = logger.level <= logging.DEBUG

    if ix is None and modelgrid is None and ds_regis is not None:
        modelgrid = modelgrid_from_ds(ds_regis)

    if ix is None and modelgrid is not None:
        ix = GridIntersect(modelgrid, method="vertex")

    ncell = len(ix.mfgrid.cell2d)
    layer_names = ["S11", "S12", "S13", "S21", "S22", "S31", "S32"]
    data = {}

    for name in layer_names:
        # Compute where the layer is _not_ present
        logger.info("Interpolating aquitard layer %s data and its transition zone", name)
        fp_mask = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_mask_combined.geojson"
        gdf_mask = gpd.read_file(fp_mask)

        multipolygon = unary_union(gdf_mask.geometry)
        ids = ix.intersect(multipolygon, contains_centroid=False, min_area_fraction=0.5).cellids.astype(int)
        data[f"{name}_mask"] = np.zeros(ncell, dtype=bool)
        data[f"{name}_mask"][ids] = True

        # Compute where the layer transitions to REGIS
        multipolygon_transition = multipolygon.buffer(transition_length).difference(multipolygon)
        ids_trans = ix.intersect(
            multipolygon_transition, contains_centroid=False, min_area_fraction=0.5
        ).cellids.astype(int)
        data[f"{name}_transition"] = np.zeros(ncell, dtype=bool)
        data[f"{name}_transition"][ids_trans] = True

        # Interpolate thickness points using Kriging
        fp_pts = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)
        ok = pykrige.ok.OrdinaryKriging(
            gdf_pts.geometry.x.values,
            gdf_pts.geometry.y.values,
            gdf_pts.value.values,
            variogram_model="linear",
            verbose=verbose,
            enable_plotting=False,
        )
        xq = ix.mfgrid.xcellcenters[~data[f"{name}_mask"]]
        yq = ix.mfgrid.ycellcenters[~data[f"{name}_mask"]]
        kriging_result = ok.execute("points", xq, yq)
        data[f"D{name}_value"] = np.zeros(ncell)
        data[f"D{name}_value"][~data[f"{name}_mask"]] = kriging_result[0]
        data[f"D{name}_value_unc"] = np.zeros(ncell)
        data[f"D{name}_value_unc"][~data[f"{name}_mask"]] = kriging_result[1]

        # Interpolate top aquitard points using Kriging
        fp_pts = data_dir / "top_aquitard" / f"T{name}" / f"T{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)
        ok = pykrige.ok.OrdinaryKriging(
            gdf_pts.geometry.x.values,
            gdf_pts.geometry.y.values,
            gdf_pts.value.values,
            variogram_model="linear",
            verbose=verbose,
            enable_plotting=False,
        )
        kriging_result = ok.execute("points", xq, yq)
        data[f"T{name}_value"] = np.zeros(ncell)
        data[f"T{name}_value"][~data[f"{name}_mask"]] = kriging_result[0]
        data[f"T{name}_value_unc"] = np.zeros(ncell)
        data[f"T{name}_value_unc"][~data[f"{name}_mask"]] = kriging_result[1]

    return data
