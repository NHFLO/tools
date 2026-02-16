"""Module containing functions to retrieve PWN bodemlagen."""

import logging
from importlib import metadata
from pathlib import Path

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import xarray as xr
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridintersect import GridIntersect
from nlmod import cache
from nlmod.dims.grid import gdf_to_bool_da, modelgrid_from_ds
from packaging import version
from scipy.interpolate import griddata
from shapely.ops import unary_union

from nhflotools.pwnlayers.io import read_pwn_data2
from nhflotools.pwnlayers.layers import get_bergen_kh, get_bergen_kv
from nhflotools.pwnlayers.merge_layer_models import combine_two_layer_models
from nhflotools.pwnlayers.utils import fix_missings_botms_and_min_layer_thickness

logger = logging.getLogger(__name__)

layer_names = pd.Index(
    ["W11", "S11", "W12", "S12", "W13", "S13", "W21", "S21", "W22", "S22", "W31", "S31", "W32", "S32"], name="layer"
)

if version.parse(metadata.version("nlmod")) < version.parse("0.9.1.dev0"):
    msg = "nlmod version 0.9.1.dev0 or higher is required"
    raise ImportError(msg)


logger = logging.getLogger(__name__)

translate_pwn_names_to_index = {
    "W11": 0,
    "S11": 1,
    "W12": 2,
    "S12": 3,
    "W13": 4,
    "S13": 5,
    "W21": 6,
    "S21": 7,
    "W22": 8,
    "S22": 9,
    "W31": 10,
    "S31": 11,
    "W32": 12,
    "S32": 13,
}

@cache.cache_netcdf(coords_3d=True, attrs_ds=True, datavars=["kh", "kv", "botm", "top"], attrs=[])
def get_pwn_layer_model(
    ds_regis=None,
    data_path_mensink=None,
    data_path_bergen=None,
    data_path_2024=None,
    fname_koppeltabel=None,
    top=None,
    length_transition=100.0,
    fix_min_layer_thickness=True,
):
    """
    Merge PWN layer model with ds_regis.

    The PWN layer model is merged with the REGISII layer model. The values of the PWN layer model are used where the layer_model_pwn is not nan.

    The values of the REGISII layer model are used where the layer_model_pwn is nan and transition_model_pwn is False. The remaining values are where the transition_model_pwn is True. Those values are linearly interpolated from the REGISII layer model to the PWN layer model.

    The following order should be maintained in you modelscript:
    - Get REGIS ds using nlmod.read.regis.get_combined_layer_models() and nlmod.to_model_ds()
    - Refine grid with surface water polygons and areas of interest with nlmod.grid.refine()
    - Get AHN with nlmod.read.ahn.get_ahn4() and resample to model grid with nlmod.dims.resample.structured_da_to_ds()
    - Get PWN layer model with nlmod.read.pwn.get_pwn_layer_model()

    Parameters
    ----------
    ds_regis : xarray Dataset
        The dataset to merge the PWN layer model with. This dataset should contain the variables 'kh', 'kv', 'botm'. And 'top' if 'replace_top_with_ahn_key' is None.
        Produced by nlmod.read.regis.get_combined_layer_models(), nlmod.to_model_ds(), nlmod.grid.refine().
    data_path_mensink : str
        The path to the Mensink data directory.
    data_path_bergen : str
        The path to the Bergen data directory.
    data_path_2024 : str
        The path to the 2024 data directory. In 2024, work is done to improve the position of the aquitards.
    fname_koppeltabel : str
        The filename of the koppeltabel (translation table) CSV file.
    top : xarray DataArray, optional
        The top of the model grid. The default is None in which case the top of REGIS is used.
    length_transition : float, optional
        The length of the transition zone between layer_model_regis and layer_model_other in meters. The default is 100.
    fix_min_layer_thickness : bool, optional
        Fix the minimum layer thickness. The default is True.

    Returns
    -------
    ds : xarray Dataset
        The merged dataset.

    TODO: Reverse order coupling Mensink-REGIS and Bergen-REGIS
    """
    cachedir = None  # Cache not needed for underlying functions

    if (ds_regis.layer != nlmod.read.regis.get_layer_names()).any():
        msg = "All REGIS layers should be present in `ds_regis`. Use `get_regis(.., remove_nan_layers=False)`."
        raise ValueError(msg)

    layer_model_regis = ds_regis[["botm", "kh", "kv", "xv", "yv", "icvert"]]
    layer_model_regis = layer_model_regis.sel(layer=layer_model_regis.layer != "mv")
    layer_model_regis.attrs = {
        "extent": ds_regis.attrs["extent"],
        "gridtype": ds_regis.attrs["gridtype"],
    }

    # Use AHN as top. Top of layer_model_regis is used in layer_model_mensink and bergen.
    logger.info("Using top from input")
    if top.isnull().any():
        msg = "Variable top should not contain nan values"
        raise ValueError(msg)
    layer_model_regis["top"] = top

    if fix_min_layer_thickness:
        layer_model_regis["botm"] = fix_missings_botms_and_min_layer_thickness(top=top, botm=layer_model_regis["botm"])

    # Read the koppeltabel CSV file
    df_koppeltabel = pd.read_csv(fname_koppeltabel, skiprows=0, index_col=0)

    # Get PWN layer models
    ds_pwn_data = read_pwn_data2(
        layer_model_regis,
        datadir_mensink=data_path_mensink,
        datadir_bergen=data_path_bergen,
        length_transition=length_transition,
        cachedir_sub=cachedir,
        cachedir=cachedir,
        cachename="read_pwn_data2",
    )

    ds_pwn_data.update(get_pwn_aquitard_data(
        ds=ds_regis, data_dir=data_path_2024, ix=None, transition_length=length_transition
    ))
    ds_pwn_data["top"] = top
    layer_model_nhd, mask_model_nhd, transition_model_nhd = get_layer_model(ds_pwn_data)
    thick_layer_model_nhd = nlmod.dims.layers.calculate_thickness(layer_model_nhd)
    assert ~(thick_layer_model_nhd < 0.0).any(), "NHD thickness of layers should be positive"

    # Combine PWN layer model with REGIS layer model
    layer_model_mensink_bergen_regis, cat = combine_two_layer_models(
        layer_model_regis=layer_model_regis,
        layer_model_other=layer_model_nhd,
        mask_model_other=mask_model_nhd,
        transition_model=transition_model_nhd,
        top=top,
        df_koppeltabel=df_koppeltabel,
        koppeltabel_header_regis="Regis II v2.2",
        koppeltabel_header_other="ASSUMPTION1",
    )
    if fix_min_layer_thickness:
        fix_missings_botms_and_min_layer_thickness(layer_model_mensink_bergen_regis)

    m = np.isclose(layer_model_mensink_bergen_regis.kh, 0.0)
    if np.any(m):
        msg = f"Setting {m.sum().item()} values of kh that are exactly zero to 1e-6m/d."
        logger.warning(msg)
        layer_model_mensink_bergen_regis["kh"] = layer_model_mensink_bergen_regis.kh.where(~m, 1e-6)

    m = np.isclose(layer_model_mensink_bergen_regis.kv, 0.0)
    if np.any(m):
        msg = f"Setting {m.sum().item()} values of kv that are exactly zero to 1e-6m/d."
        logger.warning(msg)
        layer_model_mensink_bergen_regis["kv"] = layer_model_mensink_bergen_regis.kv.where(~m, 1e-6)

    layer_model_active = nlmod.layers.fill_nan_top_botm_kh_kv(
        layer_model_mensink_bergen_regis,
        anisotropy=5.0,
        fill_value_kh=5.0,
        fill_value_kv=1.0,
        remove_nan_layers=True,
    )

    # Get idomain based on layer thickness
    idomain = nlmod.dims.layers.get_idomain(layer_model_active)

    return xr.Dataset(
        data_vars={
            "kh": layer_model_active["kh"],
            "kv": layer_model_active["kv"],
            "botm": layer_model_active["botm"],
            "top": layer_model_active["top"],
            "area": ds_regis.get("area", nlmod.dims.get_area(ds_regis)),
            "xv": ds_regis["xv"],
            "yv": ds_regis["yv"],
            "icvert": ds_regis["icvert"],
            "idomain": idomain,
        },
        coords={
            "x": layer_model_active.coords["x"],
            "y": layer_model_active.coords["y"],
            "layer": layer_model_active.coords["layer"],
        },
        attrs=ds_regis.attrs,
    )

def get_pwn_aquitard_data(
    ds: xr.Dataset,
    data_dir: Path,
    ix: GridIntersect = None,
    modelgrid: VertexGrid = None,
    transition_length: float = 1000.0,
) -> dict:
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
    ds : xr.Dataset
        The model dataset that contains the vertex grid information.
    data_dir : Path, optional
        The directory containing the data. Contains folders `dikte_aquitard` and `top_aquitard`. Default is Path("/default/path/to/data").
    ix : flopy.utils.GridIntersect, optional
        The index of the model grid. Default is None.
    modelgrid : flopy.discretization.VertexGrid, optional
        The model grid. Default is None.
    transition_length : float, optional
        The length of the transition zone in meters. Default is 1000.0.

    Returns
    -------
    dict
        A dictionary containing the interpolated values of the aquitard layers.
    """
    verbose = logger.level <= logging.DEBUG

    data_dir = Path(data_dir)

    if ix is None and modelgrid is None and ds is not None:
        modelgrid = modelgrid_from_ds(ds, rotated=False)

    if ix is None and modelgrid is not None:
        ix = GridIntersect(modelgrid, method="vertex")

    layer_names = ["S11", "S12", "S13", "S21", "S22", "S31", "S32"]
    data = {}

    for name in layer_names:
        # Compute where the layer __is__ present
        logger.info("Interpolating aquitard layer %s data and its transition zone", name)
        fp_mask = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_mask_combined.geojson"
        gdf_mask = gpd.read_file(fp_mask)
        data[f"{name}_mask"] = gdf_to_bool_da(
            gdf=gdf_mask,
            ds=ds,
            ix=ix,
            contains_centroid=False,
            min_area_fraction=0.5,
        )

        # Compute where the layer transitions to REGIS
        multipolygon = unary_union(gdf_mask.geometry)
        multipolygon_transition = multipolygon.buffer(transition_length).difference(multipolygon)
        data[f"{name}_transition"] = gdf_to_bool_da(
            gdf=multipolygon_transition,
            ds=ds,
            ix=ix,
            contains_centroid=False,
            min_area_fraction=0.5,
        )

        # Interpolate thickness
        fp_pts = data_dir / "dikte_aquitard" / f"D{name}" / f"D{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)

        points_interp_in = np.column_stack((gdf_pts.geometry.x.values, gdf_pts.geometry.y.values))
        points_interp_out = np.column_stack((ds.x[data[f"{name}_mask"]], ds.y[data[f"{name}_mask"]]))
        values = griddata(points_interp_in, gdf_pts.value.values, points_interp_out, method="linear")
        isextrap = np.isnan(values)
        points_extrap_in = np.concatenate((points_interp_in, points_interp_out[~isextrap]))
        points_extrap_in_values = np.concatenate((gdf_pts.value.values, values[~isextrap]))
        points_extrap_out = points_interp_out[isextrap]
        values[isextrap] = griddata(points_extrap_in, points_extrap_in_values, points_extrap_out, method="nearest")
        data[f"D{name}_value"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=np.nan)
        data[f"D{name}_value"][data[f"{name}_mask"]] = values

        # Interpolate top aquitard points
        fp_pts = data_dir / "top_aquitard" / f"T{name}" / f"T{name}_interpolation_points.geojson"
        gdf_pts = gpd.read_file(fp_pts)

        points_interp_in = np.column_stack((gdf_pts.geometry.x.values, gdf_pts.geometry.y.values))
        points_interp_out = np.column_stack((ds.x[data[f"{name}_mask"]], ds.y[data[f"{name}_mask"]]))
        values = griddata(points_interp_in, gdf_pts.value.values, points_interp_out, method="linear")
        isextrap = np.isnan(values)
        points_extrap_in = np.concatenate((points_interp_in, points_interp_out[~isextrap]))
        points_extrap_in_values = np.concatenate((gdf_pts.value.values, values[~isextrap]))
        points_extrap_out = points_interp_out[isextrap]
        values[isextrap] = griddata(points_extrap_in, points_extrap_in_values, points_extrap_out, method="nearest")
        data[f"T{name}_value"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=np.nan)
        data[f"T{name}_value"][data[f"{name}_mask"]] = values

    return data


def get_layer_model(ds_pwn_data_2024, fix_min_layer_thickness=True):
    layer_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data_2024["top"],
            "botm": get_botm_values(ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
            "kh": get_kh_values(ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
            "kv": get_kv_values(ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
        },
        coords={"layer": layer_names},
        attrs={
            "extent": ds_pwn_data_2024.attrs["extent"],
            "gridtype": ds_pwn_data_2024.attrs["gridtype"],
        },
    )
    mask = get_botm_mask(
        ds_pwn_data_2024,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )
    mask_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data_2024["top_mask"],
            "botm": mask,
            "kh": mask,
            "kv": mask,
        },
    )
    transition = get_botm_transition(
        ds_pwn_data_2024,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )
    transition_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data_2024["top_transition"],
            "botm": transition,
            "kh": transition,
            "kv": transition,
        },
    )

    for var in ["kh", "kv", "botm"]:
        layer_model_mensink[var] = layer_model_mensink[var].where(mask_model_mensink[var], np.nan)
        assert (layer_model_mensink[var].notnull() == mask_model_mensink[var]).all(), (
            f"There were nan values present in {var} in cells that should be valid"
        )
        assert ((mask_model_mensink[var] + transition_model_mensink[var]) <= 1).all(), (
            f"There should be no overlap between mask and transition of {var}"
        )

    return (
        layer_model_mensink,
        mask_model_mensink,
        transition_model_mensink,
    )
