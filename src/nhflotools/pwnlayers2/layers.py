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

translate_triwaco_bergen_names_to_index = {
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
}
translate_triwaco_mensink_names_to_index = {
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
    # layer_model_nhd, mask_model_nhd, transition_model_nhd = get_mensink_layer_model2(
    #     ds_pwn_data=ds_pwn_data, ds_pwn_data_2024=ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness
    # )
    get_layer_model(ds_pwn_data)
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

    # else:
    #     thick_layer_model_regis = nlmod.dims.layers.calculate_thickness(layer_model_regis)
    #     assert ~(thick_layer_model_regis < 0.0).any(), "Regis thickness of layers should be positive"

    #     if data_path_mensink:
    #         layer_model_mensink, mask_model_mensink, transition_model_mensink = get_mensink_layer_model(
    #             ds_pwn_data=ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness
    #         )
    #         thick_layer_model_mensink = nlmod.dims.layers.calculate_thickness(layer_model_mensink)
    #         assert ~(thick_layer_model_mensink < 0.0).any(), "Mensink thickness of layers should be positive"

    #     if data_path_bergen:
    #         layer_model_bergen, mask_model_bergen, transition_model_bergen = get_bergen_layer_model(
    #             ds_pwn_data=ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness
    #         )
    #         thick_layer_model_bergen = nlmod.dims.layers.calculate_thickness(layer_model_bergen)
    #         assert ~(thick_layer_model_bergen < 0.0).any(), "Bergen thickness of layers should be positive"

    #     if data_path_mensink and data_path_bergen:
    #         # Combine PWN layer model with REGIS layer model
            # layer_model_mensink_regis, _ = combine_two_layer_models(
            #     layer_model_regis=layer_model_regis,
            #     layer_model_other=layer_model_mensink,
            #     mask_model_other=mask_model_mensink,
            #     transition_model=transition_model_mensink,
            #     top=top,
            #     df_koppeltabel=df_koppeltabel,
            #     koppeltabel_header_regis="Regis II v2.2",
            #     koppeltabel_header_other="ASSUMPTION1",
            #     remove_nan_layers=False,
            # )
    #         if fix_min_layer_thickness:
    #             fix_missings_botms_and_min_layer_thickness(layer_model_mensink_regis)

    #         # Combine PWN layer model with Bergen layer model and REGIS layer model
    #         (
    #             layer_model_mensink_bergen_regis,
    #             _,
    #         ) = combine_two_layer_models(
    #             layer_model_regis=layer_model_mensink_regis,
    #             layer_model_other=layer_model_bergen,
    #             mask_model_other=mask_model_bergen,
    #             transition_model=transition_model_bergen,
    #             top=top,
    #             df_koppeltabel=df_koppeltabel,
    #             koppeltabel_header_regis="Regis II v2.2",
    #             koppeltabel_header_other="ASSUMPTION1",
    #         )

    #     elif data_path_mensink:
    #         layer_model_mensink_bergen_regis, _ = combine_two_layer_models(
    #             layer_model_regis=layer_model_regis,
    #             layer_model_other=layer_model_mensink,
    #             mask_model_other=mask_model_mensink,
    #             transition_model=transition_model_mensink,
    #             top=top,
    #             df_koppeltabel=df_koppeltabel,
    #             koppeltabel_header_regis="Regis II v2.2",
    #             koppeltabel_header_other="ASSUMPTION1",
    #         )
    #     elif data_path_bergen:
    #         layer_model_mensink_bergen_regis, _ = combine_two_layer_models(
    #             layer_model_regis=layer_model_regis,
    #             layer_model_other=layer_model_bergen,
    #             mask_model_other=mask_model_bergen,
    #             transition_model=transition_model_bergen,
    #             top=top,
    #             df_koppeltabel=df_koppeltabel,
    #             koppeltabel_header_regis="Regis II v2.2",
    #             koppeltabel_header_other="ASSUMPTION1",
    #         )

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

        # Interpolate thickness points using Kriging
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

        # Interpolate top aquitard points using Kriging
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


def get_mensink_thickness(ds_pwn_data_2024, fix_min_layer_thickness=True):
    """
    Calculate the thickness of layers in a given dataset.

    If mask is True, the function returns a boolean mask indicating the valid
    thickness values, requiering all dependent values to be valid.
    If transisition is True, the function returns a boolean mask indicating the
    cells for which any of the dependent values is marked as a transition.

    The masks are computated with nan's for False, so that if any of the dependent
    values is nan, the mask_float will be nan and mask will be False.
    The transitions are computed with nan's for True, so that if any of the dependent
    values is nan, the transition_float will be nan and transition will be True.

    If the dataset contains a variable 'top', the thickness is calculated
    from the difference between the top and bottom of each layer. If the
    dataset does not contain a variable 'top', the thickness is calculated
    from the difference between the bottoms.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input dataset containing the layer data.

    Returns
    -------
    thickness: xarray.DataArray or numpy.ndarray
        If mask is True, returns a boolean mask indicating the valid thickness values.
        If mask is False, returns the thickness values as a DataArray or ndarray.

    """
    botm = get_botm_values(
        ds_pwn_data_2024,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )

    if "top" in ds_pwn_data_2024.data_vars:
        top_botm = xr.concat((ds_pwn_data_2024["top"].expand_dims(dim={"layer": ["mv"]}), botm), dim="layer")
    else:
        top_botm = botm

    out = -top_botm.diff(dim="layer")
    out = out.where(~np.isclose(out, 0.0), other=0.0)

    if (out < 0.0).any():
        logger.warning("Botm is not monotonically decreasing.")
    return out


def get_botm_values(a2024, fix_min_layer_thickness=True):
    out = xr.concat(
        (
            a2024["TS11_value"],  # Base aquifer 11
            a2024["TS11_value"] - a2024["DS11_value"],  # Base aquitard 11
            a2024["TS12_value"],  # Base aquifer 12
            a2024["TS12_value"] - a2024["DS12_value"],  # Base aquitard 12
            a2024["TS13_value"],  # Base aquifer 13
            a2024["TS13_value"] - a2024["DS13_value"],  # Base aquitard 13
            a2024["TS21_value"],  # Base aquifer 21
            a2024["TS21_value"] - a2024["DS21_value"],  # Base aquitard 21
            a2024["TS22_value"],  # Base aquifer 22
            a2024["TS22_value"] - a2024["DS22_value"],  # Base aquitard 22
            a2024["TS31_value"],  # Base aquifer 31
            a2024["TS31_value"] - a2024["DS31_value"],  # Base aquitard 31
            a2024["TS32_value"],  # Base aquifer 32
            a2024["TS32_value"] - 5.0,  # Base aquitard 33
            # a["TS32"] - 105., # Base aquifer 41
        ),
        dim=layer_names,
    )
    if fix_min_layer_thickness:
        out = fix_missings_botms_and_min_layer_thickness(top=a2024["top"], botm=out)

    return out


def get_botm_mask(a):
    return xr.concat(
        (
            a["TS11_mask"],
            a["TS11_mask"],
            a["TS12_mask"],
            a["TS12_mask"],
            a["TS13_mask"],
            a["TS13_mask"],
            a["TS21_mask"],
            a["TS21_mask"],
            a["TS22_mask"],
            a["TS22_mask"],
            a["TS31_mask"],
            a["TS31_mask"],
            a["TS32_mask"],
            a["TS32_mask"],
            # a["TS32"] - 105., # Base aquifer 41
        ),
        dim=layer_names,
    )


def get_botm_transition(a):
    return xr.concat(
        (
            a["TS11_transition"],
            a["TS11_transition"],
            a["TS12_transition"],
            a["TS12_transition"],
            a["TS13_transition"],
            a["TS13_transition"],
            a["TS21_transition"],
            a["TS21_transition"],
            a["TS22_transition"],
            a["TS22_transition"],
            a["TS31_transition"],
            a["TS31_transition"],
            a["TS32_transition"],
            a["TS32_transition"],
            # a["TS32"] - 105., # Base aquifer 41
        ),
        dim=layer_names,
    )


def get_kh_values(data_2024, anisotropy=5.0, *, fix_min_layer_thickness=True):
    """
    Calculate the hydraulic conductivity (kh) based on the given data.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing the necessary variables.
    anisotropy : float, optional
        Anisotropy factor to be applied to the aquitard layers. Default is 5.0.

    Returns
    -------
    kh: xarray.DataArray
        The calculated hydraulic conductivity.

    """
    kh_mensink = get_mensink_kh(data_2024, mask=False, anisotropy=anisotropy, transition=False, fix_min_layer_thickness=fix_min_layer_thickness)
    kh_bergen = get_bergen_kh(data_2024, mask=False, anisotropy=anisotropy, transition=False)

    return xr.where(kh_mensink.notnull(), kh_mensink, kh_bergen)


def get_kv_values(data_2024, anisotropy=5.0, *, fix_min_layer_thickness=True):
    """
    Calculate the hydraulic conductivity (kv) based on the given data.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing the necessary variables.
    anisotropy : float, optional
        Anisotropy factor to be applied to the aquitard layers. Default is 5.0.

    Returns
    -------
    kv: xarray.DataArray
        The calculated hydraulic conductivity.

    """
    kv_mensink = get_mensink_kv(data_2024, mask=False, anisotropy=anisotropy, transition=False, fix_min_layer_thickness=fix_min_layer_thickness)
    kv_bergen = get_bergen_kv(data_2024, mask=False, anisotropy=anisotropy, transition=False)

    return xr.where(kv_mensink.notnull(), kv_mensink, kv_bergen)


def get_mensink_kh(data_2024, *, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
    """
    Calculate the hydraulic conductivity (kh) based on the given data.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing the necessary variables.
    mask : bool, optional
        Flag indicating whether to apply a mask to the data. Default is False.
    anisotropy : float, optional
        Anisotropy factor to be applied to the aquitard layers. Default is 5.0.
    transition : bool, optional
        Flag indicating whether to treat data as a mask with True for transition cells.

    Returns
    -------
    kh: xarray.DataArray
        The calculated hydraulic conductivity.

    """
    thickness = get_mensink_thickness(
        data_2024,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    ).drop_vars("layer")
    thickness = get_mensink_thickness(data_2024, fix_min_layer_thickness=fix_min_layer_thickness)
    
    assert not (thickness < 0.0).any(), "Negative thickness values are not allowed."

    if mask:
        _a = data_2024[[var for var in data_2024.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)
        b = thickness.where(thickness, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data_2024[[var for var in data_2024.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)
        b = thickness.where(~thickness, np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data_2024

        if fix_min_layer_thickness:
            # Should not matter too much because mask == False
            b = thickness.where(thickness != 0.0, other=0.005)

        def n(s):
            return s

    out = xr.concat(
        (
            a[n("KW11")],  # Aquifer 11
            b.isel(layer=1) / a[n("C11AREA")] * anisotropy,  # Aquitard 11
            a[n("KW12")],  # Aquifer 12
            b.isel(layer=3) / a[n("C12AREA")] * anisotropy,  # Aquitard 12
            a[n("KW13")],  # Aquifer 13
            b.isel(layer=5) / a[n("C13AREA")] * anisotropy,  # Aquitard 13
            a[n("KW21")],  # Aquifer 21
            b.isel(layer=7) / a[n("C21AREA")] * anisotropy,  # Aquitard 21
            a[n("KW22")],  # Aquifer 22
            b.isel(layer=9) / a[n("C22AREA")] * anisotropy,  # Aquitard 22
            a[n("KW31")],  # Aquifer 31
            b.isel(layer=11) / a[n("C31AREA")] * anisotropy,  # Aquitard 31
            a[n("KW32")],  # Aquifer 32
            b.isel(layer=13) / a[n("C32AREA")] * anisotropy,  # Aquitard 32
        ),
        dim=layer_names,
    )

    s12k = (
        a[n("s12kd")] * (a[n("ms12kd")] == 1)
        + 0.5 * a[n("s12kd")] * (a[n("ms12kd")] == 2)
        + 3 * a[n("s12kd")] * (a[n("ms12kd")] == 3)
    ) / b.isel(layer=3)
    s13k = a[n("s13kd")] * (a[n("ms13kd")] == 1) + 1.12 * a[n("s13kd")] * (a[n("ms13kd")] == 2) / b.isel(layer=5)
    s21k = a[n("s21kd")] * (a[n("ms21kd")] == 1) + a[n("s21kd")] * (a[n("ms21kd")] == 2) / b.isel(layer=7)
    s22k = 2 * a[n("s22kd")] * (a[n("ms22kd")] == 1) + a[n("s22kd")] * (a[n("ms22kd")] == 1) / b.isel(layer=9)

    out.loc[{"layer": "S12"}] = out.loc[{"layer": "S12"}].where(np.isnan(s12k), other=s12k)
    out.loc[{"layer": "S13"}] = out.loc[{"layer": "S13"}].where(np.isnan(s13k), other=s13k)
    out.loc[{"layer": "S21"}] = out.loc[{"layer": "S21"}].where(np.isnan(s21k), other=s21k)
    out.loc[{"layer": "S22"}] = out.loc[{"layer": "S22"}].where(np.isnan(s22k), other=s22k)

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_mensink_kh(data_2024, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    return out


def get_mensink_kv(data_2024, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
    """
    Calculate the hydraulic conductivity (KV) for different aquifers and aquitards.

    Parameters
    ----------
        data (xarray.Dataset): Dataset containing the necessary variables for calculation.
        mask (bool, optional): Flag indicating whether to apply a mask to the data. Defaults to False.
        anisotropy (float, optional): Anisotropy factor for adjusting the hydraulic conductivity. Defaults to 5.0.

    Returns
    -------
        xarray.DataArray: Array containing the calculated hydraulic conductivity values for each layer.

    Notes
    -----
        - The function expects the input dataset to contain the following variables:
            - KW11, KW12, KW13, KW21, KW22, KW31, KW32: Hydraulic conductivity values for aquifers.
            - C11AREA, C12AREA, C13AREA, C21AREA, C22AREA, C31AREA, C32AREA: Areas of aquitards corresponding to each aquifer.
        - The function also requires the `get_mensink_thickness` function to be defined and accessible.

    Example:
        # Calculate hydraulic conductivity values without applying a mask
        kv_values = get_mensink_kv(data)

        # Calculate hydraulic conductivity values with a mask applied
        kv_values_masked = get_mensink_kv(data, mask=True)
    """
    thickness = get_mensink_thickness(
        data_2024,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    ).drop_vars("layer")

    if mask:
        _a = data_2024[[var for var in data_2024.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)
        b = thickness.where(thickness, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data_2024[[var for var in data_2024.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)
        b = thickness.where(~thickness, np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data_2024
        b = thickness

        if fix_min_layer_thickness:
            # Should not matter too much because mask == False
            b = thickness.where(thickness != 0.0, other=0.005)

        def n(s):
            return s

    out = xr.concat(
        (
            a[n("KW11")] / anisotropy,  # Aquifer 11
            b.isel(layer=1) / a[n("C11AREA")],  # Aquitard 11
            a[n("KW12")] / anisotropy,  # Aquifer 12
            b.isel(layer=3) / a[n("C12AREA")],  # Aquitard 12
            a[n("KW13")] / anisotropy,  # Aquifer 13
            b.isel(layer=5) / a[n("C13AREA")],  # Aquitard 13
            a[n("KW21")] / anisotropy,  # Aquifer 21
            b.isel(layer=7) / a[n("C21AREA")],  # Aquitard 21
            a[n("KW22")] / anisotropy,  # Aquifer 22
            b.isel(layer=9) / a[n("C22AREA")],  # Aquitard 22
            a[n("KW31")] / anisotropy,  # Aquifer 31
            b.isel(layer=11) / a[n("C31AREA")],  # Aquitard 31
            a[n("KW32")] / anisotropy,  # Aquifer 32
            b.isel(layer=13) / a[n("C32AREA")],  # Aquitard 32
        ),
        dim=layer_names,
    )

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_mensink_kv(data_2024, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    return out
