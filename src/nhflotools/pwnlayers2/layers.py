"""Module containing functions to retrieve PWN bodemlagen."""

import logging
from importlib import metadata
from pathlib import Path

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import pykrige.ok
import xarray as xr
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridintersect import GridIntersect
from nlmod.dims.grid import gdf_to_bool_da, modelgrid_from_ds
from packaging import version
from shapely.ops import unary_union

from nhflotools.pwnlayers.utils import fix_missings_botms_and_min_layer_thickness

logger = logging.getLogger(__name__)

layer_names = pd.Index(
    ["W11", "S11", "W12", "S12", "W13", "S13", "W21", "S21", "W22", "S22", "W31", "S31", "W32", "S32"], name="layer"
)

if version.parse(metadata.version("nlmod")) < version.parse("0.9.1.dev0"):
    msg = "nlmod version 0.9.1.dev0 or higher is required"
    raise ImportError(msg)


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

        f, ax = nlmod.plot.get_map(ds.extent, base=1e4)
        ax.set_aspect("equal", adjustable="box")
        pc = nlmod.plot.data_array(data[f"{name}_mask"], ds=ds)
        nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
        gdf_mask.plot(ax=ax, facecolor="red", alpha=0.5, edgecolor="r")

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
        ok = pykrige.ok.OrdinaryKriging(
            gdf_pts.geometry.x.values,
            gdf_pts.geometry.y.values,
            gdf_pts.value.values,
            variogram_model="linear",
            verbose=verbose,
            enable_plotting=False,
        )
        xq = ix.mfgrid.xcellcenters[data[f"{name}_mask"]]
        yq = ix.mfgrid.ycellcenters[data[f"{name}_mask"]]
        kriging_result = ok.execute("points", xq, yq)
        data[f"D{name}_value"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=0.0)
        data[f"D{name}_value"][data[f"{name}_mask"]] = kriging_result[0]
        data[f"D{name}_value_unc"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=np.nan)
        data[f"D{name}_value_unc"][data[f"{name}_mask"]] = kriging_result[1]

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
        data[f"T{name}_value"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=np.nan)
        data[f"T{name}_value"][data[f"{name}_mask"]] = kriging_result[0]
        data[f"T{name}_value_unc"] = nlmod.util.get_da_from_da_ds(ds, dims=("icell2d",), data=np.nan)
        data[f"T{name}_value_unc"][data[f"{name}_mask"]] = kriging_result[1]

    return data


def get_mensink_layer_model(ds_pwn_data, ds_pwn_data_2024, fix_min_layer_thickness=True):
    layer_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top"],
            "botm": get_mensink_botm(ds_pwn_data, ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
            "kh": get_mensink_kh(ds_pwn_data, ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
            "kv": get_mensink_kv(ds_pwn_data, ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness),
        },
        coords={"layer": layer_names},
        attrs={
            "extent": ds_pwn_data.attrs["extent"],
            "gridtype": ds_pwn_data.attrs["gridtype"],
        },
    )
    mask = get_mensink_botm(
        ds_pwn_data,
        ds_pwn_data_2024,
        mask=True,
        transition=False,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )
    mask_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top_mask"],
            "botm": mask,
            "kh": mask,
            "kv": mask,
        },
    )
    transition = get_mensink_botm(
        ds_pwn_data,
        ds_pwn_data_2024,
        mask=False,
        transition=True,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )
    transition_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top_transition"],
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


def get_mensink_thickness(data, ds_pwn_data_2024, mask=False, transition=False, fix_min_layer_thickness=True):
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
    mask : bool, optional
        If True, returns a boolean mask indicating the valid thickness values.
        If False, returns the thickness values directly. Default is False.
    transition : bool, optional
        If True, treat data as a mask with True for transition cells. Default is False.

    Returns
    -------
    thickness: xarray.DataArray or numpy.ndarray
        If mask is True, returns a boolean mask indicating the valid thickness values.
        If mask is False, returns the thickness values as a DataArray or ndarray.

    """
    botm = get_mensink_botm(
        data,
        ds_pwn_data_2024,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )

    if "top" in data.data_vars:
        top_botm = xr.concat((data["top"].expand_dims(dim={"layer": ["mv"]}), botm), dim="layer")
    else:
        top_botm = botm

    out = -top_botm.diff(dim="layer")
    out = out.where(~np.isclose(out, 0.0), other=0.0)

    if (out < 0.0).any():
        logger.warning("Botm is not monotonically decreasing.")
    return out


def get_mensink_botm(
    a: xr.Dataset, a2024: xr.Dataset, mask: bool = False, transition: bool = False, fix_min_layer_thickness=True
):
    """
    Calculate the bottom elevation of each layer in the model.

    Parameters
    ----------
    a :

    Returns
    -------
    out (xarray.DataArray): Array containing the bottom elevation of each layer.
    """
    if mask:
        return get_mensink_botm_mask(a)
    if transition:
        return get_mensink_botm_transition(a)

    out = get_mensink_botm_values(a, a2024)

    if fix_min_layer_thickness:
        ds = xr.Dataset({"botm": out, "top": a["top"]})
        fix_missings_botms_and_min_layer_thickness(ds)
        out = ds["botm"]

    return out


def get_mensink_botm_values(a, a2024):
    out = xr.concat(
        (
            a2024["TS11_value"].fillna(a["TS11"]),  # Base aquifer 11
            a2024["TS11_value"].fillna(a["TS11"]) - a2024["DS11_value"],  # Base aquitard 11
            a2024["TS12_value"].fillna(a["TS12"]),  # Base aquifer 12
            a2024["TS12_value"].fillna(a["TS12"]) - a2024["DS12_value"],  # Base aquitard 12
            a2024["TS13_value"].fillna(a["TS13"]),  # Base aquifer 13
            a2024["TS13_value"].fillna(a["TS13"]) - a2024["DS13_value"],  # Base aquitard 13
            a2024["TS21_value"].fillna(a["TS21"]),  # Base aquifer 21
            a2024["TS21_value"].fillna(a["TS21"]) - a2024["DS21_value"],  # Base aquitard 21
            a2024["TS22_value"].fillna(a["TS22"]),  # Base aquifer 22
            a2024["TS22_value"].fillna(a["TS22"]) - a2024["DS22_value"],  # Base aquitard 22
            a2024["TS31_value"].fillna(a["TS31"]),  # Base aquifer 31
            a2024["TS31_value"].fillna(a["TS31"]) - a2024["DS31_value"],  # Base aquitard 31
            a2024["TS32_value"].fillna(a["TS32"]),  # Base aquifer 32
            a2024["TS32_value"].fillna(a["TS32"]) - 5.0,  # Base aquitard 33
            # a["TS32"] - 105., # Base aquifer 41
        ),
        dim=layer_names,
    )
    return out


def get_mensink_botm_mask(a):
    out = xr.concat(
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
    return out


def get_mensink_botm_transition(a):
    out = xr.concat(
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
    return out


def get_mensink_kh(data, data_2024, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
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
        data,
        data_2024,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    ).drop_vars("layer")
    assert not (thickness < 0.0).any(), "Negative thickness values are not allowed."

    if mask:
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)
        b = thickness.where(thickness, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)
        b = thickness.where(~thickness, np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data

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
        mask = get_mensink_kh(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    return out


def get_mensink_kv(data, data_2024, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
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
        data,
        data_2024,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    ).drop_vars("layer")

    if mask:
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)
        b = thickness.where(thickness, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)
        b = thickness.where(~thickness, np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data
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
        mask = get_mensink_kv(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    return out
