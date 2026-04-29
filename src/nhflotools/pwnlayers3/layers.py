"""Module containing functions to retrieve PWN bodemlagen.

Open concerns from Edinsi Groundwater report
---------------------------------------------
Source: rapportage_lagenmodel_pwn_concept.pdf, Vincent Post (Edinsi Groundwater), Aug 2024.
Located at: bodemlagen_pwn_2024/v2.0.0/report/

The following concerns from the Edinsi report affect this module's processing.
Detailed per-file TODOs are placed in the data preparation scripts (botm.py,
conductances.py, boundaries/_convert_boundaries.py). The items below are the
concerns most relevant to *this* module's interpretation of the data:

- [Edinsi 3.1, p.12] Depth data extends into the North Sea. Whether layers
  truly exist under the seabed is unverified. The nearest-neighbor fallback
  used in get_botm() may assign inappropriate values there.
- [Edinsi 3.2, p.24] C12AREA has extreme resistance (37800 d) west of
  Castricum. This flows through to get_kv() without validation.
- [Edinsi 4.1, p.32] S2.1 boundary may be too small (Eem clay has larger
  extent per REGIS). Affects get_mask() and get_botm() for S21/W21.
- [Edinsi 4.2-4.4, p.34-37] Several layer boundaries (S1.1, S1.2, S1.3)
  have known gaps or inconsistencies at the NHDZ/Bergen boundary. These
  affect get_botm() interpolation quality in that transition zone.
- [Edinsi 6, p.40] S3.2, S3.1, S2.2 only have NHDZ boundaries (no Bergen
  data). Edinsi recommends extending these northward.
"""

import logging
from importlib import metadata

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import xarray as xr
from nlmod import cache
from nlmod.dims.grid import gdf_to_bool_da
from packaging import version
from scipy.interpolate import griddata

from nhflotools.pwnlayers.merge_layer_models import combine_two_layer_models

logger = logging.getLogger(__name__)

layer_names = pd.Index(
    ["W11", "S11", "W12", "S12", "W13", "S13", "W21", "S21", "W22", "S22", "W31", "S31", "W32", "S32"], name="layer"
)
if version.parse(metadata.version("nlmod")) < version.parse("0.9.1.dev0"):
    msg = "nlmod version 0.9.1.dev0 or higher is required"
    raise ImportError(msg)


@cache.cache_netcdf(
    coords_3d=True,
    attrs_ds=True,
    datavars=["kh", "kv", "botm", "top"],
    attrs=[
        "extent",
        "gridtype",
        "model_name",
        "mfversion",
        "exe_name",
        "model_ws",
        "figdir",
        "cachedir",
        "transport",
    ],
)
def get_pwn_layer_model(
    ds_regis=None,
    data_path_2024=None,
    fname_koppeltabel=None,
    top=None,
    anisotropy=10.0,
    distance_transition=250.0,
    fix_min_layer_thickness=True,
    fill_value_kh=5.0,
    fill_value_kv=1.0,
    split_method="nearest_ratio",
    return_diagnostics=False,
):
    """Merge PWN layer model with the REGISII layer model.

    The PWN layer model values are used where available (not NaN). REGISII
    values are used where PWN data is absent and outside the transition zone.
    In the transition zone, values are linearly interpolated between the two
    models.

    The following order should be maintained in the modelscript:

    1. Get REGIS ds using ``nlmod.read.regis.get_combined_layer_models()``
       and ``nlmod.to_model_ds()``.
    2. Refine grid with ``nlmod.grid.refine()``.
    3. Get AHN top elevation and resample to model grid.
    4. Call this function to merge the PWN layer model.

    Parameters
    ----------
    ds_regis : xr.Dataset
        REGISII dataset containing 'kh', 'kv', 'botm', 'xv', 'yv', 'icvert'.
        Must include all REGIS layers (use ``remove_nan_layers=False``).
    data_path_2024 : pathlib.Path
        Path to the 2024 PWN data directory containing conductances,
        boundaries, and bottom elevation data.
    fname_koppeltabel : str or pathlib.Path
        Path to the koppeltabel (translation table) CSV file mapping REGIS
        layers to PWN layers.
    top : xr.DataArray
        Top elevation of the model grid (e.g., from AHN). Must not contain
        NaN values.
    anisotropy : float, optional
        Ratio of horizontal to vertical hydraulic conductivity (kh / kv).
        Used for deriving kh from vertical resistance and kv from kh.
        Default is 10.0.
    distance_transition : float, optional
        Width of the transition zone in meters between the PWN and REGIS
        layer models. Default is 250.0.
    fix_min_layer_thickness : bool, optional
        Whether to fix missing bottom elevations and enforce a minimum layer
        thickness. Default is True.
    fill_value_kh : float, optional
        Fill value for kh in cells not covered by either model (m/day).
        Default is 5.0.
    fill_value_kv : float, optional
        Fill value for kv in cells not covered by either model (m/day).
        Default is 1.0.
    split_method : {'equal', 'nearest_ratio'}, optional
        How to distribute thickness among sublayers created by splitting.
        If 'equal' (default), sublayers receive equal thickness.  If
        'nearest_ratio', thickness ratios are taken from the model that has
        actual sublayer data, extrapolated via nearest-neighbor to cells
        where that model is absent.
    return_diagnostics : bool, optional
        If True, include diagnostic variables in the returned Dataset:
        ``cat_botm``, ``cat_kh``, ``cat_kv`` (int, dims layer x icell2d,
        values 1=REGIS 2=PWN 3=Transition), ``botm_pwn`` (float, dims
        layer_pwn x icell2d), ``botm_method``, ``kh_method``,
        ``kv_method`` (int, dims layer_pwn x icell2d). Default is False.

    Returns
    -------
    xr.Dataset
        Merged dataset with variables 'kh', 'kv', 'botm', 'top', 'area',
        'xv', 'yv', 'icvert', and 'idomain'. When ``return_diagnostics``
        is True, also contains ``cat_botm``, ``cat_kh``, ``cat_kv``,
        ``botm_pwn``, ``botm_method``, ``kh_method``, and ``kv_method``.
    """
    if (ds_regis.layer != nlmod.read.regis.get_layer_names()).any():
        msg = "All REGIS layers should be present in `ds_regis`. Use `get_regis(.., remove_nan_layers=False)`."
        raise ValueError(msg)

    layer_model_regis = ds_regis[["botm", "kh", "kv", "xv", "yv", "icvert"]]
    layer_model_regis = layer_model_regis.sel(layer=layer_model_regis.layer != "mv")
    layer_model_regis.attrs = {
        "extent": ds_regis.attrs["extent"],
        "gridtype": ds_regis.attrs["gridtype"],
    }

    # Use AHN as top
    logger.info("Using top from input")
    if top.isnull().any():
        msg = "Variable top should not contain nan values"
        raise ValueError(msg)
    layer_model_regis["top"] = top

    # Read the koppeltabel CSV file
    df_koppeltabel = pd.read_csv(fname_koppeltabel, skiprows=0, index_col=0)

    # Get PWN layer models
    get_ds_result = get_ds(
        ds_regis=ds_regis,
        data_path_2024=data_path_2024,
        top=top,
        anisotropy=anisotropy,
        distance_transition=distance_transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
        return_methods=return_diagnostics,
    )
    if return_diagnostics:
        ds_pwn, mask_pwn, transition_pwn, method_ds = get_ds_result
    else:
        ds_pwn, mask_pwn, transition_pwn = get_ds_result
    ds_pwn_thickness = nlmod.dims.layers.calculate_thickness(ds_pwn)
    if (ds_pwn_thickness < 0.0).any():
        msg = "PWN layer thickness should be positive"
        raise ValueError(msg)

    # Combine PWN layer model with REGIS layer model
    layer_model_pwn_regis, cat = combine_two_layer_models(
        layer_model_regis=layer_model_regis,
        layer_model_other=ds_pwn[["botm", "kh", "kv"]],
        mask_model_other=mask_pwn[["botm", "kh", "kv"]],
        transition_model=transition_pwn[["botm", "kh", "kv"]],
        top=top,
        df_koppeltabel=df_koppeltabel,
        koppeltabel_header_regis="Regis II v2.2",
        koppeltabel_header_other="ASSUMPTION1",
        split_method=split_method,
    )
    if fix_min_layer_thickness:
        layer_model_pwn_regis["botm"] = fix_missings_botms_and_min_layer_thickness(
            top=top, botm=layer_model_pwn_regis["botm"]
        )
    merged_thickness = nlmod.dims.layers.calculate_thickness(layer_model_pwn_regis)
    if (merged_thickness < 0.0).any():
        msg = "Merged layer thickness should be non-negative"
        raise ValueError(msg)

    m = np.isclose(layer_model_pwn_regis.kh, 0.0) | ~np.isfinite(layer_model_pwn_regis.kh)
    if np.any(m):
        logger.warning("Setting %d problematic kh values (zero/inf/nan) to NaN m/d.", m.sum().item())
        layer_model_pwn_regis["kh"] = layer_model_pwn_regis.kh.where(~m, np.nan)

    m = np.isclose(layer_model_pwn_regis.kv, 0.0) | ~np.isfinite(layer_model_pwn_regis.kv)
    if np.any(m):
        logger.warning("Setting %d problematic kv values (zero/inf/nan) to NaN m/d.", m.sum().item())
        layer_model_pwn_regis["kv"] = layer_model_pwn_regis.kv.where(~m, np.nan)

    layer_model_active = nlmod.layers.fill_nan_top_botm_kh_kv(
        layer_model_pwn_regis,
        anisotropy=anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
        remove_nan_layers=True,
    )

    # Get idomain based on layer thickness
    idomain = nlmod.dims.layers.get_idomain(layer_model_active)

    data_vars = {
        "kh": layer_model_active["kh"],
        "kv": layer_model_active["kv"],
        "botm": layer_model_active["botm"],
        "top": layer_model_active["top"],
        "area": ds_regis.get("area", nlmod.dims.get_area(ds_regis)),
        "xv": ds_regis["xv"],
        "yv": ds_regis["yv"],
        "icvert": ds_regis["icvert"],
        "idomain": idomain,
    }
    if return_diagnostics:
        # Category arrays share the merged layer dim
        cat = cat.sel(layer=layer_model_active.layer.values)
        data_vars["cat_botm"] = cat["botm"]
        data_vars["cat_kh"] = cat["kh"]
        data_vars["cat_kv"] = cat["kv"]
        # PWN-layer variables use a separate dim to avoid layer name conflicts
        data_vars["botm_pwn"] = ds_pwn["botm"].rename({"layer": "layer_pwn"})
        data_vars["botm_method"] = method_ds["botm_method"].rename({"layer": "layer_pwn"})
        data_vars["kh_method"] = method_ds["kh_method"].rename({"layer": "layer_pwn"})
        data_vars["kv_method"] = method_ds["kv_method"].rename({"layer": "layer_pwn"})

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": layer_model_active.coords["x"],
            "y": layer_model_active.coords["y"],
            "layer": layer_model_active.coords["layer"],
        },
        attrs=ds_regis.attrs,
    )


def get_ds(
    *,
    ds_regis,
    data_path_2024,
    top,
    anisotropy=10.0,
    distance_transition=250.0,
    fix_min_layer_thickness=True,
    fill_value_kh=5.0,
    fill_value_kv=1.0,
    return_methods=False,
):
    """Compute PWN layer model dataset with mask and transition zone.

    Assembles the PWN layer model by computing bottom elevations, horizontal
    and vertical hydraulic conductivity, and determining which cells are
    valid (mask) and which are in the transition zone towards REGIS.

    Parameters
    ----------
    ds_regis : xr.Dataset
        REGISII model dataset with grid cell coordinates.
    data_path_2024 : pathlib.Path
        Path to the 2024 PWN data directory.
    top : xr.DataArray
        Top elevation of the model grid.
    anisotropy : float, optional
        Ratio of horizontal to vertical hydraulic conductivity (kh / kv).
        Default is 10.0.
    distance_transition : float, optional
        Width of the transition zone in meters. Default is 250.0.
    fix_min_layer_thickness : bool, optional
        Whether to fix missing bottom elevations and enforce a minimum layer
        thickness. Default is True.
    fill_value_kh : float, optional
        Fill value for kh in cells with zero layer thickness (m/day).
        Default is 1.0.
    fill_value_kv : float, optional
        Fill value for kv in cells with zero layer thickness (m/day).
        Default is 0.1.
    return_methods : bool, optional
        If True, return an additional xr.Dataset containing integer arrays
        ``botm_method``, ``kh_method``, and ``kv_method`` that record which
        computation method determined the value of each cell. Default is False.

    Returns
    -------
    ds : xr.Dataset
        Dataset with variables 'top', 'botm', 'kh', 'kv'.
    mask : xr.Dataset
        Boolean mask per variable indicating valid cells.
    transition : xr.Dataset
        Boolean mask per variable indicating transition zone cells.
    methods : xr.Dataset, optional
        Only returned when ``return_methods=True``. Contains integer
        DataArrays ``botm_method``, ``kh_method``, ``kv_method`` with
        per-cell codes describing the computation method. See the
        ``flag_meanings`` attribute on each DataArray for descriptions.
    """
    # Compute boundaries and per-layer masks once, pass to sub-functions
    gdf_boundaries = get_gdf_boundaries(data_path_2024)
    isin_bounds = _compute_isin_bounds(ds=ds_regis, gdf_boundaries=gdf_boundaries)

    botm_result = get_botm(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        fix_min_layer_thickness=fix_min_layer_thickness,
        top=top,
        isin_bounds=isin_bounds,
        return_method=return_methods,
    )
    if return_methods:
        botm, botm_method = botm_result
    else:
        botm = botm_result
    thickness = get_thickness(botm=botm)
    kh_result = get_kh(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        botm=botm,
        anisotropy=anisotropy,
        fill_value_kh=fill_value_kh,
        isin_bounds=isin_bounds,
        return_method=return_methods,
    )
    if return_methods:
        kh, kh_method = kh_result
    else:
        kh = kh_result
    kv_result = get_kv(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        kh=kh,
        thickness=thickness,
        anisotropy=anisotropy,
        fill_value_kv=fill_value_kv,
        isin_bounds=isin_bounds,
        return_method=return_methods,
    )
    if return_methods:
        kv, kv_method = kv_result
    else:
        kv = kv_result

    ds = xr.Dataset(
        {
            "top": top,
            "botm": botm,
            "kh": kh,
            "kv": kv,
            "xv": ds_regis["xv"],
            "yv": ds_regis["yv"],
            "icvert": ds_regis["icvert"],
        },
        coords={"layer": layer_names},
        attrs=ds_regis.attrs.copy(),
    )

    # Compute mask; where ds is valid
    mask_da = _isin_bounds_to_da(isin_bounds=isin_bounds, ds=ds_regis)
    mask = xr.Dataset(
        {
            "botm": mask_da,
            "kh": mask_da,
            "kv": mask_da,
        },
    )

    # Compute transition; where ds is in transition zone towards REGIS
    transition_da = get_transition(
        ds=ds_regis, gdf_boundaries=gdf_boundaries, distance_transition=distance_transition, mask=mask_da
    )
    transition = xr.Dataset(
        {
            "botm": transition_da,
            "kh": transition_da,
            "kv": transition_da,
        },
    )

    # Validate consistency between data, mask, and transition
    for var in ["kh", "kv", "botm"]:
        if not (ds[var].notnull() == mask[var]).all():
            msg = f"{var} has NaN values in cells where mask is True"
            raise ValueError(msg)
        if not (ds[var].isnull() == ~mask[var]).all():
            msg = f"{var} has valid values in cells where mask is False"
            raise ValueError(msg)
        if not np.isfinite(ds[var].where(mask[var], 0.0)).all():
            msg = f"{var} has non-finite values (inf/-inf) in cells where mask is True"
            raise ValueError(msg)
    for var in ["kh", "kv"]:
        if not (ds[var].where(mask[var], 1.0) > 0.0).all():
            msg = f"{var} has non-positive values in cells where mask is True"
            raise ValueError(msg)
        if not ((mask[var].astype(int) + transition[var].astype(int)) <= 1).all():
            msg = f"Overlap between mask and transition for {var}"
            raise ValueError(msg)

    if return_methods:
        methods = xr.Dataset(
            {
                "botm_method": botm_method,
                "kh_method": kh_method,
                "kv_method": kv_method,
            },
        )
        return (
            ds,
            mask,
            transition,
            methods,
        )
    return (
        ds,
        mask,
        transition,
    )


def get_gdf_boundaries(data_path_2024):
    """Load layer boundary polygons from GeoJSON files.

    Each hydrogeological unit (e.g., "11", "12") has a single boundary polygon
    stored in boundaries/S{unit}/S{unit}.geojson. This boundary is shared by
    both the aquifer (W) and aquitard (S) of that unit.

    Data comes from the nhdz model and the Bergen model. There is no data for
    "22", "31", "32" for Bergen, so the boundaries for these units are only
    based on the nhdz model.

    Parameters
    ----------
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing boundaries/.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame indexed by layer name (W11, S11, W12, ..., S32) with
        boundary geometry for each layer.
    """
    data = {}
    for name in ["11", "12", "13", "21", "22", "31", "32"]:
        fp = data_path_2024 / "boundaries" / f"S{name}" / f"S{name}.geojson"
        gdf = gpd.read_file(fp, driver="GeoJSON")
        for aquifer_aquitard in ["W", "S"]:
            data[f"{aquifer_aquitard}{name}"] = gdf.iloc[0].copy()
    gdf_out = gpd.GeoDataFrame(data).T
    gdf_out = gdf_out.set_geometry("geometry")
    return gdf_out.set_crs(gdf.crs)


def _compute_isin_bounds(*, ds, gdf_boundaries):
    """Compute per-layer boolean array indicating cells within boundary polygons.

    Cells where ``isin_bounds`` is True are expected to have valid (non-NaN)
    data values for all layer variables (botm, kh, kv).

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates.
    gdf_boundaries : gpd.GeoDataFrame
        Boundary polygons indexed by layer name.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_layers, n_cells). True where the cell is
        fully contained within the layer boundary polygon.
    """
    mask_data = np.full((len(layer_names), ds.icell2d.size), False)
    for i, name in enumerate(layer_names):
        mask_data[i] = gdf_to_bool_da(gdf_boundaries.loc[[name]], ds, min_area_fraction=1.0).values
    return mask_data


def _isin_bounds_to_da(*, isin_bounds, ds):
    """Convert per-layer boolean mask array to xr.DataArray.

    Parameters
    ----------
    isin_bounds : np.ndarray
        Boolean array of shape (n_layers, n_cells).
    ds : xr.Dataset
        Model dataset with grid cell coordinates.

    Returns
    -------
    xr.DataArray
        Boolean mask with dimensions (layer, icell2d).
    """
    return xr.DataArray(
        isin_bounds,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="mask",
        attrs={"long_name": "Mask indicating valid model cells"},
    )


def get_botm(*, ds, data_path_2024, fix_min_layer_thickness=True, top=None, isin_bounds=None, return_method=False):
    """Compute bottom elevations for all model layers.

    Reads bottom elevation point data from a GeoJSON file and interpolates
    it onto the model grid using linear griddata interpolation. Each layer
    is only assigned values within its defined boundary polygon.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing botm/botm.geojson and
        boundary definitions.
    fix_min_layer_thickness : bool, optional
        Whether to fix missing bottom elevations and enforce a minimum layer
        thickness. Default is True.
    top : xr.DataArray, optional
        Top elevations of the model layers. Required if fix_min_layer_thickness
        is True and not provided, the top elevations will be taken from ds.top.
    isin_bounds : np.ndarray, optional
        Pre-computed boolean array of shape (n_layers, n_cells) from
        ``_compute_isin_bounds``. True where valid data values are expected.
        If None, computed internally.
    return_method : bool, optional
        If True, also return an integer DataArray indicating the computation
        method used per cell. Default is False.

    Returns
    -------
    botm : xr.DataArray
        Bottom elevations with dimensions (layer, icell2d) in mNAP. Cells
        outside the layer boundary are NaN.
    botm_method : xr.DataArray, optional
        Only returned when ``return_method=True``. Integer array (same shape)
        encoding the method used per cell. See ``flag_meanings`` attribute.
    """
    # TODO: [Edinsi 3.1, p.12] Edinsi notes that depth shapefiles extend west of the coastline
    #   into the North Sea. The nearest-neighbor fallback below may assign inappropriate values
    #   to cells under the seabed where layers may not exist. Edinsi recommends investigating.
    #   Consider using the noordzee_clip polygon (bodemlagen_pwn_2024/v2.0.0/noordzee_clip/
    #   noordzee_clip.geojson) to mask out North Sea cells and ensure interpolation does not
    #   extrapolate layer elevations into areas where geological layers may be absent.
    # TODO: [Edinsi 4.4, p.36-37] Edinsi notes a thickness jump from 1.5m to 0.2m for S1.1 at
    #   the Koster/Bergen boundary. This discontinuity propagates through the interpolation here.
    fp = data_path_2024 / "botm" / "botm.geojson"
    gdf_botm = gpd.read_file(fp, driver="GeoJSON")

    if isin_bounds is None:
        gdf_boundaries = get_gdf_boundaries(data_path_2024)
        isin_bounds = _compute_isin_bounds(ds=ds, gdf_boundaries=gdf_boundaries)

    xy_in = np.column_stack((gdf_botm.geometry.x.values, gdf_botm.geometry.y.values))
    xy_out = np.column_stack((ds.x.values, ds.y.values))

    data = np.full((len(layer_names), xy_out.shape[0]), np.nan)
    if return_method:
        method_data = np.zeros((len(layer_names), xy_out.shape[0]), dtype=np.int8)
    for i, layer_name in enumerate(layer_names):
        mask = isin_bounds[i]
        src_values = gdf_botm[layer_name].values.astype(float)
        valid_src = np.isfinite(src_values)
        values = griddata(xy_in[valid_src], src_values[valid_src], xy_out[mask], method="linear")
        nan_mask = np.isnan(values)
        if return_method:
            method_local = np.ones(mask.sum(), dtype=np.int8)  # 1 = linear interpolation
        if nan_mask.any():
            values[nan_mask] = griddata(
                xy_in[valid_src], src_values[valid_src], xy_out[mask][nan_mask], method="nearest"
            )
            if return_method:
                method_local[nan_mask] = 2  # nearest-neighbor fallback
        data[i, mask] = values
        if return_method:
            method_data[i, mask] = method_local

    botm = xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="botm",
        attrs={"units": "mNAP", "long_name": "Bottom elevation of model layers with respect to NAP"},
    )

    # Clip to boundary mask before fix so ffill only operates within layer masks
    botm = botm.where(isin_bounds)

    if fix_min_layer_thickness:
        if top is None:
            top = ds.top

        botm_before_fix = botm.copy() if return_method else None
        botm = fix_missings_botms_and_min_layer_thickness(top=top, botm=botm)
        # Re-mask to avoid synthetic fill-from-top outside all layer boundaries
        any_layer_mask = isin_bounds.any(axis=0)
        botm = botm.where(any_layer_mask)
        # Ensure final NaN pattern matches isin_bounds per layer
        botm = botm.where(isin_bounds)
        if return_method:
            was_null = botm_before_fix.isnull().values
            now_valid = botm.notnull().values
            was_shifted = (
                ~np.isclose(botm_before_fix.values, botm.values, equal_nan=True) & botm_before_fix.notnull().values
            )
            method_data[was_null & now_valid] = 3  # forward-fill from layer above
            method_data[was_shifted] = 4  # shifted for minimum thickness

    if return_method:
        method_data[~isin_bounds] = 0
        method_da = xr.DataArray(
            method_data,
            dims=("layer", "icell2d"),
            coords={"layer": layer_names, "icell2d": ds.icell2d},
            name="botm_method",
            attrs={
                "long_name": "Method used to compute botm",
                "flag_values": [0, 1, 2, 3, 4],
                "flag_meanings": (
                    "0: no_data (outside boundary polygon); "
                    "1: linear_interpolation (griddata linear from botm.geojson point data); "
                    "2: nearest_interpolation (griddata nearest fallback where linear produced NaN); "
                    "3: forward_fill (missing botm filled from layer above by fix_missings_botms_and_min_layer_thickness); "
                    "4: shifted_for_min_thickness (botm shifted downward to enforce monotonically decreasing sequence)"
                ),
            },
        )
        return botm, method_da
    return botm


def get_thickness(*, botm):
    """Compute layer thicknesses from bottom elevations.

    Thickness is calculated as the difference between consecutive layer
    bottoms: ``thickness[k] = botm[k-1] - botm[k]``. The first layer (W11)
    is excluded because it requires the model top elevation.

    Parameters
    ----------
    botm : xr.DataArray
        Bottom elevations with dimensions (layer, icell2d), as returned by
        ``get_botm``.

    Returns
    -------
    xr.DataArray
        Layer thicknesses with dimensions (layer, icell2d) in meters.
        Contains one fewer layer than the input (layers S11 through S32).
    """
    thickness = -botm.diff(dim="layer")
    thickness.name = "thickness"
    thickness.attrs["units"] = "m"
    thickness.attrs["long_name"] = "Thickness of model layers"
    return thickness


def _guard_zero_thickness(values, thickness, fill_value, layer_name, region=""):
    """Replace values at zero-thickness cells with a fill value.

    Parameters
    ----------
    values : np.ndarray
        Computed kh or kv values.
    thickness : np.ndarray
        Layer thickness array (same shape as values).
    fill_value : float
        Fill value for zero-thickness cells.
    layer_name : str
        Layer name for log messages.
    region : str, optional
        Region name for log messages (e.g., "NHDZ", "Bergen").

    Returns
    -------
    np.ndarray
        Values with zero-thickness cells replaced by fill_value.
    """
    zero_d = np.isclose(thickness, 0.0) | ~np.isfinite(thickness)
    if zero_d.any():
        region_str = f" {region}" if region else ""
        logger.warning(
            "Layer %s%s: %d cells have zero or undefined thickness. Setting to fill_value=%.2f.",
            layer_name,
            region_str,
            zero_d.sum(),
            fill_value,
        )
        values[zero_d] = fill_value
    return values


def get_kh(*, ds, data_path_2024, botm=None, anisotropy=10.0, fill_value_kh=5.0, isin_bounds=None, return_method=False):
    """Compute horizontal hydraulic conductivity (kh) for all model layers.

    Assigns kh values to each grid cell per layer, using different data sources
    and methods depending on the layer type:

    - **W-layers (aquifers):** kh is read directly from polygon GeoJSON files
      (K{name}_combined.geojson) and mapped to the grid using area-weighted
      aggregation.
    - **S-layers with KD data (S12, S13, S21, S22, S31):** Two sub-regions are
      handled separately:

      - *NHDZ area:* Transmissivity (KD) is interpolated from point data and
        converted to kh via kh = KD / d.
      - *Bergen area:* Vertical resistance (c) is read from polygon data,
        aggregated using a harmonic area-weighted mean, and converted to kh
        via kh = d * anisotropy / c.
    - **S-layers with c data only (S11, S32):** Vertical resistance (c) from
      polygon data is aggregated using a harmonic area-weighted mean and
      converted to kh via kh = d * anisotropy / c across the full grid.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing conductances, boundaries,
        and bottom elevation data.
    botm : xr.DataArray, optional
        Bottom elevations per layer. If None, computed via ``get_botm``.
    anisotropy : float, optional
        Ratio of horizontal to vertical hydraulic conductivity (kh / kv).
        Used when deriving kh from vertical resistance. Default is 10.0.
    fill_value_kh : float, optional
        Fill value for kh in cells with zero layer thickness (m/day).
        These cells will be set inactive downstream. Default is 1.0.
    isin_bounds : np.ndarray, optional
        Pre-computed boolean array of shape (n_layers, n_cells) from
        ``_compute_isin_bounds``. True where valid data values are expected.
        If None, computed internally.
    return_method : bool, optional
        If True, also return an integer DataArray indicating the computation
        method used per cell. Default is False.

    Returns
    -------
    kh : xr.DataArray
        Horizontal hydraulic conductivity with dimensions (layer, icell2d)
        in m/day. Cells outside the defined boundaries are NaN.
    kh_method : xr.DataArray, optional
        Only returned when ``return_method=True``. Integer array (same shape)
        encoding the method used per cell. See ``flag_meanings`` attribute.
    """
    if botm is None:
        botm = get_botm(ds=ds, data_path_2024=data_path_2024)

    thickness = get_thickness(botm=botm)

    gdf_nhdz = gpd.read_file(data_path_2024 / "boundaries" / "triwaco_model_nhdz.geojson", driver="GeoJSON")
    xy_out = np.column_stack((ds.x.values, ds.y.values))
    is_within_nhdz_bound = gdf_to_bool_da(gdf_nhdz, ds, min_area_fraction=1.0).values

    if isin_bounds is None:
        gdf_boundaries = get_gdf_boundaries(data_path_2024)
        isin_bounds = _compute_isin_bounds(ds=ds, gdf_boundaries=gdf_boundaries)

    data = np.full((len(layer_names), ds.icell2d.size), np.nan)
    if return_method:
        method_data = np.zeros((len(layer_names), ds.icell2d.size), dtype=np.int8)

    for i, name in enumerate(layer_names):
        mask = isin_bounds[i]
        if name[0] == "W":
            fp = data_path_2024 / "conductances" / f"K{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            values = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="VALUE", agg_method="area_weighted").values
            data[i, mask] = values[mask]
            if return_method:
                method_data[i, mask] = 1
        elif name in {"S12", "S13", "S21", "S22", "S31"}:
            # NHDZ area: kh from transmissivity (KD) point data
            is_nhdz = mask & is_within_nhdz_bound
            fp = data_path_2024 / "conductances" / f"KD{name}_NHDZ.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            xy_in = np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))
            d_nhdz = thickness.sel(layer=name).isel(icell2d=is_nhdz).values
            kd_nhdz = griddata(xy_in, gdf["VALUE"].values, xy_out[is_nhdz], method="linear")
            nan_mask = np.isnan(kd_nhdz)
            if return_method:
                method_nhdz = np.full(is_nhdz.sum(), 2, dtype=np.int8)  # linear KD interp
            if nan_mask.any():
                kd_nhdz[nan_mask] = griddata(xy_in, gdf["VALUE"].values, xy_out[is_nhdz][nan_mask], method="nearest")
                if return_method:
                    method_nhdz[nan_mask] = 3  # nearest KD fallback
            safe_d = np.where(np.isclose(d_nhdz, 0.0), np.nan, d_nhdz)
            kh_values = kd_nhdz / safe_d
            data[i, is_nhdz] = _guard_zero_thickness(kh_values, d_nhdz, fill_value_kh, name, "NHDZ")
            if return_method:
                method_nhdz[np.isclose(d_nhdz, 0.0)] = 6  # fill value
                method_data[i, is_nhdz] = method_nhdz

            # Bergen area: kh from vertical resistance (c) polygon data
            is_bergen = mask & ~is_within_nhdz_bound
            fp = data_path_2024 / "conductances" / f"C{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]

            # Harmonic area-weighted mean: aggregate 1/c, then kh = d * anisotropy / c
            d_bergen = thickness.sel(layer=name).isel(icell2d=is_bergen).values
            gdf["inv_VALUE"] = 1.0 / gdf["VALUE"]
            inv_c = nlmod.dims.gdf_to_da(
                gdf=gdf, ds=ds.isel(icell2d=is_bergen), column="inv_VALUE", agg_method="area_weighted"
            ).values
            safe_d = np.where(np.isclose(d_bergen, 0.0), np.nan, d_bergen)
            kh_values = safe_d * anisotropy * inv_c
            data[i, is_bergen] = _guard_zero_thickness(kh_values, d_bergen, fill_value_kh, name, "Bergen")
            if return_method:
                method_bergen = np.full(is_bergen.sum(), 4, dtype=np.int8)  # Bergen c->kh
                method_bergen[np.isclose(d_bergen, 0.0)] = 6  # fill value
                method_data[i, is_bergen] = method_bergen
        elif name in {"S11", "S32"}:
            fp = data_path_2024 / "conductances" / f"C{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]

            # Harmonic area-weighted mean: aggregate 1/c, then kh = d * anisotropy / c
            d = thickness.sel(layer=name).values
            gdf["inv_VALUE"] = 1.0 / gdf["VALUE"]
            inv_c = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="inv_VALUE", agg_method="area_weighted").values
            safe_d = np.where(np.isclose(d, 0.0), np.nan, d)
            kh_values = safe_d * anisotropy * inv_c
            kh_values = _guard_zero_thickness(kh_values, d, fill_value_kh, name)
            data[i, mask] = kh_values[mask]
            if return_method:
                method_local = np.full(d.shape, 5, dtype=np.int8)  # S11/S32 c->kh full extent
                method_local[np.isclose(d, 0.0)] = 6  # fill value
                method_data[i, mask] = method_local[mask]
        else:
            msg = f"Unknown layer name {name} for assigning kh"
            raise ValueError(msg)

    kh_da = xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="kh",
        attrs={"units": "m/day", "long_name": "Horizontal hydraulic conductivity"},
    )
    if return_method:
        method_da = xr.DataArray(
            method_data,
            dims=("layer", "icell2d"),
            coords={"layer": layer_names, "icell2d": ds.icell2d},
            name="kh_method",
            attrs={
                "long_name": "Method used to compute kh",
                "flag_values": [0, 1, 2, 3, 4, 5, 6],
                "flag_meanings": (
                    "0: no_data (outside boundary polygon); "
                    "1: W_layer_polygon_kh (direct kh from area-weighted K polygon data); "
                    "2: S_layer_NHDZ_KD_linear (kh = KD/d, KD from linear interpolation of point data); "
                    "3: S_layer_NHDZ_KD_nearest (kh = KD/d, KD from nearest-neighbor interpolation fallback); "
                    "4: S_layer_Bergen_c_to_kh (kh = d*anisotropy/c, c from harmonic area-weighted polygon data); "
                    "5: S_layer_c_to_kh (kh = d*anisotropy/c, c from harmonic area-weighted polygon data, full extent); "
                    "6: fill_value (zero-thickness cell, set to fill_value_kh)"
                ),
            },
        )
        return kh_da, method_da
    return kh_da


def get_kv(
    *, ds, data_path_2024, kh, thickness, anisotropy=10.0, fill_value_kv=1.0, isin_bounds=None, return_method=False
):
    """Compute vertical hydraulic conductivity (kv) for all model layers.

    Assigns kv values to each grid cell per layer:

    - **W-layers (aquifers):** kv is derived from kh using the anisotropy
      ratio: kv = kh / anisotropy.
    - **S-layers (aquitards):** kv is computed from vertical resistance (c)
      polygon data using a harmonic area-weighted mean: kv = d / c.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing conductance GeoJSON files.
    kh : xr.DataArray
        Horizontal hydraulic conductivity with dimensions (layer, icell2d),
        as returned by ``get_kh``.
    thickness : xr.DataArray
        Layer thicknesses with dimensions (layer, icell2d), as returned by
        ``get_thickness``.
    anisotropy : float, optional
        Ratio of horizontal to vertical hydraulic conductivity (kh / kv).
        Used for W-layers. Default is 10.0.
    fill_value_kv : float, optional
        Fill value for kv in cells with zero layer thickness (m/day).
        These cells will be set inactive downstream. Default is 0.1.
    isin_bounds : np.ndarray, optional
        Pre-computed boolean array of shape (n_layers, n_cells) from
        ``_compute_isin_bounds``. True where valid data values are expected.
        If None, computed internally.
    return_method : bool, optional
        If True, also return an integer DataArray indicating the computation
        method used per cell. Default is False.

    Returns
    -------
    kv : xr.DataArray
        Vertical hydraulic conductivity with dimensions (layer, icell2d)
        in m/day.
    kv_method : xr.DataArray, optional
        Only returned when ``return_method=True``. Integer array (same shape)
        encoding the method used per cell. See ``flag_meanings`` attribute.
    """
    if isin_bounds is None:
        gdf_boundaries = get_gdf_boundaries(data_path_2024)
        isin_bounds = _compute_isin_bounds(ds=ds, gdf_boundaries=gdf_boundaries)

    data = np.full((len(layer_names), ds.icell2d.size), np.nan)
    if return_method:
        method_data = np.zeros((len(layer_names), ds.icell2d.size), dtype=np.int8)

    for i, name in enumerate(layer_names):
        mask = isin_bounds[i]
        if name[0] == "W":
            data[i, mask] = (kh.sel(layer=name) / anisotropy).values[mask]
            if return_method:
                method_data[i, mask] = 1
        elif name[0] == "S":
            fp = data_path_2024 / "conductances" / f"C{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]

            # Harmonic area-weighted mean: aggregate 1/c, then kv = d / c
            d = thickness.sel(layer=name).values
            gdf["inv_VALUE"] = 1.0 / gdf["VALUE"]
            inv_c = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="inv_VALUE", agg_method="area_weighted").values
            safe_d = np.where(np.isclose(d, 0.0), np.nan, d)
            kv_values = safe_d * inv_c
            kv_values = _guard_zero_thickness(kv_values, d, fill_value_kv, name)
            data[i, mask] = kv_values[mask]
            if return_method:
                method_local = np.full(d.shape, 2, dtype=np.int8)  # kv = d / c
                method_local[np.isclose(d, 0.0)] = 3  # fill value
                method_data[i, mask] = method_local[mask]
        else:
            msg = f"Unknown layer name {name} for assigning kv"
            raise ValueError(msg)

    kv_da = xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="kv",
        attrs={"units": "m/day", "long_name": "Vertical hydraulic conductivity"},
    )
    if return_method:
        method_da = xr.DataArray(
            method_data,
            dims=("layer", "icell2d"),
            coords={"layer": layer_names, "icell2d": ds.icell2d},
            name="kv_method",
            attrs={
                "long_name": "Method used to compute kv",
                "flag_values": [0, 1, 2, 3],
                "flag_meanings": (
                    "0: no_data (outside boundary polygon); "
                    "1: W_layer_kh_anisotropy (kv = kh/anisotropy); "
                    "2: S_layer_d_over_c (kv = d/c, c from harmonic area-weighted polygon data); "
                    "3: fill_value (zero-thickness cell, set to fill_value_kv)"
                ),
            },
        )
        return kv_da, method_da
    return kv_da


def get_mask(*, ds, data_path_2024):
    """Compute mask for valid model cells based on layer boundaries.

    A cell is considered valid (True) for a given layer only if it is fully
    contained within the boundary polygon (``min_area_fraction=1``). Cells
    that merely touch the boundary are excluded, ensuring all masked cells
    have complete data coverage.

    Uses ``nlmod.dims.grid.gdf_to_bool_da`` with ``min_area_fraction=1`` to
    determine full containment via grid intersection rather than center point
    containment.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing boundary GeoJSON files.

    Returns
    -------
    xr.DataArray
        Boolean mask with dimensions (layer, icell2d), where True indicates
        cells fully within the boundary.
    """
    gdf_boundaries = get_gdf_boundaries(data_path_2024)
    isin_bounds = _compute_isin_bounds(ds=ds, gdf_boundaries=gdf_boundaries)
    return _isin_bounds_to_da(isin_bounds=isin_bounds, ds=ds)


def get_transition(*, ds, gdf_boundaries=None, data_path_2024=None, distance_transition, mask=None):
    """Compute transition zone mask around layer boundaries.

    A cell is in the transition zone if it is fully inside the buffered
    boundary but not fully inside the original boundary (i.e., not in mask).
    The buffered boundary is the original boundary expanded outward by
    ``distance_transition``.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    gdf_boundaries : gpd.GeoDataFrame, optional
        Pre-computed boundary polygons from ``get_gdf_boundaries``. If None,
        computed from ``data_path_2024``.
    data_path_2024 : pathlib.Path, optional
        Path to the 2024 data directory. Used only if ``gdf_boundaries`` is
        None.
    distance_transition : float
        Buffer distance in meters to expand the boundary outward.
    mask : xr.DataArray, optional
        Pre-computed mask from ``get_mask``. If None, it is computed
        internally.

    Returns
    -------
    xr.DataArray
        Boolean mask with dimensions (layer, icell2d), where True indicates
        cells in the transition zone.
    """
    if mask is None:
        mask = get_mask(ds=ds, data_path_2024=data_path_2024)

    if gdf_boundaries is None:
        gdf_boundaries = get_gdf_boundaries(data_path_2024)

    data = np.full((len(layer_names), ds.icell2d.size), False)
    for i, name in enumerate(layer_names):
        boundary_geom = gdf_boundaries.loc[name].geometry
        buffered_geom = boundary_geom.buffer(distance_transition)

        # Cells fully inside the buffered boundary
        data[i] = gdf_to_bool_da(buffered_geom, ds, min_area_fraction=1.0).values & ~mask.isel(layer=i).values

    return xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="transition",
        attrs={"long_name": "Transition zone around layer boundaries"},
    )


def fix_missings_botms_and_min_layer_thickness(*, top=None, botm=None):
    """Fix missing bottom elevations and enforce positive layer thickness.

    Missing bottom elevations are filled downward from the layer above.
    Bottom elevations are then adjusted so that each layer's bottom is at
    or below the bottom of the layer above (monotonically decreasing),
    ensuring all layers have non-negative thickness.

    Parameters
    ----------
    top : xr.DataArray
        Top elevation of the model. Must not contain NaN values.
    botm : xr.DataArray
        Bottom elevations with dimensions (layer, icell2d).

    Returns
    -------
    xr.DataArray
        Corrected bottom elevations.

    Raises
    ------
    ValueError
        If top contains NaN values.
    """
    if top.isnull().any():
        msg = "Top should not contain nan values"
        raise ValueError(msg)

    out = xr.concat((top.expand_dims(dim={"layer": ["mv"]}, axis=0), botm), dim="layer")
    # Use ffill here to fill the nan's with the previous layer. Layer thickness is zero for non existing layers
    out = out.ffill(dim="layer")
    layer_axis = out.get_axis_num("layer")
    out = xr.apply_ufunc(
        lambda a: np.minimum.accumulate(a, axis=layer_axis),
        out,
        dask="parallelized",
        output_dtypes=[out.dtype],
    )
    botm_fixed = out.isel(layer=slice(1, None)).transpose("layer", "icell2d")

    # inform
    ncell = botm.size
    nisnull = int(botm.isnull().sum())
    nshifted = int((~np.isclose(botm, botm_fixed) & botm.notnull()).sum())
    logger.info(
        "Fixed %.1f%% missing botms using downward fill. Shifted %.1f%% botms to ensure all layers have a positive thickness, assuming more info is in the upper layer.",
        nisnull / ncell * 100.0,
        nshifted / ncell * 100.0,
    )
    return botm_fixed
