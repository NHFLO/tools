"""Module containing functions to retrieve PWN bodemlagen."""

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

# from nhflotools.pwnlayers.utils import fix_missings_botms_and_min_layer_thickness

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

    Returns
    -------
    xr.Dataset
        Merged dataset with variables 'kh', 'kv', 'botm', 'top', 'area',
        'xv', 'yv', 'icvert', and 'idomain'.
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
    ds_pwn, mask_pwn, transition_pwn = get_ds(
        ds_regis=ds_regis,
        data_path_2024=data_path_2024,
        top=top,
        anisotropy=anisotropy,
        distance_transition=distance_transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )
    ds_pwn_thickness = nlmod.dims.layers.calculate_thickness(ds_pwn)
    if (ds_pwn_thickness < 0.0).any():
        msg = "PWN layer thickness should be positive"
        raise ValueError(msg)

    # Combine PWN layer model with REGIS layer model
    layer_model_pwn_regis, _ = combine_two_layer_models(
        layer_model_regis=layer_model_regis,
        layer_model_other=ds_pwn[["botm", "kh", "kv"]],
        mask_model_other=mask_pwn[["botm", "kh", "kv"]],
        transition_model=transition_pwn[["botm", "kh", "kv"]],
        top=top,
        df_koppeltabel=df_koppeltabel,
        koppeltabel_header_regis="Regis II v2.2",
        koppeltabel_header_other="ASSUMPTION1",
    )
    if fix_min_layer_thickness:
        layer_model_pwn_regis["botm"] = fix_missings_botms_and_min_layer_thickness(top=top, botm=layer_model_pwn_regis["botm"])

    m = np.isclose(layer_model_pwn_regis.kh, 0.0)
    if np.any(m):
        logger.warning("Setting %d values of kh that are exactly zero to 1e-6 m/d.", m.sum().item())
        layer_model_pwn_regis["kh"] = layer_model_pwn_regis.kh.where(~m, 1e-6)

    m = np.isclose(layer_model_pwn_regis.kv, 0.0)
    if np.any(m):
        logger.warning("Setting %d values of kv that are exactly zero to 1e-6 m/d.", m.sum().item())
        layer_model_pwn_regis["kv"] = layer_model_pwn_regis.kv.where(~m, 1e-6)

    layer_model_active = nlmod.layers.fill_nan_top_botm_kh_kv(
        layer_model_pwn_regis,
        anisotropy=anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
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


def get_ds(
    *,
    ds_regis,
    data_path_2024,
    top,
    anisotropy=10.0,
    distance_transition=250.0,
    fix_min_layer_thickness=True,
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

    Returns
    -------
    ds : xr.Dataset
        Dataset with variables 'top', 'botm', 'kh', 'kv'.
    mask : xr.Dataset
        Boolean mask per variable indicating valid cells.
    transition : xr.Dataset
        Boolean mask per variable indicating transition zone cells.
    """
    botm = get_botm(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        fix_min_layer_thickness=fix_min_layer_thickness,
        top=top,
    )
    thickness = get_thickness(botm=botm)
    kh = get_kh(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        botm=botm,
        anisotropy=anisotropy,
    )
    kv = get_kv(
        ds=ds_regis,
        data_path_2024=data_path_2024,
        kh=kh,
        thickness=thickness,
        anisotropy=anisotropy,
    )

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
    mask_da = get_mask(ds=ds_regis, data_path_2024=data_path_2024)
    mask = xr.Dataset(
        {
            "botm": mask_da,
            "kh": mask_da,
            "kv": mask_da,
        },
    )

    # Compute transition; where ds is in transition zone towards REGIS
    transition_da = get_transition(
        ds=ds_regis, data_path_2024=data_path_2024, distance_transition=distance_transition, mask=mask_da
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
        if not ((mask[var].astype(int) + transition[var].astype(int)) <= 1).all():
            msg = f"Overlap between mask and transition for {var}"
            raise ValueError(msg)

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


def get_botm(*, ds, data_path_2024, fix_min_layer_thickness=True, top=None):
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

    Returns
    -------
    xr.DataArray
        Bottom elevations with dimensions (layer, icell2d) in mNAP. Cells
        outside the layer boundary are NaN.
    """
    fp = data_path_2024 / "botm" / "botm.geojson"
    gdf_botm = gpd.read_file(fp, driver="GeoJSON")

    gdf_boundaries = get_gdf_boundaries(data_path_2024)

    xy_in = np.column_stack((gdf_botm.geometry.x.values, gdf_botm.geometry.y.values))
    xy_out = np.column_stack((ds.x.values, ds.y.values))

    data = np.full((len(layer_names), xy_out.shape[0]), np.nan)
    mask_data = np.full_like(data, False, dtype=bool)
    for i, layer_name in enumerate(layer_names):
        mask = gdf_to_bool_da(gdf_boundaries.loc[[layer_name]], ds, min_area_fraction=1.0).values
        mask_data[i] = mask
        values = griddata(xy_in, gdf_botm[layer_name].values, xy_out[mask], method="linear")
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = griddata(xy_in, gdf_botm[layer_name].values, xy_out[mask][nan_mask], method="nearest")
        data[i, mask] = values

    botm = xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="botm",
        attrs={"units": "mNAP", "long_name": "Bottom elevation of model layers with respect to NAP"},
    )

    if fix_min_layer_thickness:
        if top is None:
            top = ds.top

        botm = fix_missings_botms_and_min_layer_thickness(top=top, botm=botm)

    # Clip to boundary mask (fix_min_layer_thickness ffill may extend beyond boundaries)
    botm = botm.where(mask_data)
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


def get_kh(*, ds, data_path_2024, botm=None, anisotropy=10.0):
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

    Returns
    -------
    xr.DataArray
        Horizontal hydraulic conductivity with dimensions (layer, icell2d)
        in m/day. Cells outside the defined boundaries are NaN.
    """
    if botm is None:
        botm = get_botm(ds=ds, data_path_2024=data_path_2024)

    thickness = get_thickness(botm=botm)

    gdf_nhdz = gpd.read_file(data_path_2024 / "boundaries" / "triwaco_model_nhdz.geojson", driver="GeoJSON")
    xy_out = np.column_stack((ds.x.values, ds.y.values))
    is_within_nhdz_bound = gdf_to_bool_da(gdf_nhdz, ds, min_area_fraction=1.0).values

    gdf_boundaries = get_gdf_boundaries(data_path_2024)

    data = np.full((len(layer_names), ds.icell2d.size), np.nan)

    for i, name in enumerate(layer_names):
        mask = gdf_to_bool_da(gdf_boundaries.loc[[name]], ds, min_area_fraction=1.0).values
        if name[0] == "W":
            fp = data_path_2024 / "conductances" / f"K{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            values = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="VALUE", agg_method="area_weighted").values
            data[i, mask] = values[mask]
        elif name in {"S12", "S13", "S21", "S22", "S31"}:
            # NHDZ area: kh from transmissivity (KD) point data
            is_nhdz = mask & is_within_nhdz_bound
            fp = data_path_2024 / "conductances" / f"KD{name}_NHDZ.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            xy_in = np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))
            d_nhdz = thickness.sel(layer=name).isel(icell2d=is_nhdz).values
            kd_nhdz = griddata(xy_in, gdf["VALUE"].values, xy_out[is_nhdz], method="linear")
            nan_mask = np.isnan(kd_nhdz)
            if nan_mask.any():
                kd_nhdz[nan_mask] = griddata(
                    xy_in, gdf["VALUE"].values, xy_out[is_nhdz][nan_mask], method="nearest"
                )
            data[i, is_nhdz] = kd_nhdz / d_nhdz

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
            data[i, is_bergen] = d_bergen * anisotropy * inv_c
        elif name in {"S11", "S32"}:
            fp = data_path_2024 / "conductances" / f"C{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]

            # Harmonic area-weighted mean: aggregate 1/c, then kh = d * anisotropy / c
            d = thickness.sel(layer=name).values
            gdf["inv_VALUE"] = 1.0 / gdf["VALUE"]
            inv_c = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="inv_VALUE", agg_method="area_weighted").values
            data[i, mask] = (d * anisotropy * inv_c)[mask]
        else:
            msg = f"Unknown layer name {name} for assigning kh"
            raise ValueError(msg)

    return xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="kh",
        attrs={"units": "m/day", "long_name": "Horizontal hydraulic conductivity"},
    )


def get_kv(*, ds, data_path_2024, kh, thickness, anisotropy=10.0):
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

    Returns
    -------
    xr.DataArray
        Vertical hydraulic conductivity with dimensions (layer, icell2d)
        in m/day.
    """
    gdf_boundaries = get_gdf_boundaries(data_path_2024)
    data = np.full((len(layer_names), ds.icell2d.size), np.nan)

    for i, name in enumerate(layer_names):
        mask = gdf_to_bool_da(gdf_boundaries.loc[[name]], ds, min_area_fraction=1.0).values
        if name[0] == "W":
            data[i, mask] = (kh.sel(layer=name) / anisotropy).values[mask]
        elif name[0] == "S":
            fp = data_path_2024 / "conductances" / f"C{name}_combined.geojson"
            gdf = gpd.read_file(fp, driver="GeoJSON")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]

            # Harmonic area-weighted mean: aggregate 1/c, then kv = d / c
            d = thickness.sel(layer=name).values
            gdf["inv_VALUE"] = 1.0 / gdf["VALUE"]
            inv_c = nlmod.dims.gdf_to_da(gdf=gdf, ds=ds, column="inv_VALUE", agg_method="area_weighted").values
            data[i, mask] = (d * inv_c)[mask]
        else:
            msg = f"Unknown layer name {name} for assigning kv"
            raise ValueError(msg)

    return xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="kv",
        attrs={"units": "m/day", "long_name": "Vertical hydraulic conductivity"},
    )


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

    data = np.full((len(layer_names), ds.icell2d.size), False)
    for i, name in enumerate(layer_names):
        data[i] = gdf_to_bool_da(
            gdf_boundaries.loc[[name]],
            ds,
            min_area_fraction=1.0,
        ).values

    return xr.DataArray(
        data,
        dims=("layer", "icell2d"),
        coords={"layer": layer_names, "icell2d": ds.icell2d},
        name="mask",
        attrs={"long_name": "Mask indicating valid model cells"},
    )


def get_transition(*, ds, data_path_2024, distance_transition, mask=None):
    """Compute transition zone mask around layer boundaries.

    A cell is in the transition zone if it is fully inside the buffered
    boundary but not fully inside the original boundary (i.e., not in mask).
    The buffered boundary is the original boundary expanded outward by
    ``distance_transition``.

    Parameters
    ----------
    ds : xr.Dataset
        Model dataset with grid cell coordinates (x, y, icell2d).
    data_path_2024 : pathlib.Path
        Path to the 2024 data directory containing boundary GeoJSON files.
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
    out.values = np.minimum.accumulate(out.values, axis=out.dims.index("layer"))
    botm_fixed = out.isel(layer=slice(1, None)).transpose("layer", "icell2d")

    # inform
    ncell, nisnull = botm.size, botm.isnull().sum()
    nfixed = (~np.isclose(botm, botm_fixed)).sum()
    logger.info(
        "Fixed %.1f%% missing botms using downward fill. Shifted %.1f%% botms to ensure all layers have a positive thickness, assuming more info is in the upper layer.",
        nisnull / ncell * 100.0,
        (nfixed - nisnull) / ncell * 100.0,
    )
    return botm_fixed
