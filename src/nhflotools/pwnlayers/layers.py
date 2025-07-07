import logging

import nlmod
import numpy as np
import pandas as pd
import xarray as xr
from nlmod import cache
from scipy.interpolate import griddata

from nhflotools.pwnlayers.io import read_pwn_data2
from nhflotools.pwnlayers.merge_layer_models import combine_two_layer_models
from nhflotools.pwnlayers.utils import fix_missings_botms_and_min_layer_thickness
from nhflotools.pwnlayers2.layers import get_mensink_layer_model as get_mensink_layer_model2
from nhflotools.pwnlayers2.layers import get_pwn_aquitard_data

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
        fix_missings_botms_and_min_layer_thickness(layer_model_regis)

    # Get PWN layer models
    ds_pwn_data = read_pwn_data2(
        layer_model_regis,
        datadir_mensink=data_path_mensink,
        datadir_bergen=data_path_bergen,
        length_transition=length_transition,
        cachedir=cachedir,
    )
    # Read the koppeltabel CSV file
    df_koppeltabel = pd.read_csv(fname_koppeltabel, skiprows=0, index_col=0)
    df_koppeltabel = df_koppeltabel[~df_koppeltabel["ASSUMPTION1"].isna()]

    if data_path_2024 is not None:
        ds_pwn_data_2024 = get_pwn_aquitard_data(
            ds=ds_regis, data_dir=data_path_2024, ix=None, transition_length=length_transition
        )
        layer_model_nhd, mask_model_nhd, transition_model_nhd = get_mensink_layer_model2(
            ds_pwn_data=ds_pwn_data, ds_pwn_data_2024=ds_pwn_data_2024, fix_min_layer_thickness=fix_min_layer_thickness
        )
        thick_layer_model_nhd = nlmod.dims.layers.calculate_thickness(layer_model_nhd)
        assert ~(thick_layer_model_nhd < 0.0).any(), "NHD thickness of layers should be positive"

        # Combine PWN layer model with REGIS layer model
        layer_model_mensink_bergen_regis, _ = combine_two_layer_models(
            layer_model_regis=layer_model_regis,
            layer_model_other=layer_model_nhd,
            mask_model_other=mask_model_nhd,
            transition_model=transition_model_nhd,
            top=top,
            df_koppeltabel=df_koppeltabel,
            koppeltabel_header_regis="Regis II v2.2",
            koppeltabel_header_other="ASSUMPTION1",
        )

    else:
        layer_model_mensink, mask_model_mensink, transition_model_mensink = get_mensink_layer_model(
            ds_pwn_data=ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness
        )
        layer_model_bergen, mask_model_bergen, transition_model_bergen = get_bergen_layer_model(
            ds_pwn_data=ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness
        )
        thick_layer_model_mensink = nlmod.dims.layers.calculate_thickness(layer_model_mensink)
        thick_layer_model_bergen = nlmod.dims.layers.calculate_thickness(layer_model_bergen)
        thick_layer_model_regis = nlmod.dims.layers.calculate_thickness(layer_model_regis)

        assert ~(thick_layer_model_mensink < 0.0).any(), "Mensink thickness of layers should be positive"
        assert ~(thick_layer_model_bergen < 0.0).any(), "Bergen thickness of layers should be positive"
        assert ~(thick_layer_model_regis < 0.0).any(), "Regis thickness of layers should be positive"

        # Combine PWN layer model with REGIS layer model
        layer_model_mensink_regis, _ = combine_two_layer_models(
            layer_model_regis=layer_model_regis,
            layer_model_other=layer_model_mensink,
            mask_model_other=mask_model_mensink,
            transition_model=transition_model_mensink,
            top=top,
            df_koppeltabel=df_koppeltabel,
            koppeltabel_header_regis="Regis II v2.2",
            koppeltabel_header_other="ASSUMPTION1",
        )
        if fix_min_layer_thickness:
            fix_missings_botms_and_min_layer_thickness(layer_model_mensink_regis)

        # Combine PWN layer model with Bergen layer model and REGIS layer model
        (
            layer_model_mensink_bergen_regis,
            _,
        ) = combine_two_layer_models(
            layer_model_regis=layer_model_mensink_regis,
            layer_model_other=layer_model_bergen,
            mask_model_other=mask_model_bergen,
            transition_model=transition_model_bergen,
            top=top,
            df_koppeltabel=df_koppeltabel,
            koppeltabel_header_regis="Regis II v2.2",
            koppeltabel_header_other="ASSUMPTION1",
        )

    if fix_min_layer_thickness:
        fix_missings_botms_and_min_layer_thickness(layer_model_mensink_bergen_regis)

    # Remove inactive layers and set kh and kv of non-existing cells to default values
    layer_model_mensink_bergen_regis["kh"] = layer_model_mensink_bergen_regis.kh.where(
        layer_model_mensink_bergen_regis.kh != 0.0, np.nan
    )
    layer_model_mensink_bergen_regis["kv"] = layer_model_mensink_bergen_regis.kv.where(
        layer_model_mensink_bergen_regis.kv != 0.0, np.nan
    )
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


def get_top_from_ahn(
    ds,
    replace_surface_water_with_peil=True,
    replace_northsea_with_constant=None,
    method_elsewhere="nearest",
    cachedir=None,
):
    """
    Get top from AHN and fill the missing values with surface water levels or interpolation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the model grid.
    replace_surface_water_with_peil : bool, optional
        Replace missing values with peil. The default is True.
    replace_northsea_with_constant : float, optional
        Replace missing values with a constant. The default is None.
    method_elsewhere : str, optional
        Interpolation method. The default is "nearest".
    cachedir : str, optional
        Directory to cache the data. The default is None.

    Returns
    -------
    top : xarray.DataArray
        The top of the model grid.
    """
    if "ahn" not in ds:
        msg = "Dataset should contain the AHN data"
        raise ValueError(msg)

    top = ds["ahn"].copy()

    if replace_surface_water_with_peil:
        rws_ds = nlmod.read.rws.get_surface_water(
            ds, da_basename="rws_oppwater", cachedir=cachedir, cachename="rws_ds.nc"
        )
        fill_mask = np.logical_and(top.isnull(), np.isclose(rws_ds["rws_oppwater_area"], ds["area"]))
        top.values = xr.where(fill_mask, rws_ds["rws_oppwater_stage"], top)

    if replace_northsea_with_constant is not None:
        isnorthsea = nlmod.read.rws.get_northsea(ds, cachedir=cachedir, cachename="sea_ds.nc")["northsea"]
        fill_mask = np.logical_and(top.isnull(), isnorthsea)
        top.values = xr.where(fill_mask, replace_northsea_with_constant, top)

    # interpolate remainder
    points = list(
        zip(
            top.y.sel(icell2d=top.notnull()).values,
            top.x.sel(icell2d=top.notnull()).values,
            strict=False,
        )
    )
    values = top.sel(icell2d=top.notnull()).values
    qpoints = list(
        zip(
            top.y.sel(icell2d=top.isnull()).values,
            top.x.sel(icell2d=top.isnull()).values,
            strict=False,
        )
    )
    qvalues = griddata(points=points, values=values, xi=qpoints, method=method_elsewhere)
    top.loc[{"icell2d": top.isnull()}] = qvalues
    return top


def get_mensink_layer_model(ds_pwn_data, fix_min_layer_thickness=True):
    layer_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top"],
            "botm": get_mensink_botm(ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness),
            "kh": get_mensink_kh(ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness),
            "kv": get_mensink_kv(ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness),
        },
        coords={"layer": list(translate_triwaco_mensink_names_to_index.keys())},
        attrs={
            "extent": ds_pwn_data.attrs["extent"],
            "gridtype": ds_pwn_data.attrs["gridtype"],
        },
    )
    mask_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top_mask"],
            "botm": get_mensink_botm(
                ds_pwn_data,
                mask=True,
                transition=False,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_mensink_kh(
                ds_pwn_data,
                mask=True,
                transition=False,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kv": get_mensink_kv(
                ds_pwn_data,
                mask=True,
                transition=False,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
        },
        coords={"layer": list(translate_triwaco_mensink_names_to_index.keys())},
    )
    transition_model_mensink = xr.Dataset(
        {
            "top": ds_pwn_data["top_transition"],
            "botm": get_mensink_botm(
                ds_pwn_data,
                mask=False,
                transition=True,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_mensink_kh(
                ds_pwn_data,
                mask=False,
                transition=True,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kv": get_mensink_kv(
                ds_pwn_data,
                mask=False,
                transition=True,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
        },
        coords={"layer": list(translate_triwaco_mensink_names_to_index.keys())},
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


def get_bergen_layer_model(ds_pwn_data, fix_min_layer_thickness=True):
    layer_model_bergen = xr.Dataset(
        {
            "top": ds_pwn_data["top"],
            "botm": get_bergen_botm(ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness),
            "kh": get_bergen_kh(ds_pwn_data),
            "kv": get_bergen_kv(ds_pwn_data),
        },
        coords={"layer": list(translate_triwaco_bergen_names_to_index.keys())},
        attrs={
            "extent": ds_pwn_data.attrs["extent"],
            "gridtype": ds_pwn_data.attrs["gridtype"],
        },
    )
    mask_model_bergen = xr.Dataset(
        {
            "top": ds_pwn_data["top_mask"],
            "botm": get_bergen_botm(
                ds_pwn_data,
                mask=True,
                transition=False,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_bergen_kh(ds_pwn_data, mask=True, transition=False),
            "kv": get_bergen_kv(ds_pwn_data, mask=True, transition=False),
        },
        coords={"layer": list(translate_triwaco_bergen_names_to_index.keys())},
    )
    transition_model_bergen = xr.Dataset(
        {
            "top": ds_pwn_data["top_transition"],
            "botm": get_bergen_botm(
                ds_pwn_data,
                mask=False,
                transition=True,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_bergen_kh(ds_pwn_data, mask=False, transition=True),
            "kv": get_bergen_kv(ds_pwn_data, mask=False, transition=True),
        },
        coords={"layer": list(translate_triwaco_bergen_names_to_index.keys())},
    )

    for var in ["kh", "kv", "botm"]:
        layer_model_bergen[var] = layer_model_bergen[var].where(mask_model_bergen[var], np.nan)
        assert (layer_model_bergen[var].notnull() == mask_model_bergen[var]).all(), (
            f"There were nan values present in {var} in cells that should be valid"
        )
        assert ((mask_model_bergen[var] + transition_model_bergen[var]) <= 1).all(), (
            f"There should be no overlap between mask and transition of {var}"
        )

    return (
        layer_model_bergen,
        mask_model_bergen,
        transition_model_bergen,
    )


def get_bergen_thickness(data, mask=False, transition=False, fix_min_layer_thickness=True):
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
    botm = get_bergen_botm(
        data,
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )

    if mask:
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, other=np.nan)
        botm_nodata_isnan = botm.where(botm, other=np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, other=np.nan).where(_a, 1.0)
        botm_nodata_isnan = botm.where(~botm, other=np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data
        botm_nodata_isnan = botm

        def n(s):
            return s

    if "top" in data.data_vars:
        top_botm = xr.concat((a[n("top")].expand_dims(dim={"layer": ["mv"]}, axis=0), botm_nodata_isnan), dim="layer")
    else:
        top_botm = botm

    out = -top_botm.diff(dim="layer")

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_bergen_thickness(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    out = out.where(~np.isclose(out, 0.0), other=0.0)

    if (out < 0.0).any():
        logger.warning("Botm Bergen is not monotonically decreasing. Resulting in negative conductivity values.")
    return out


def get_bergen_kh(data, mask=False, anisotropy=5.0, transition=False):
    """
    Calculate the hydraulic conductivity (kh) based on the given data.

    Values may be applied everywhere. Use mask and/or thickness to determine
    where the values are valid.

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

    kh = xr.zeros_like(t_da)
    kh[0] = 7.0
    kh[1] = thickness[1] / clist[0] / f_anisotropy
    kh[2] = 7.0
    kh[3] = thickness[3] / clist[1] / f_anisotropy
    kh[4] = 12.0
    kh[5] = thickness[5] / clist[2] / f_anisotropy
    kh[6] = 15.0
    kh[7] = thickness[7] / clist[3] / f_anisotropy
    kh[8] = 20.0

    """
    if mask:
        # valid value if valid thickness and valid BER_C
        out = get_bergen_thickness(data, mask=True, transition=False).rename("kh").drop_vars("layer")
        out[{"layer": 1}] *= data["BER_C1A_mask"]
        out[{"layer": 3}] *= data["BER_C1B_mask"]
        out[{"layer": 5}] *= data["BER_C1C_mask"]
        out[{"layer": 7}] *= data["BER_C1D_mask"]
        out[{"layer": 9}] *= data["BER_C2_mask"]

    elif transition:
        # Valid value if valid thickness or valid BER_C
        out = get_bergen_thickness(data, mask=True, transition=False).rename("kh").drop_vars("layer")
        out[{"layer": 1}] |= data["BER_C1A_mask"]
        out[{"layer": 3}] |= data["BER_C1B_mask"]
        out[{"layer": 5}] |= data["BER_C1C_mask"]
        out[{"layer": 7}] |= data["BER_C1D_mask"]
        out[{"layer": 9}] |= data["BER_C2_mask"]

    else:
        thickness = get_bergen_thickness(data, mask=mask, transition=transition).drop_vars("layer")
        out = xr.ones_like(thickness).rename("kh")

        out[{"layer": [0, 2, 4, 6, 8]}] *= [[8.0], [7.0], [12.0], [15.0], [20.0]]
        out[{"layer": 1}] = thickness[{"layer": 1}] / data["BER_C1A"] * anisotropy
        out[{"layer": 3}] = thickness[{"layer": 3}] / data["BER_C1B"] * anisotropy
        out[{"layer": 5}] = thickness[{"layer": 5}] / data["BER_C1C"] * anisotropy
        out[{"layer": 7}] = thickness[{"layer": 7}] / data["BER_C1D"] * anisotropy
        out[{"layer": 9}] = thickness[{"layer": 9}] / data["BER_C2"] * anisotropy

    return out


def get_bergen_kv(data, mask=False, anisotropy=5.0, transition=False):
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

    Example:
        # Calculate hydraulic conductivity values without applying a mask
        kv_values = get_bergen_kv(data)

        # Calculate hydraulic conductivity values with a mask applied
        kv_values_masked = get_bergen_kv(data, mask=True)

        Note the f_anisotropy vs anisotropy
        # kv[0] = kh[0] * f_anisotropy
        # kv[1] = thickness[1] / clist[0]
        # kv[2] = kh[2] * f_anisotropy
        # kv[3] = thickness[3] / clist[1]
        # kv[4] = kh[4] * f_anisotropy
        # kv[5] = thickness[5] / clist[2]
        # kv[6] = kh[6] * f_anisotropy
        # kv[7] = thickness[7] / clist[3]
        # kv[8] = kh[8] * f_anisotropy
    """
    kh = get_bergen_kh(data, mask=mask, anisotropy=anisotropy, transition=transition)

    if not mask and not transition:
        # bool divided by float is float
        out = kh / anisotropy
    else:
        out = kh

    return out


def get_bergen_botm(data, mask=False, transition=False, fix_min_layer_thickness=True):
    """
    Calculate the bottom elevation of each layer in the Bergen model.

    Parameters
    ----------
    data (xarray.Dataset): Dataset containing the necessary variables.
    mask (bool, optional): If True, return a mask indicating the valid values. Default is False.

    Returns
    -------
    out (xarray.DataArray): Array containing the bottom elevation of each layer.
    """
    if mask:
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)

        def n(s):
            return f"BER_{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)

        def n(s):
            return f"BER_{s}_transition"

    else:
        a = data

        def n(s):
            return f"BER_{s}"

    out = xr.concat(
        (
            a[n("BA1A")] + a[n("DI1A")],  # Base aquifer 11
            a[n("BA1A")],  # Base aquitard 11
            a[n("BA1B")] + a[n("DI1B")],  # Base aquifer 12
            a[n("BA1B")],  # Base aquitard 12
            a[n("BA1C")] + a[n("DI1C")],  # Base aquifer 13
            a[n("BA1C")],  # Base aquitard 13
            a[n("BA1D")] + a[n("DI1D")],  # Base aquifer 14
            a[n("BA1D")],  # Base aquitard 14
            a[n("BAq2")] + a[n("DIq2")],  # Base aquifer 21
            a[n("BAq2")],  # Base aquitard 21
        ),
        dim="layer",
    ).transpose("layer", "icell2d")
    out.coords["layer"] = list(translate_triwaco_bergen_names_to_index.keys())

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_bergen_botm(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    if fix_min_layer_thickness:
        ds = xr.Dataset({"botm": out, "top": data["top"]})
        fix_missings_botms_and_min_layer_thickness(ds)
        out = ds["botm"]

    return out


def get_mensink_thickness(data, mask=False, transition=False, fix_min_layer_thickness=True):
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
        mask=mask,
        transition=transition,
        fix_min_layer_thickness=fix_min_layer_thickness,
    )

    if mask:
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, other=np.nan)
        botm_nodata_isnan = botm.where(botm, other=np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, other=np.nan).where(_a, 1.0)
        botm_nodata_isnan = botm.where(~botm, other=np.nan)

        def n(s):
            return f"{s}_transition"

    else:
        a = data
        botm_nodata_isnan = botm

        def n(s):
            return s

    if "top" in data.data_vars:
        top_botm = xr.concat((a[n("top")].expand_dims(dim={"layer": ["mv"]}, axis=0), botm_nodata_isnan), dim="layer")
    else:
        top_botm = botm

    out = -top_botm.diff(dim="layer")

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_mensink_thickness(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    out = out.where(~np.isclose(out, 0.0), other=0.0)

    if (out < 0.0).any():
        logger.warning("Botm is not monotonically decreasing.")
    return out


def get_mensink_kh(data, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
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
        dim="layer",
    )

    s12k = (
        a[n("s12kd")] * (a[n("ms12kd")] == 1)
        + 0.5 * a[n("s12kd")] * (a[n("ms12kd")] == 2)
        + 3 * a[n("s12kd")] * (a[n("ms12kd")] == 3)
    ) / b.isel(layer=3)
    s13k = a[n("s13kd")] * (a[n("ms13kd")] == 1) + 1.12 * a[n("s13kd")] * (a[n("ms13kd")] == 2) / b.isel(layer=5)
    s21k = a[n("s21kd")] * (a[n("ms21kd")] == 1) + a[n("s21kd")] * (a[n("ms21kd")] == 2) / b.isel(layer=7)
    s22k = 2 * a[n("s22kd")] * (a[n("ms22kd")] == 1) + a[n("s22kd")] * (a[n("ms22kd")] == 1) / b.isel(layer=9)

    out.loc[{"layer": 3}] = out.loc[{"layer": 3}].where(np.isnan(s12k), other=s12k)
    out.loc[{"layer": 5}] = out.loc[{"layer": 5}].where(np.isnan(s13k), other=s13k)
    out.loc[{"layer": 7}] = out.loc[{"layer": 7}].where(np.isnan(s21k), other=s21k)
    out.loc[{"layer": 9}] = out.loc[{"layer": 9}].where(np.isnan(s22k), other=s22k)

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_mensink_kh(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    return out


def get_mensink_kv(data, mask=False, anisotropy=5.0, transition=False, fix_min_layer_thickness=True):
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
        dim="layer",
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


def get_mensink_botm(data, mask=False, transition=False, fix_min_layer_thickness=True):
    """
    Calculate the bottom elevation of each layer in the model.

    Parameters
    ----------
    data (xarray.Dataset): Dataset containing the necessary variables.
    mask (bool, optional): If True, return a mask indicating the valid values. Default is False.

    Returns
    -------
    out (xarray.DataArray): Array containing the bottom elevation of each layer.
    """
    if mask:
        _a = data.copy()[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data.copy()[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)

        def n(s):
            return f"{s}_transition"

    else:
        a = data.copy()

        # for name in ["DS11", "DS12", "DS13", "DS21", "DS22", "DS31"]:
        #     a[name] = a[name].fillna(0.0)

        def n(s):
            return s

    out = xr.concat(
        (
            a[n("TS11")],  # Base aquifer 11
            a[n("TS11")] - a[n("DS11")],  # Base aquitard 11
            a[n("TS12")],  # Base aquifer 12
            a[n("TS12")] - a[n("DS12")],  # Base aquitard 12
            a[n("TS13")],  # Base aquifer 13
            a[n("TS13")] - a[n("DS13")],  # Base aquitard 13
            a[n("TS21")],  # Base aquifer 21
            a[n("TS21")] - a[n("DS21")],  # Base aquitard 21
            a[n("TS22")],  # Base aquifer 22
            a[n("TS22")] - a[n("DS22")],  # Base aquitard 22
            a[n("TS31")],  # Base aquifer 31
            a[n("TS31")] - a[n("DS31")],  # Base aquitard 31
            a[n("TS32")],  # Base aquifer 32
            a[n("TS32")] - 5.0,  # Base aquitard 33
            # a[n("TS32")] - 105., # Base aquifer 41
        ),
        dim="layer",
    ).transpose("layer", "icell2d")
    out.coords["layer"] = list(translate_triwaco_mensink_names_to_index.keys())

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_mensink_botm(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    if fix_min_layer_thickness:
        ds = xr.Dataset({"botm": out, "top": data["top"]})
        fix_missings_botms_and_min_layer_thickness(ds)
        out = ds["botm"]

    return out
