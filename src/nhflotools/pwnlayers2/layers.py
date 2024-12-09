import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

translate_triwaco_nhd_names_to_index = {
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


def get_nhd_layer_model(ds_pwn_data, fix_min_layer_thickness=True):
    layer_model_nhd = xr.Dataset(
        {
            "top": ds_pwn_data["top"],
            "botm": get_nhd_botm(ds_pwn_data, fix_min_layer_thickness=fix_min_layer_thickness),
            "kh": get_nhd_kh(ds_pwn_data),
            "kv": get_nhd_kv(ds_pwn_data),
        },
        coords={"layer": list(translate_triwaco_nhd_names_to_index.keys())},
        attrs={
            "extent": ds_pwn_data.attrs["extent"],
            "gridtype": ds_pwn_data.attrs["gridtype"],
        },
    )
    mask_model_nhd = xr.Dataset(
        {
            "top": ds_pwn_data["top_mask"],
            "botm": get_nhd_botm(
                ds_pwn_data,
                mask=True,
                transition=False,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_nhd_kh(ds_pwn_data, mask=True, transition=False),
            "kv": get_nhd_kv(ds_pwn_data, mask=True, transition=False),
        },
        coords={"layer": list(translate_triwaco_nhd_names_to_index.keys())},
    )
    transition_model_nhd = xr.Dataset(
        {
            "top": ds_pwn_data["top_transition"],
            "botm": get_nhd_botm(
                ds_pwn_data,
                mask=False,
                transition=True,
                fix_min_layer_thickness=fix_min_layer_thickness,
            ),
            "kh": get_nhd_kh(ds_pwn_data, mask=False, transition=True),
            "kv": get_nhd_kv(ds_pwn_data, mask=False, transition=True),
        },
        coords={"layer": list(translate_triwaco_nhd_names_to_index.keys())},
    )

    for var in ["kh", "kv", "botm"]:
        layer_model_nhd[var] = layer_model_nhd[var].where(mask_model_nhd[var], np.nan)
        assert (
            layer_model_nhd[var].notnull() == mask_model_nhd[var]
        ).all(), f"There were nan values present in {var} in cells that should be valid"
        assert (
            (mask_model_nhd[var] + transition_model_nhd[var]) <= 1
        ).all(), f"There should be no overlap between mask and transition of {var}"

    return (
        layer_model_nhd,
        mask_model_nhd,
        transition_model_nhd,
    )


def get_nhd_thickness(data, mask=False, transition=False, fix_min_layer_thickness=True):
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
    botm = get_nhd_botm(
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
        mask = get_nhd_thickness(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    out = out.where(~np.isclose(out, 0.0), other=0.0)

    if (out < 0.0).any():
        logger.warning("Botm nhd is not monotonically decreasing. Resulting in negative conductivity values.")
    return out


def get_nhd_kh(data, mask=False, anisotropy=5.0, transition=False):
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
        # valid value if valid thickness and valid NHD_C
        out = get_nhd_thickness(data, mask=True, transition=False).rename("kh").drop_vars("layer")
        out[{"layer": 1}] *= data["NHD_C1A_mask"]
        out[{"layer": 3}] *= data["NHD_C1B_mask"]
        out[{"layer": 5}] *= data["NHD_C1C_mask"]
        out[{"layer": 7}] *= data["NHD_C1D_mask"]
        out[{"layer": 9}] *= data["NHD_C2_mask"]

    elif transition:
        # Valid value if valid thickness or valid NHD_C
        out = get_nhd_thickness(data, mask=True, transition=False).rename("kh").drop_vars("layer")
        out[{"layer": 1}] |= data["NHD_C1A_mask"]
        out[{"layer": 3}] |= data["NHD_C1B_mask"]
        out[{"layer": 5}] |= data["NHD_C1C_mask"]
        out[{"layer": 7}] |= data["NHD_C1D_mask"]
        out[{"layer": 9}] |= data["NHD_C2_mask"]

    else:
        thickness = get_nhd_thickness(data, mask=mask, transition=transition).drop_vars("layer")
        out = xr.ones_like(thickness).rename("kh")

        out[{"layer": [0, 2, 4, 6, 8]}] *= [[8.0], [7.0], [12.0], [15.0], [20.0]]
        out[{"layer": 1}] = thickness[{"layer": 1}] / data["NHD_C1A"] * anisotropy
        out[{"layer": 3}] = thickness[{"layer": 3}] / data["NHD_C1B"] * anisotropy
        out[{"layer": 5}] = thickness[{"layer": 5}] / data["NHD_C1C"] * anisotropy
        out[{"layer": 7}] = thickness[{"layer": 7}] / data["NHD_C1D"] * anisotropy
        out[{"layer": 9}] = thickness[{"layer": 9}] / data["NHD_C2"] * anisotropy

    return out


def get_nhd_kv(data, mask=False, anisotropy=5.0, transition=False):
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
        kv_values = get_nhd_kv(data)

        # Calculate hydraulic conductivity values with a mask applied
        kv_values_masked = get_nhd_kv(data, mask=True)

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
    kh = get_nhd_kh(data, mask=mask, anisotropy=anisotropy, transition=transition)

    if not mask and not transition:
        # bool divided by float is float
        out = kh / anisotropy
    else:
        out = kh

    return out


def get_nhd_botm(data, mask=False, transition=False, fix_min_layer_thickness=True):
    """
    Calculate the bottom elevation of each layer in the nhd model.

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
            return f"NHD_{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)

        def n(s):
            return f"NHD_{s}_transition"

    else:
        a = data

        def n(s):
            return f"NHD_{s}"

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
    out.coords["layer"] = list(translate_triwaco_nhd_names_to_index.keys())

    if mask:
        return ~np.isnan(out)
    if transition:
        mask = get_nhd_botm(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    if fix_min_layer_thickness:
        ds = xr.Dataset({"botm": out, "top": data["top"]})
        _fix_missings_botms_and_min_layer_thickness(ds)
        out = ds["botm"]

    return out
