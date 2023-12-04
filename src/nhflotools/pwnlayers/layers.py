import logging
import os

import geopandas as gpd
import nlmod
import numpy as np
import xarray as xr
from flopy.utils.gridintersect import GridIntersect
from nlmod import cache
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def combine_two_layer_models(
    df_koppeltabel,
    layer_model_regis,
    layer_model_pwn,
    transition_model_pwn=None,
    koppeltabel_header_regis="Regis II v2.2",
    koppeltabel_header_pwn="ASSUMPTION1",
):
    """
    Combine the layer models of REGISII and PWN.

    The values of the PWN layer model are used where the layer_model_pwn is not nan.
    The values of the REGISII layer model are used where the layer_model_pwn is nan
    and transition_model_pwn is False. The remaining values are where the
    transition_model_pwn is True. Those values are linearly interpolated from the
    REGISII layer model to the PWN layer model.

    `layer_model_regis` and `layer_model_pwn` should have the same grid.
    The layer names of `layer_model_pwn` should be present in koppeltabel[`koppeltabel_header_pwn`].
    The layer names of `layer_model_regis` should be present in koppeltabel[`koppeltabel_header_regis`].
    To guarantee the coupling is always valid, the koppeltabel should be defined for all interlaying
    REGISII layers, this is not enforced.

    Note that the top variable is required in both layer models to be able to split
    and combine the top layer.

    Parameters
    ----------
    df_koppeltabel : pandas DataFrame
        DataFrame containing the koppeltabel. koppeltabel[`koppeltabel_header_pwn`]
        should contain the layer names of `layer_model_pwn` and
        koppeltabel[`koppeltabel_header_regis`] should contain the layer names of
        `layer_model_regis`.
    layer_model_regis : xarray Dataset
        Dataset containing the layer model of REGISII. It should contain the
        variables 'kh', 'kv', 'botm', and 'top'.
    layer_model_pwn : xarray Dataset
        Dataset containing the layer model of PWN. It should have nan values
        where the layer model is not defined. It should contain the variables
        'kh', 'kv', 'botm', and 'top'.
    transition_model_pwn : xarray Dataset, optional
        Dataset containing the transition model of PWN. It should contain
        the variables 'kh', 'kv', 'botm'. The default is None.
        It should be True where the transition between layer_model_regis and layer_model_pwn
        is defined and False where it is not. Where True, the values of are linearly interpolated
        from the REGISII layer model to the PWN layer model. If None, the transition is not used.
    koppeltabel_header_regis : str, optional
        Column name of the koppeltabel containing the REGISII layer names.
        The default is 'Regis II v2.2'.
    koppeltabel_header_pwn : str, optional
        Column name of the koppeltabel containing the PWN layer names.
        The default is 'ASSUMPTION1'.

    Returns
    -------
    layer_model_out : xarray Dataset
        Dataset containing the combined layer model.
    """
    assert (
        layer_model_regis.attrs["extent"] == layer_model_pwn.attrs["extent"]
    ), "Extent of layer models are not equal"
    assert (
        layer_model_regis.attrs["gridtype"] == layer_model_pwn.attrs["gridtype"]
    ), "Gridtype of layer models are not equal"
    assert (
        df_koppeltabel[koppeltabel_header_regis]
        .isin(layer_model_regis.layer.values)
        .all()
    ), (
        "Not all REGIS layers of the koppeltabel are in layer_model_regis. Make sure you set "
        "remove_nan_layers=False when refining the grid and in nlmod.to_model_ds()."
    )
    assert (
        df_koppeltabel[koppeltabel_header_pwn].isin(layer_model_pwn.layer.values).all()
    ), "Not all PWN layers of the koppeltabel are in layer_model_pwn"
    assert all(
        var in layer_model_regis.variables for var in ["kh", "kv", "botm", "top"]
    ), "Variable 'kh', 'kv', 'botm', or 'top' is missing in layer_model_regis"
    assert all(
        var in layer_model_pwn.variables for var in ["kh", "kv", "botm", "top"]
    ), "Variable 'kh', 'kv', 'botm', or 'top' is missing in layer_model_pwn"
    if transition_model_pwn is not None:
        assert all(
            var in transition_model_pwn.variables for var in ["kh", "kv", "botm"]
        ), "Variable 'kh', 'kv', or 'botm' is missing in transition_model_pwn"
        assert all(
            [
                np.issubdtype(dtype, bool)
                for dtype in transition_model_pwn.dtypes.values()
            ]
        ), "Variable 'kh', 'kv', and 'botm' in transition_model_pwn should be boolean"

    logger.info("Combining layer models")

    df_koppeltabel = df_koppeltabel.copy()

    # Only select part of the table that appears in the two layer models
    df_koppeltabel["layer_index"] = (
        df_koppeltabel.groupby(koppeltabel_header_regis).cumcount() + 1
    ).astype(str)
    df_koppeltabel["Regis_split"] = df_koppeltabel[koppeltabel_header_regis].str.cat(
        df_koppeltabel["layer_index"], sep="_"
    )

    # Leave out lower REGIS layers
    # nans halfway should not be allowed
    layer_model_tophalf = layer_model_regis.sel(
        layer=layer_model_regis.layer.isin(df_koppeltabel[koppeltabel_header_regis])
    )
    layer_model_bothalf = layer_model_regis.sel(
        layer=~layer_model_regis.layer.isin(df_koppeltabel[koppeltabel_header_regis])
    )

    # Count in how many layers the REGISII layers need to be split
    split_counts_regis = dict(
        zip(*np.unique(df_koppeltabel[koppeltabel_header_regis], return_counts=True))
    )

    # Count in how many layers the PWN layers need to be split
    split_counts_pwn = dict(
        zip(*np.unique(df_koppeltabel[koppeltabel_header_pwn], return_counts=True))
    )

    # Split both layer models with arbitrary thickness
    layer_model_top_split, _ = nlmod.layers.split_layers_ds(
        layer_model_tophalf, split_counts_regis, return_reindexer=True
    )
    layer_model_pwn_split, _ = nlmod.layers.split_layers_ds(
        layer_model_pwn, split_counts_pwn, return_reindexer=True
    )
    layer_model_pwn_split = layer_model_pwn_split.assign_coords(
        layer=df_koppeltabel["Regis_split"]
    )

    # extrapolate thickness of split layers
    thick_regis_top_split = nlmod.dims.layers.calculate_thickness(layer_model_top_split)
    thick_pwn_split = nlmod.dims.layers.calculate_thickness(layer_model_pwn_split)

    # Adjust the botm of the newly split REGIS layers
    logger.info("Adjusting the botm of the newly split REGIS layers")
    for name, group in df_koppeltabel.groupby(koppeltabel_header_regis):
        layers = group["Regis_split"].values

        # thick ratios of pwn layers that need to be extrapolated into the areas of the regis layers
        thick_ratio_pwn = thick_pwn_split.sel(layer=layers) / thick_pwn_split.sel(
            layer=layers
        ).sum(dim="layer")

        for layer in layers:
            mask = ~np.isnan(thick_ratio_pwn.sel(layer=layer))
            if mask.sum() == 0:
                continue

            griddata_points = list(
                zip(
                    thick_ratio_pwn.coords["x"].sel(icell2d=mask).values,
                    thick_ratio_pwn.coords["y"].sel(icell2d=mask).values,
                )
            )
            gridpoint_values = thick_ratio_pwn.sel(layer=layer, icell2d=mask).values
            qpoints = list(
                zip(
                    thick_ratio_pwn.coords["x"].sel(icell2d=~mask).values,
                    thick_ratio_pwn.coords["y"].sel(icell2d=~mask).values,
                )
            )
            qvalues = griddata(
                points=griddata_points,
                values=gridpoint_values,
                xi=qpoints,
                method="nearest",
            )

            thick_ratio_pwn.loc[{"layer": layer, "icell2d": ~mask}] = qvalues

        # convert thickness ratios to elevations
        elev = xr.concat(
            (
                thick_ratio_pwn * thick_pwn_split.sel(layer=layers).sum(dim="layer"),
                layer_model_regis["botm"].sel(layer=name),
            ),
            dim="layer",
        )
        top_botm_split = (
            elev.isel(layer=slice(None, None, -1))
            .cumsum(dim="layer")
            .isel(layer=slice(None, None, -1))
        )
        botm_split = top_botm_split.isel(layer=slice(1, None)).assign_coords(
            layer=layers
        )
        layer_model_top_split["botm"].loc[{"layer": layers}] = botm_split

    # Adjust the botm of the newly split PWN layers
    logger.info("Adjusting the botm of the newly split PWN layers")
    for name, group in df_koppeltabel.groupby(koppeltabel_header_pwn):
        if len(group) == 1:
            continue

        layers = group["Regis_split"].values

        # thick ratios of regis layers that need to be extrapolated into the areas of the pwn layers
        thick_ratio_regis = thick_regis_top_split.sel(
            layer=layers
        ) / thick_regis_top_split.sel(layer=layers).sum(dim="layer")

        for layer in layers:
            # Locate cells where REGIS ratios are used
            mask = np.isnan(thick_ratio_regis.sel(layer=layer))
            if mask.sum() == 0:
                continue

            griddata_points = list(
                zip(
                    thick_ratio_regis.coords["x"].sel(icell2d=mask).values,
                    thick_ratio_regis.coords["y"].sel(icell2d=mask).values,
                )
            )
            gridpoint_values = thick_ratio_regis.sel(layer=layer, icell2d=mask).values
            qpoints = list(
                zip(
                    thick_ratio_regis.coords["x"].sel(icell2d=~mask).values,
                    thick_ratio_regis.coords["y"].sel(icell2d=~mask).values,
                )
            )
            qvalues = griddata(
                points=griddata_points,
                values=gridpoint_values,
                xi=qpoints,
                method="nearest",
            )

            thick_ratio_regis.loc[{"layer": layer, "icell2d": ~mask}] = qvalues

        # convert thickness ratios to elevations
        botm = layer_model_pwn["botm"].sel(layer=name)
        elev = xr.concat(
            (
                thick_ratio_regis
                * thick_regis_top_split.sel(layer=layers).sum(dim="layer"),
                botm,
            ),
            dim="layer",
        )
        top_botm_split = (
            elev.isel(layer=slice(None, None, -1))
            .cumsum(dim="layer")
            .isel(layer=slice(None, None, -1))
        )
        botm_split = top_botm_split.isel(layer=slice(1, None)).assign_coords(
            layer=layers
        )
        layer_model_pwn_split["botm"].loc[{"layer": layers}] = botm_split

    # Merge the two layer models
    logger.info("Merging the two layer models")
    layer_model_top = xr.Dataset(
        {
            "botm": xr.where(
                np.isnan(layer_model_pwn_split["botm"]),
                layer_model_top_split["botm"],
                layer_model_pwn_split["botm"],
            ),
            "kh": xr.where(
                np.isnan(layer_model_pwn_split["kh"]),
                layer_model_top_split["kh"],
                layer_model_pwn_split["kh"],
            ),
            "kv": xr.where(
                np.isnan(layer_model_pwn_split["kv"]),
                layer_model_top_split["kv"],
                layer_model_pwn_split["kv"],
            ),
        },
        attrs={
            "extent": layer_model_regis.attrs["extent"],
            "gridtype": layer_model_regis.attrs["gridtype"],
        },
    )

    # introduce transition of layers
    if transition_model_pwn is not None:
        logger.info(
            "Linear interpolation of transition region inbetween the two layer models"
        )
        transition_model_pwn_split = transition_model_pwn.sel(
            layer=df_koppeltabel[koppeltabel_header_pwn].values
        ).assign_coords(layer=df_koppeltabel["Regis_split"].values)

        for key in ["botm", "kh", "kv"]:
            var = layer_model_top[key]
            trans = transition_model_pwn_split[key]

            for layer in var.layer.values:
                vari = var.sel(layer=layer)
                transi = trans.sel(layer=layer)
                if transi.sum() == 0:
                    continue

                griddata_points = list(
                    zip(
                        vari.coords["x"].sel(icell2d=~transi).values,
                        vari.coords["y"].sel(icell2d=~transi).values,
                    )
                )
                gridpoint_values = vari.sel(icell2d=~transi).values
                qpoints = list(
                    zip(
                        vari.coords["x"].sel(icell2d=transi).values,
                        vari.coords["y"].sel(icell2d=transi).values,
                    )
                )
                qvalues = griddata(
                    points=griddata_points,
                    values=gridpoint_values,
                    xi=qpoints,
                    method="linear",
                )

                var.loc[{"layer": layer, "icell2d": transi}] = qvalues
    else:
        logger.info("No transition of the two layer models")

    layer_model_out = xr.concat((layer_model_top, layer_model_bothalf), dim="layer")
    layer_model_out["top"] = layer_model_regis["top"]

    # categorize layers
    # 1: regis
    # 2: pwn
    # 3: transition
    cat_top = xr.zeros_like(layer_model_top[["botm", "kh", "kv"]], dtype=int)
    cat_top = cat_top.where(
        cond=~np.isnan(layer_model_pwn_split[["botm", "kh", "kv"]]), other=1
    )
    cat_top = cat_top.where(
        cond=np.isnan(layer_model_pwn_split[["botm", "kh", "kv"]]), other=2
    )
    cat_top = cat_top.where(
        cond=~transition_model_pwn_split[["botm", "kh", "kv"]], other=3
    )

    cat_botm = xr.ones_like(layer_model_bothalf[["botm", "kh", "kv"]], dtype=int)
    cat = xr.concat((cat_top, cat_botm), dim="layer")

    return layer_model_out, cat


def get_thickness(data, mask=False, transition=False):
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
    botm = get_botm(data, mask=mask, transition=transition)

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
        top_botm = xr.concat((a[n("top")], botm_nodata_isnan), dim="layer")
    else:
        top_botm = botm

    out = top_botm.diff(dim="layer")

    if mask:
        return ~np.isnan(out)
    elif transition:
        mask = get_thickness(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), (
            "Transition cells should not overlap with mask."
        )
        return transition
    else:
        return out


def get_kh(data, mask=False, anisotropy=5.0, transition=False):
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

    thickness = get_thickness(data, mask=mask, transition=transition)

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
    s13k = a[n("s13kd")] * (a[n("ms13kd")] == 1) + 1.12 * a[n("s13kd")] * (
        a[n("ms13kd")] == 2
    ) / b.isel(layer=5)
    s21k = a[n("s21kd")] * (a[n("ms21kd")] == 1) + a[n("s21kd")] * (
        a[n("ms21kd")] == 2
    ) / b.isel(layer=7)
    s22k = 2 * a[n("s22kd")] * (a[n("ms22kd")] == 1) + a[n("s22kd")] * (
        a[n("ms22kd")] == 1
    ) / b.isel(layer=9)

    out.loc[{"layer": 3}] = out.loc[{"layer": 3}].where(np.isnan(s12k), other=s12k)
    out.loc[{"layer": 5}] = out.loc[{"layer": 5}].where(np.isnan(s13k), other=s13k)
    out.loc[{"layer": 7}] = out.loc[{"layer": 7}].where(np.isnan(s21k), other=s21k)
    out.loc[{"layer": 9}] = out.loc[{"layer": 9}].where(np.isnan(s22k), other=s22k)

    if mask:
        return ~np.isnan(out)
    elif transition:
        mask = get_kh(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), (
            "Transition cells should not overlap with mask."
        )
        return transition
    else:
        return out


def get_kv(data, mask=False, anisotropy=5.0, transition=False):
    """
    Calculate the hydraulic conductivity (KV) for different aquifers and aquitards.

    Parameters:
        data (xarray.Dataset): Dataset containing the necessary variables for calculation.
        mask (bool, optional): Flag indicating whether to apply a mask to the data. Defaults to False.
        anisotropy (float, optional): Anisotropy factor for adjusting the hydraulic conductivity. Defaults to 5.0.

    Returns:
        xarray.DataArray: Array containing the calculated hydraulic conductivity values for each layer.

    Notes:
        - The function expects the input dataset to contain the following variables:
            - KW11, KW12, KW13, KW21, KW22, KW31, KW32: Hydraulic conductivity values for aquifers.
            - C11AREA, C12AREA, C13AREA, C21AREA, C22AREA, C31AREA, C32AREA: Areas of aquitards corresponding to each aquifer.
        - The function also requires the `get_thickness` function to be defined and accessible.

    Example:
        # Calculate hydraulic conductivity values without applying a mask
        kv_values = get_kv(data)

        # Calculate hydraulic conductivity values with a mask applied
        kv_values_masked = get_kv(data, mask=True)
    """
    thickness = get_thickness(data, mask=mask, transition=transition)

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
    elif transition:
        mask = get_kv(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), (
            "Transition cells should not overlap with mask."
        )
        return transition
    else:
        return out


def get_botm(data, mask=False, transition=False):
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
        _a = data[[var for var in data.variables if var.endswith("_mask")]]
        a = _a.where(_a, np.nan)

        def n(s):
            return f"{s}_mask"

    elif transition:
        # note the ~ operator
        _a = data[[var for var in data.variables if var.endswith("_transition")]]
        a = _a.where(~_a, np.nan).where(_a, 1.0)

        def n(s):
            return f"{s}_transition"

    else:
        a = data

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
    )
    if mask:
        return ~np.isnan(out)
    elif transition:
        mask = get_botm(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), (
            "Transition cells should not overlap with mask."
        )
        return transition
    else:
        return out


def get_pwn_layer_model(ds, data_path, length_transition=100.0, cachedir=None):
    pwn = read_pwn_data2(
        ds,
        datadir=data_path,
        length_transition=length_transition,
        cachedir=cachedir,
    )
    translate_triwaco_names_to_index = {
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

    layer_model_pwn_attrs = {
        "extent": ds.attrs["extent"],
        "gridtype": ds.attrs["gridtype"],
    }
    layer_model_pwn = xr.Dataset(
        {
            "top": pwn["top"],
            "botm": get_botm(pwn),
            "kh": get_kh(pwn),
            "kv": get_kv(pwn),
        },
        coords={"layer": list(translate_triwaco_names_to_index.keys())},
        attrs=layer_model_pwn_attrs,
    )
    transition_model_pwn = xr.Dataset(
        {
            "top": pwn["top_transition"],
            "botm": get_botm(pwn, mask=False, transition=True),
            "kh": get_kh(pwn, mask=False, transition=True),
            "kv": get_kv(pwn, mask=False, transition=True),
        },
        coords={"layer": list(translate_triwaco_names_to_index.keys())},
    )
    return layer_model_pwn, transition_model_pwn


def read_pwn_data2(ds, datadir=None, length_transition=100.0, cachedir=None):
    """reads model data from a directory


    Parameters
    ----------
    ds : xarray Dataset
        model dataset.
    datadir : str, optional
        directory with modeldata. The default is None.
    cachedir : str, optional
        cachedir used to cache files using the decorator
        nlmod.cache.cache_netcdf. The default is None.

    Returns
    -------
    ds : xarray Dataset
        model dataset.
    ds_mask : xarray Dataset
        mask dataset. True where values are valid
    ds_mask_transition : xarray Dataset
        mask dataset. True in transition zone.
    """

    modelgrid = nlmod.dims.grid.modelgrid_from_ds(ds)
    ix = GridIntersect(modelgrid, method="vertex")

    intersect_kwargs = dict()

    ds_out = xr.Dataset()

    functions = [
        _read_top_of_aquitards,
        _read_thickness_of_aquitards,
        _read_kd_of_aquitards,
        _read_mask_of_aquifers,
        _read_layer_kh,
        _read_kv_area,
    ]

    for func in functions:
        logger.info(f"Gathering PWN layer info with: {func.__name__}")

        out = func(
            ds,
            datadir,
            length_transition=length_transition,
            cachedir=cachedir,
            cachename=f"triw_{func.__name__}",
            ix=ix,
            **intersect_kwargs,
        )
        ds_out.update(out)

    # Add top from ds
    ds_out["top"] = ds["top"]
    ds_out["top_mask"] = xr.ones_like(ds["top"], dtype=bool)
    ds_out["top_transition"] = xr.zeros_like(ds["top"], dtype=bool)

    return ds_out


@cache.cache_netcdf
def _read_top_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """read top of aquitards


    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'TS11', 'TS12', 'TS13', 'TS21', 'TS22', 'TS31',
        'TS32' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read top of aquitards")

    ds_out = {}

    import dask
    _gdf_to_da = dask.delayed(nlmod.dims.grid.gdf_to_da)

    for name in ["TS11", "TS12", "TS13", "TS21", "TS22", "TS31", "TS32"]:
        fname = os.path.join(
            pathname, "laagopbouw", "Top_aquitard", "{}.shp".format(name)
        )
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        delayed = _gdf_to_da(
            gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix
        )
        ds_out[name] = dask.array.from_delayed(delayed, shape=ds.top.shape, dtype=float)
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(
            gdf, ds, ix=ix, buffer=length_transition
        )
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]
    return ds_out


@cache.cache_netcdf
def _read_thickness_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """read thickness of aquitards

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'DS11', 'DS12', 'DS13', 'DS21', 'DS22', 'DS31'
        variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read thickness of aquitards")

    ds_out = xr.Dataset()

    # read thickness of aquitards
    for name in ["DS11", "DS12", "DS13", "DS21", "DS22", "DS31"]:
        fname = os.path.join(
            pathname, "laagopbouw", "Dikte_aquitard", "{}.shp".format(name)
        )

        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(
            gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix
        )
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(
            gdf, ds, ix=ix, buffer=length_transition
        )
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf
def _read_kd_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """read kd of aquitards

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 's11kd', 's12kd', 's13kd', 's21kd', 's22kd', 's31kd',
        's32kd' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read kd of aquifers")

    ds_out = xr.Dataset()

    # read kD-waarden of aquifers
    for name in ["s11kd", "s12kd", "s13kd", "s21kd", "s22kd", "s31kd", "s32kd"]:
        fname = os.path.join(
            pathname, "Bodemparams", "KDwaarden_aquitards", "{}.shp".format(name)
        )
        gdf = gpd.read_file(fname)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(
            gdf, ds, column="VALUE", agg_method="nearest"
        )
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(
            gdf, ds, ix=ix, buffer=length_transition
        )
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf
def _read_mask_of_aquifers(ds, pathname, length_transition=100.0, ix=None):
    """read mask of aquifers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with '12', '13', '21', '22' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.
    """
    logging.info("read mask of aquifers")

    ds_out = xr.Dataset()

    # read masks of auifers
    for name in ["12", "13", "21", "22"]:
        key = f"ms{name}kd"
        fname = os.path.join(
            pathname,
            "Bodemparams",
            "Maskers_kdwaarden_aquitards",
            "masker_aquitard{}_kd.shp".format(name),
        )
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        ds_out[key] = nlmod.dims.gdf_to_da(
            gdf, ds, column="VALUE", agg_method="nearest", ix=ix
        )
        ds_out[f"{key}_mask"] = ~np.isnan(ds_out[key])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(
            gdf, ds, ix=ix, buffer=length_transition
        )
        ds_out[f"{key}_transition"] = in_transition & ~ds_out[f"{key}_mask"]

    return ds_out


@cache.cache_netcdf
def _read_layer_kh(ds, pathname, length_transition=100.0, ix=None):
    """read hydraulic conductivity of layers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'KW11', 'KW12', 'KW13', 'KW21', 'KW22', 'KW31',
        'KW32' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read hydraulic conductivity of layers")

    ds_out = xr.Dataset()

    # read hydraulic conductivity of layers
    for name in ["KW11", "KW12", "KW13", "KW21", "KW22", "KW31", "KW32"]:
        fname = os.path.join(
            pathname, "Bodemparams", "Kwaarden_aquifers", "{}.shp".format(name)
        )
        gdf = gpd.read_file(fname)
        ds_out[name] = nlmod.dims.gdf_to_da(
            gdf, ds, column="VALUE", agg_method="area_weighted"
        )
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(
            gdf, ds, ix=ix, buffer=length_transition
        )
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf
def _read_kv_area(ds, pathname, length_transition=100.0, ix=None):
    """read vertical resistance of layers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'C11AREA', 'C12AREA', 'C13AREA', 'C21AREA',
        'C22AREA', 'C31AREA', 'C32AREA' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.
    """
    logging.info("read vertical resistance of layers")

    ds_out = xr.Dataset()

    # read vertical resistance per area
    for name in [
        "C11AREA",
        "C12AREA",
        "C13AREA",
        "C21AREA",
        "C22AREA",
        "C31AREA",
        "C32AREA",
    ]:
        fname = os.path.join(
            pathname, "Bodemparams", "Cwaarden_aquitards", "{}.shp".format(name)
        )
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)

        # some overlying shapes give different results when aggregated with
        # nearest. Remove underlying shape to get same results as Triwaco
        if name == "C13AREA":
            gdf2 = gdf.copy()
            for i in [7, 8, 12, 13]:
                gdf2.geometry = [
                    geom.difference(gdf.loc[i, "geometry"])
                    for geom in gdf.geometry.values
                ]
                gdf2.loc[i, "geometry"] = gdf.loc[i, "geometry"]
            gdf = gdf2
        elif name == "C21AREA":
            geom_1 = gdf.loc[1].geometry.difference(gdf.loc[4].geometry)
            gdf.loc[1, "geometry"] = geom_1
        elif name == "C22AREA":
            gdf2 = gdf.copy()
            for i in [6, 8, 9]:
                gdf2.geometry = [
                    geom.difference(gdf.loc[i, "geometry"])
                    for geom in gdf.geometry.values
                ]
                gdf2.loc[i, "geometry"] = gdf.loc[i, "geometry"]
            gdf = gdf2

        ds_out[name] = nlmod.dims.gdf_to_da(
            gdf, ds, column="VALUE", agg_method="nearest", ix=ix
        )

        nanmask = np.isnan(ds_out[name])
        if name == "C11AREA":
            ds_out[name].values[nanmask] = 1.0
        else:
            ds_out[name].values[nanmask] = 10.0

        ds_out[f"{name}_mask"] = xr.ones_like(ds_out[name], dtype=bool)
        ds_out[f"{name}_transition"] = xr.zeros_like(ds_out[name], dtype=bool)

    return ds_out


@cache.cache_netcdf
def _read_topsysteem(ds, pathname):
    """read topsysteem

    Not tested yet after delivered by Artesia

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'mvpolder', 'MVdtm', 'mvDTM', 'gempeil',
        'TOP', 'codesoort', 'draindiepte' variables.

    """
    logging.info("read topsysteem")

    ds_out = xr.Dataset()

    # read surface level
    fname = os.path.join(pathname, "Topsyst", "mvpolder2007.shp")
    gdf = gpd.read_file(fname)
    ds_out["mvpolder"] = nlmod.dims.gdf_to_da(
        gdf, ds, column="VALUE", agg_method="nearest"
    )

    fname = os.path.join(pathname, "Topsyst", "MVdtm2007.shp")
    gdf = gpd.read_file(fname)
    ds_out["MVdtm"] = nlmod.dims.gdf_to_da(
        gdf, ds, column="VALUE", agg_method="nearest"
    )
    ds_out["mvDTM"] = ds_out["MVdtm"]  # both ways are used in expressions

    fname = os.path.join(pathname, "Topsyst", "gem_polderpeil2007.shp")
    gdf = gpd.read_file(fname)
    gdf.geometry = gdf.buffer(0)
    ds_out["gempeil"] = nlmod.dims.gdf_to_da(
        gdf, ds, column="VALUE", agg_method="area_weighted"
    )

    # determine the top of the groundwater system
    top = ds_out["gempeil"].copy()
    # use nearest interpolation to fill gaps
    top = nlmod.dims.fillnan_da(top, ds=ds, method="nearest")
    ds_out["TOP"] = top

    fname = os.path.join(pathname, "Topsyst", "codes_voor_typedrainage.shp")
    gdf = gpd.read_file(fname)
    gdf.geometry = gdf.buffer(0)
    ds_out["codesoort"] = nlmod.dims.gdf_to_da(
        gdf, ds, column="VALUE", agg_method="nearest"
    )

    fname = os.path.join(pathname, "Topsyst", "diepte_landbouw_drains.shp")
    gdf = gpd.read_file(fname)
    ds_out["draindiepte"] = nlmod.dims.gdf_to_da(
        gdf, ds, column="VALUE", agg_method="nearest"
    )

    return ds_out
