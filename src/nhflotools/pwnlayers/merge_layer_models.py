import logging

import nlmod
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from nhflotools.pwnlayers.utils import fix_missings_botms_and_min_layer_thickness

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


def combine_two_layer_models(
    layer_model_regis,
    layer_model_other,
    mask_model_other,
    transition_model,
    top,
    df_koppeltabel,
    koppeltabel_header_regis="Regis II v2.2",
    koppeltabel_header_other="ASSUMPTION1",
    transition_method="linear",
):
    """
    Combine the layer models of REGISII and OTHER.

    The values of the OTHER layer model are used where the mask_model_other is True. The
    transition zone is a buffer around layer_model_other to create a smooth transition
    between layer_model_regis and layer_model_other, and is configured by setting
    transition_model is True.

    The values of the REGISII layer model are used where the layer_model_other is nan
    and transition_model is False. The remaining values are where the
    transition_model is True. Those values are linearly interpolated from the
    REGISII layer model to the OTHER layer model.

    `layer_model_regis` and `layer_model_other` should have the same grid.
    The layer names of `layer_model_other` should be present in koppeltabel[`koppeltabel_header_other`].
    The layer names of `layer_model_regis` should be present in koppeltabel[`koppeltabel_header_regis`].
    To guarantee the coupling is always valid, the koppeltabel should be defined for all interlaying
    REGISII layers, this is not enforced.

    Note that the top variable is required in both layer models to be able to split
    and combine the top layer.

    TODO: |
        - Refactor part of _interpolate_ds to _interpolate_da, with method linear or nearest
        - Convert ratio_split to botm where cat == 1 or where cat == 2
        - Interpolate botm where cat == 3
        - Replace .assign_coords(layer=dfk["Regis_split"]) with an additional coordinate added to
        - Add tests for _interpolate_ds
        - Add tests for _interpolate_da


    Parameters
    ----------
    layer_model_regis : xarray Dataset
        Dataset containing the layer model of REGISII. It should contain the
        variables 'kh', 'kv', 'botm'.
    layer_model_other : xarray Dataset
        Dataset containing the layer model of OTHER on the same grid as layer_model_regis.
        It should contain the variables 'kh', 'kv', 'botm'.
    mask_model_other : xarray Dataset
        Dataset containing the mask of the OTHER layer model. It should contain the variables
        'kh', 'kv', and 'botm'. It should be True where the layer_model_other is defined and False
        where layer_model_regis should be used.
    transition_model : xarray Dataset
        Dataset containing the transition model of OTHER on the same grid as layer_model_regis.
        It should contain the variables 'kh', 'kv', 'botm'.
        It should be True where the transition between layer_model_regis and layer_model_other
        is defined and False where it is not. Where True, the values of are linearly interpolated
        from the REGISII layer model to the OTHER layer model.
    top : xarray DataArray
        DataArray containing the top of the layers.
    df_koppeltabel : pandas DataFrame
        DataFrame containing the koppeltabel. koppeltabel[`koppeltabel_header_other`]
        should contain the layer names of `layer_model_other` and
        koppeltabel[`koppeltabel_header_regis`] should contain the layer names of
        `layer_model_regis`.
    koppeltabel_header_regis : str, optional
        Column name of the koppeltabel containing the REGISII layer names.
        The default is 'Regis II v2.2'.
    koppeltabel_header_other : str, optional
        Column name of the koppeltabel containing the OTHER layer names.
        The default is 'ASSUMPTION1'.
    transition_method : {'linear', 'keep_ratios'}, optional
        Method to use for the transition zone. If 'linear', the values are linearly interpolated
        from the REGISII layer model to the OTHER layer model. If 'keep_ratios', the ratios of the
        thickness of the layers in the OTHER layer model to the thickness of the layers in the
        REGISII layer model are used to compute the botm of the newly split layers.

    Returns
    -------
    out : xarray Dataset
        Dataset containing the combined layer model with kh, kv, and botm.
    cat : xarray Dataset
        Dataset containing the category of the layers. The values are:
        1: REGISII layer
        2: OTHER layer
        3: transition zone

    """
    logger.info("Combining two layer models")

    # Validate input datasets
    _validate_inputs(
        layer_model_regis,
        layer_model_other,
        mask_model_other,
        transition_model,
        top,
        df_koppeltabel,
        koppeltabel_header_regis,
        koppeltabel_header_other,
    )
    layer_model_regis = layer_model_regis.copy()
    layer_model_other = layer_model_other.copy()
    layer_model_regis["top"] = top.copy()
    layer_model_other["top"] = top.copy()

    dfk = df_koppeltabel.copy()
    dfk_upper = dfk[~dfk[koppeltabel_header_other].isna()]
    dfk_lower = dfk[dfk[koppeltabel_header_other].isna()]

    # Fix minimum layer thickness in REGIS and OTHER. Still required to fix transition zone.
    fix_missings_botms_and_min_layer_thickness(layer_model_regis)
    fix_missings_botms_and_min_layer_thickness(layer_model_other)

    # Apply mask to other layer model
    for var in ["kh", "kv", "botm"]:
        layer_model_other[var] = layer_model_other[var].where(mask_model_other[var], np.nan)
        assert (layer_model_other[var].notnull() == mask_model_other[var]).all(), (
            f"There were nan values present in {var} in cells that should be valid"
        )

    # Basename can occur multiple times if previously combined
    basenames_regis = [layer.split("_")[0] for layer in layer_model_regis.layer.values]

    """All the layers that are being coupled"""
    # Only select part of the table that appears in the two layer models
    dfk_mask = dfk_upper[koppeltabel_header_regis].isin(basenames_regis)
    dfk_upper = dfk_upper[dfk_mask]

    # Split both layer models with evenly-split thickness
    split_dict_regis = dfk_upper.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis].count().to_dict()
    split_dict_other = dfk_upper.groupby(koppeltabel_header_other, sort=False)[koppeltabel_header_other].count().to_dict()

    layer_model_regis_split = nlmod.dims.layers.split_layers_ds(
        ds=layer_model_regis.sel(layer=list(split_dict_regis.keys())),
        split_dict=split_dict_regis,
    )
    layer_model_other_split = nlmod.dims.layers.split_layers_ds(
        ds=layer_model_other.sel(layer=list(split_dict_other.keys())),
        split_dict=split_dict_other,
    )
    layer_names = layer_model_regis_split.layer.values
    layer_model_other_split = layer_model_other_split.assign_coords(layer=layer_names)

    mask_model_other_split = mask_model_other.sel(layer=dfk_upper[koppeltabel_header_other].values).assign_coords(
        layer=layer_names
    )
    transition_model_split = transition_model.sel(layer=dfk_upper[koppeltabel_header_other].values).assign_coords(
        layer=layer_names
    )

    # Categorize layers
    # 1: regis
    # 2: other
    # 3: transition
    cat = xr.ones_like(layer_model_regis_split[["kh", "kv", "botm"]], dtype=int)
    cat = cat.where(~mask_model_other_split[["kh", "kv", "botm"]], other=2)
    cat = cat.where(~transition_model_split[["kh", "kv", "botm"]], other=3)

    # Prepare out
    # 1. layer_model_regis_split
    # 2. layer_model_other_split
    # 3. interpolation of transition zone
    out = xr.where(
        cat == 1,
        layer_model_regis_split[["kh", "kv", "botm"]].copy(),
        layer_model_other_split[["kh", "kv", "botm"]].copy(),  # has nan's in the transition zone
    )

    # Add top and remaining metadata
    out["top"] = top

    if transition_method == "linear":
        # Linearly interpolate transition zone inplace
        _interpolate_ds(out, isvalid=cat != 3, ismissing=cat == 3, method="linear")

    elif transition_method == "keep_ratios":
        raise NotImplementedError(
            "The 'keep_ratios' transition method is not implemented yet. Please use 'linear' instead."
        )
        # cat.botm.isel(layer=slice(4), icell2d=slice(4))
        thick_split = nlmod.dims.layers.calculate_thickness(out)
        thick_split.coords[koppeltabel_header_regis] = ("layer", dfk_upper[koppeltabel_header_regis])
        thick_split.coords[koppeltabel_header_other] = ("layer", dfk_upper[koppeltabel_header_other])

        assert np.isnan(thick_split).sum() == 0, (
            "There are still nan values in the thickness of the layers. Otherwise, fill up with zeros or use skipna arguments in sum/cumsum calls."
        )

        # Compute the ratios of split regis layers in other model
        thick_split_sum_other = (
            thick_split.groupby(koppeltabel_header_regis)
            .sum()
            .sel(**{koppeltabel_header_regis: dfk_upper[koppeltabel_header_regis].values})
            .rename({koppeltabel_header_regis: "layer"})
            .assign_coords(layer=layer_names)
        )
        thick_split_sum_regis = (
            thick_split.groupby(koppeltabel_header_other)
            .sum()
            .sel(**{koppeltabel_header_other: dfk_upper[koppeltabel_header_other].values})
            .rename({koppeltabel_header_other: "layer"})
            .assign_coords(layer=layer_names)
        )
        thick_split_cumsum_other = (
            thick_split.isel(layer=slice(None, None, -1))
            .groupby(koppeltabel_header_regis)
            .cumsum()
            .isel(layer=slice(None, None, -1))
        )
        thick_split_cumsum_regis = (
            thick_split.isel(layer=slice(None, None, -1))
            .groupby(koppeltabel_header_other)
            .cumsum()
            .isel(layer=slice(None, None, -1))
        )

        cum_ratio_other = xr.where(
            thick_split_sum_other != 0.0, (thick_split_cumsum_other - thick_split) / thick_split_sum_other, 0.0
        )  # Use this to assign the botms your newly split REGIS layers
        cum_ratio_regis = xr.where(
            thick_split_sum_regis != 0.0, (thick_split_cumsum_regis - thick_split) / thick_split_sum_regis, 0.0
        )  # Use this to assign the botms your newly split OTHER layers

        cum_ratio = xr.zeros_like(out["botm"], dtype=float)
        cum_ratio = xr.where(cat.botm == 1, cum_ratio_regis, cum_ratio)
        cum_ratio = xr.where(cat.botm == 2, cum_ratio_other, cum_ratio)
    else:
        msg = f"Unknown transition method: {transition_method}. Please use 'linear' or 'keep_ratios'."
        raise ValueError(msg)

    """All the layers that are not being coupled are added to the out dataset"""
    # Add the layers that are not being coupled
    layer_model_regis_not_coupled = layer_model_regis.sel(
        layer=dfk_lower[koppeltabel_header_regis].values
    )
    out_upper_lower = xr.concat(
        [
            out,
            layer_model_regis_not_coupled[["kh", "kv", "botm"]].assign_coords(
                layer=layer_model_regis_not_coupled.layer.values
            ),
        ],
        dim="layer",
    )
    cat_upper_lower = xr.concat(
        [
            cat,
            xr.ones_like(layer_model_regis_not_coupled[["kh", "kv", "botm"]], dtype=int).assign_coords(
                layer=layer_model_regis_not_coupled.layer.values
            ),
        ],
        dim="layer",
    )

    return out_upper_lower, cat_upper_lower


def _interpolate_ds(ds, isvalid, ismissing, method="linear"):
    """
    Interpolate the values of the dataset inplace where the mask is True.

    The values are interpolated from the values where the mask is False.
    The interpolation is done using the griddata function from scipy.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the values to be interpolated.
    isvalid : xarray Dataset
        Dataset containing the mask of the values to be interpolated.
    ismissing : xarray Dataset
        Dataset containing the mask of the values to be interpolated.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of

        ``nearest``
          return the value at the data point closest to
          the point of interpolation.

        ``linear``
          tessellate the input point set to N-D
          simplices, and interpolate linearly on each simplex.
    """
    for k in ds.keys():
        if "layer" not in ds[k].dims:
            continue

        for layer in ds[k].layer.values:
            _interpolate_da(
                ds[k].sel(layer=layer),
                isvalid[k].sel(layer=layer),
                ismissing[k].sel(layer=layer),
                method=method,
            )


def _interpolate_da(da, isvalid, ismissing, method="linear"):
    """
    Interpolate the values of the DataArray inplace where the mask is True.

    The values are interpolated from the values where the mask is False.
    The interpolation is done using the griddata function from scipy.

    Parameters
    ----------
    da : xarray DataArray
        DataArray containing the values to be interpolated.
    isvalid : xarray DataArray
        DataArray containing the mask of the values to be interpolated.
    ismissing : xarray DataArray
        DataArray containing the mask of the values to be interpolated.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of

        ``nearest``
          return the value at the data point closest to
          the point of interpolation.

        ``linear``
          tessellate the input point set to N-D
          simplices, and interpolate linearly on each simplex.
    """
    if ismissing.sum() == 0:
        return

    griddata_points = list(
        zip(
            da.coords["x"].sel(icell2d=isvalid).values,
            da.coords["y"].sel(icell2d=isvalid).values,
            strict=False,
        )
    )
    gridpoint_values = da.sel(icell2d=isvalid).values
    qpoints = list(
        zip(
            da.coords["x"].sel(icell2d=ismissing).values,
            da.coords["y"].sel(icell2d=ismissing).values,
            strict=False,
        )
    )
    qvalues = griddata(
        points=griddata_points,
        values=gridpoint_values,
        xi=qpoints,
        method=method,
    )

    da.loc[{"icell2d": ismissing}] = qvalues


def _validate_inputs(
    layer_model_regis,
    layer_model_other,
    mask_model_other,
    transition_model,
    top,
    dfk,
    koppeltabel_header_regis,
    koppeltabel_header_other,
):
    """
    Validate input datasets and parameters for combining layer models.
    """
    # Check model extents match
    assert layer_model_regis.attrs["extent"] == layer_model_other.attrs["extent"], (
        "Extent of layer models are not equal"
    )

    # Check grid types match
    assert layer_model_regis.attrs["gridtype"] == layer_model_other.attrs["gridtype"], (
        "Gridtype of layer models are not equal"
    )

    # Check required variables exist in REGIS model
    assert all(var in layer_model_regis.variables for var in ["kh", "kv", "botm"]), (
        "Variable 'kh', 'kv', 'botm' is missing in layer_model_regis"
    )

    # Check no NaN values in REGIS model variables
    assert all(layer_model_regis[k].notnull().all() for k in ["kh", "kv", "botm"])

    # Check required variables exist in other model
    assert all(var in layer_model_other.variables for var in ["kh", "kv", "botm"]), (
        "Variable 'kh', 'kv', 'botm' is missing in layer_model_other"
    )

    # check mask_model_other variables
    assert all(np.issubdtype(dtype, bool) for dtype in mask_model_other.dtypes.values()), (
        "Variable 'kh', 'kv', and 'botm' in transition_model should be boolean"
    )

    # Check no NaN values in layer model other variables
    assert all(layer_model_other[k].where(mask_model_other[k], -999).notnull().all() for k in ["kh", "kv", "botm"]), (
        "Variable 'kh', 'kv', 'botm' in layer_model_other not should be NaN where mask_model_other is True"
    )

    # Validate transition model
    assert all(var in transition_model.variables for var in ["kh", "kv", "botm"]), (
        "Variable 'kh', 'kv', or 'botm' is missing in transition_model"
    )

    assert all(np.issubdtype(dtype, bool) for dtype in transition_model.dtypes.values()), (
        "Variable 'kh', 'kv', and 'botm' in transition_model should be boolean"
    )

    assert dfk[koppeltabel_header_regis].isin(layer_model_regis.layer.values).all(), (
        f"All values in koppeltabel[{koppeltabel_header_regis}] should be present in layer_model_regis.layer"
    )

    assert dfk[koppeltabel_header_other].isin(layer_model_other.layer.values).all(), (
        f"All values in koppeltabel[{koppeltabel_header_other}] should be present in layer_model_other.layer"
    )

    # No overlap between mask_model_other and transition_model
    assert all(
        (mask_model_other[k].astype(int) + transition_model[k].astype(int) < 2).all()
        for k in ["kh", "kv", "botm", "top"]
    ), "mask_model_other and transition_model should not overlap"

    # Check layer names don't contain underscores
    assert (
        not dfk[koppeltabel_header_regis].str.contains("_").any()
        and not dfk[koppeltabel_header_other].str.contains("_").any()
    ), "koppeltabel_header_regis and koppeltabel_header_other should not contain '_'"

    # Validate top. No nan's and grid should be the same as the others
    assert top.notnull().all(), "Top variable should not contain NaN values"
    assert all(top.coords["x"].values == layer_model_regis.coords["x"].values), (
        "Top variable should have the same x coordinates as layer_model_regis"
    )
    assert all(top.coords["y"].values == layer_model_regis.coords["y"].values), (
        "Top variable should have the same y coordinates as layer_model_regis"
    )
    assert all(top.coords["icell2d"].values == layer_model_regis.coords["icell2d"].values), (
        "Top variable should have the same icell2d coordinates as layer_model_regis"
    )
