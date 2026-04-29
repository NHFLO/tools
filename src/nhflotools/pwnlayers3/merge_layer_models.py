import logging

import nlmod
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

# Category constants for the combined layer model
CAT_REGIS = 1
CAT_OTHER = 2
CAT_TRANSITION = 3

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


def plot_layer(da, ds, **kwargs):
    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = nlmod.plot.get_map(ds.extent, base=1e4)
        ax.set_aspect("equal", adjustable="box")
    pc = nlmod.plot.data_array(da, ds=ds, ax=ax, **kwargs)
    return ax, pc


def combine_two_layer_models(
    *,
    layer_model_regis,
    layer_model_other,
    mask_model_other,
    transition_model,
    top,
    df_koppeltabel,
    koppeltabel_header_regis="Regis II v2.2",
    koppeltabel_header_other="ASSUMPTION1",
    transition_method="linear",
    split_method="equal",
    remove_nan_layers=True,
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
    transition_method : {'linear'}, optional
        Method to use for the transition zone. If 'linear', the values are
        linearly interpolated from the REGISII layer model to the OTHER
        layer model.
    split_method : {'equal', 'nearest_ratio'}, optional
        How to distribute thickness among sublayers created by splitting.
        If 'equal' (default), sublayers receive equal thickness.  If
        'nearest_ratio', thickness ratios are taken from the model that has
        actual sublayer data, extrapolated via nearest-neighbor to cells
        where that model is absent.
    remove_nan_layers : bool, optional
        if True layers that are inactive everywhere are removed from the model.
        If False nan layers are kept which might be usefull if you want
        to keep some layers that exist in other models. The default is True.

    Returns
    -------
    out : xarray Dataset
        Dataset containing the combined layer model with kh, kv, and botm. Attributes from
        layer_model_regis are copied to the output dataset.
    cat : xarray Dataset
        Dataset containing the category of the layers. The values are:
        1: REGISII layer
        2: OTHER layer
        3: transition zone

    """
    logger.info("Combining two layer models")

    # ── Phase 1: Prepare inputs ──────────────────────────────────────────
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

    koppeltabel = df_koppeltabel.copy()
    # kt_coupled: rows where both REGIS and OTHER are defined (layers to merge)
    kt_coupled = koppeltabel[~koppeltabel[koppeltabel_header_other].isna()]
    # kt_uncoupled: rows where OTHER is NaN (deep REGIS-only layers)
    kt_uncoupled = koppeltabel[koppeltabel[koppeltabel_header_other].isna()]

    # Apply mask: set OTHER model values to NaN outside the valid region
    for var in ["kh", "kv", "botm"]:
        layer_model_other[var] = layer_model_other[var].where(mask_model_other[var], np.nan)
        assert (layer_model_other[var].notnull() == mask_model_other[var]).all(), (
            f"There were nan values present in {var} in cells that should be valid"
        )

    # Filter koppeltabel to only include REGIS layers present in the model
    # (basenames handle layers that may already have _N suffixes from a prior merge)
    regis_basenames = [layer.split("_")[0] for layer in layer_model_regis.layer.values]
    kt_coupled = kt_coupled[kt_coupled[koppeltabel_header_regis].isin(regis_basenames)]

    # ── Phase 2: Split layers to create 1:1 correspondence ──────────────
    # Each REGIS layer that maps to N OTHER layers is split into N sublayers
    # (and vice versa).  Initially all splits use equal thickness.
    split_dict_regis = (
        kt_coupled.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis].count().to_dict()
    )
    split_dict_other = (
        kt_coupled.groupby(koppeltabel_header_other, sort=False)[koppeltabel_header_other].count().to_dict()
    )

    layer_model_regis_split = nlmod.dims.layers.split_layers_ds(
        ds=layer_model_regis.sel(layer=list(split_dict_regis.keys())),
        split_dict=split_dict_regis,
    )
    # Central invariant: kt_coupled rows and split_layer_names are in the same
    # order (both determined by koppeltabel row order with sort=False).  This
    # 1:1 correspondence is used by assign_coords and ratio adjustment below.
    split_layer_names = layer_model_regis_split.layer.values
    layer_model_other_split = nlmod.dims.layers.split_layers_ds(
        ds=layer_model_other.sel(layer=list(split_dict_other.keys())),
        split_dict=split_dict_other,
    ).assign_coords(layer=split_layer_names)

    # Optionally redistribute sublayer thicknesses using ratios from the
    # model that has actual sublayer data (instead of equal thickness)
    if split_method == "nearest_ratio":
        _adjust_botm_with_nearest_ratios(
            layer_model_regis_split=layer_model_regis_split,
            layer_model_other_split=layer_model_other_split,
            layer_model_regis=layer_model_regis,
            layer_model_other=layer_model_other,
            mask_model_other=mask_model_other,
            top=top,
            kt_coupled=kt_coupled,
            koppeltabel_header_regis=koppeltabel_header_regis,
            koppeltabel_header_other=koppeltabel_header_other,
        )
    elif split_method != "equal":
        msg = f"Unknown split_method: {split_method}. Use 'equal' or 'nearest_ratio'."
        raise ValueError(msg)

    # ── Phase 3: Categorize cells ────────────────────────────────────────
    # Expand mask and transition to the split layer names
    mask_model_other_split = mask_model_other.sel(layer=kt_coupled[koppeltabel_header_other].values).assign_coords(
        layer=split_layer_names
    )
    transition_model_split = transition_model.sel(layer=kt_coupled[koppeltabel_header_other].values).assign_coords(
        layer=split_layer_names
    )

    # Category is assigned strictly from the per-layer mask/transition and is not
    # modified afterwards — values 1=REGIS, 2=OTHER, 3=transition.
    category = xr.ones_like(layer_model_regis_split[["kh", "kv", "botm"]], dtype=int)
    category = category.where(~mask_model_other_split[["kh", "kv", "botm"]], other=CAT_OTHER)
    category = category.where(~transition_model_split[["kh", "kv", "botm"]], other=CAT_TRANSITION)

    # ── Phase 4: Combine and interpolate transition zone ─────────────────
    # Use REGIS values where category=1, OTHER where category=2.
    # Transition zone (category=3) has NaN from OTHER and is interpolated.
    combined = xr.where(
        category == CAT_REGIS,
        layer_model_regis_split[["kh", "kv", "botm"]].copy(),
        layer_model_other_split[["kh", "kv", "botm"]].copy(),
    )
    combined["top"] = top

    if transition_method == "linear":
        _interpolate_ds_inplace(
            combined, isvalid=category != CAT_TRANSITION, ismissing=category == CAT_TRANSITION, method="linear"
        )
    else:
        msg = f"Unknown transition method: {transition_method}. Use 'linear'."
        raise ValueError(msg)

    # ── Phase 5: Append uncoupled (deep) REGIS-only layers ───────────────
    regis_uncoupled = layer_model_regis.sel(layer=kt_uncoupled[koppeltabel_header_regis].values)
    result = xr.concat(
        [
            combined[["kh", "kv", "botm"]],
            regis_uncoupled[["kh", "kv", "botm"]].assign_coords(layer=regis_uncoupled.layer.values),
        ],
        dim="layer",
    )
    result["top"] = top

    # Per-column monotonicity sweep: ensure botm is non-increasing along the layer
    # axis without any cross-cell movement (no ffill, no spread between columns).
    botm = result["botm"].transpose("layer", "icell2d")
    botm_vals = botm.values.copy()
    np.minimum.accumulate(botm_vals, axis=0, out=botm_vals)
    result["botm"] = xr.DataArray(
        botm_vals,
        dims=botm.dims,
        coords=botm.coords,
        attrs=botm.attrs,
    ).transpose(*result["botm"].dims)

    result_category = xr.concat(
        [
            category[["kh", "kv", "botm"]],
            xr.ones_like(regis_uncoupled[["kh", "kv", "botm"]], dtype=int).assign_coords(
                layer=regis_uncoupled.layer.values
            ),
        ],
        dim="layer",
    )
    if remove_nan_layers:
        result = nlmod.dims.layers.remove_inactive_layers(result)
        result_category = result_category.sel(layer=result.layer.values)

    result.attrs = layer_model_regis.attrs

    return result, result_category


def _interpolate_ds_inplace(ds, isvalid, ismissing, method="linear"):
    """
    Interpolate the values of the dataset inplace where the mask is True.

    The values are interpolated from the values where the mask is False.
    The interpolation is done using the griddata function from scipy.
    Source data for layer L is restricted to cells of layer L; this function
    never reads from another layer.

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
    for k in ds:
        if "layer" not in ds[k].dims:
            continue

        for layer in ds[k].layer.values:
            isvalid_layer = isvalid[k].sel(layer=layer)
            ismissing_layer = ismissing[k].sel(layer=layer)
            assert bool((isvalid_layer ^ ismissing_layer).all().item()), (
                f"isvalid and ismissing must be disjoint and complementary on icell2d for variable {k}, layer {layer}"
            )
            _interpolate_da_inplace(
                ds[k].sel(layer=layer),
                isvalid_layer,
                ismissing_layer,
                method=method,
            )
            if np.any(np.isnan(ds[k].sel(layer=layer, icell2d=ismissing_layer).values)):
                _interpolate_da_inplace(
                    ds[k].sel(layer=layer),
                    isvalid_layer,
                    ismissing_layer,
                    method="nearest",
                )


def _interpolate_da_inplace(da, isvalid, ismissing, method="linear"):
    """
    Interpolate the values of the DataArray inplace where the mask is True.

    The values are interpolated from the values where the mask is False.
    The interpolation is done using the griddata function from scipy.
    Source data for layer L is restricted to cells of layer L; this function
    never reads from another layer.

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
    if isvalid.sum() == 0:
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


def _compute_thickness_ratios(ds_source, group_source_names, mask_valid, fallback="equal"):
    """Compute per-cell thickness ratios for a group of sublayers.

    For each cell where ``mask_valid`` is True, the ratio is the sublayer
    thickness divided by the total group thickness.  Where the total
    thickness is zero (all sublayers inactive) or where ``mask_valid`` is
    False, equal ratios ``1/N`` are assigned — there is no cross-cell
    extrapolation.

    Only the group layers (plus the layer immediately above for the group
    top) are used — thickness is **not** computed for the full source model.

    Parameters
    ----------
    ds_source : xr.Dataset
        Source dataset with actual sublayer data. Must contain 'botm' and
        'top'.
    group_source_names : list of str
        Layer names in ``ds_source`` for this group.  Must be consecutive
        layers within the source model.
    mask_valid : xr.DataArray
        Boolean mask with dimension ``icell2d``.  True where the source data
        is valid for **all** layers in the group.
    fallback : {'equal'}, optional
        How to fill ratios where ``mask_valid`` is False.  Only ``'equal'``
        (assign ``1/N``) is supported.

    Returns
    -------
    xr.DataArray
        Thickness ratios with dims ``(layer, icell2d)``.  The layer
        coordinate uses ``group_source_names``.  Ratios sum to 1 per cell.
    """
    if fallback != "equal":
        msg = f"Unknown fallback: {fallback!r}. Only 'equal' is supported."
        raise ValueError(msg)

    n = len(group_source_names)

    # Build a mini-dataset with only the group layers and the correct group
    # top, so thickness computation only depends on group-internal botm values
    # (plus the botm of the layer immediately above for the group top).
    all_layers = list(ds_source.layer.values)
    first_idx = all_layers.index(group_source_names[0])
    group_top = ds_source["top"] if first_idx == 0 else ds_source["botm"].sel(layer=all_layers[first_idx - 1])

    ds_group = ds_source[["botm"]].sel(layer=group_source_names).copy()
    ds_group["top"] = group_top
    thickness_group = nlmod.dims.layers.calculate_thickness(ds_group)

    total = thickness_group.sum(dim="layer")

    # Compute ratios.  Where total > 0 use actual ratios; where total is
    # zero or NaN (e.g. group_top is NaN outside mask) fall back to 1/N.
    # Note: NaN > 0 evaluates to False, so NaN cells also get 1/N.
    safe_total = xr.where(total > 0, total, 1.0)
    ratios = thickness_group / safe_total
    ratios = xr.where(total > 0, ratios, 1.0 / n)

    # Where the per-cell ratio is not well-defined, assign equal 1/N ratios.
    # No cross-cell extrapolation: each cell is decided from its own data only.
    for layer_name in group_source_names:
        ratios.loc[{"layer": layer_name}] = ratios.sel(layer=layer_name).where(mask_valid, 1.0 / n)

    # Normalize so ratios sum to exactly 1 per cell (guards against floating-
    # point drift).
    ratio_sum = ratios.sum(dim="layer")
    ratios /= xr.where(ratio_sum > 0, ratio_sum, 1.0)

    return ratios


def _apply_ratios_to_botm(ds_target, top, target_layer_names, ratios):
    """Recompute botm for sublayers using thickness ratios.

    The total group thickness (distance from the group top to the bottom of
    the last sublayer) is preserved from the existing equal split.  Only the
    distribution among sublayers changes.

    Parameters
    ----------
    ds_target : xr.Dataset
        Target dataset to modify **in-place**. Must contain 'botm'.
    top : xr.DataArray
        Model top elevation.
    target_layer_names : list of str
        Layer names in ``ds_target`` for the sublayers to adjust.
    ratios : xr.DataArray
        Thickness ratios with dims ``(layer, icell2d)``.  Must have the same
        number of layers as ``target_layer_names`` (matched by position).
    """
    botm = ds_target["botm"]

    # Determine the top of the first sublayer in this group
    all_layers = list(ds_target.layer.values)
    first_idx = all_layers.index(target_layer_names[0])
    if first_idx == 0:
        group_top = top
    else:
        prev_layer = all_layers[first_idx - 1]
        group_top = botm.sel(layer=prev_layer)

    # Total group thickness (preserved after ratio adjustment)
    group_bot = botm.sel(layer=target_layer_names[-1])
    total_thickness = group_top - group_bot

    # Cumulative ratios → new botm values
    cum_ratios = ratios.cumsum(dim="layer")
    for i, target_name in enumerate(target_layer_names):
        cr = cum_ratios.isel(layer=i)
        new_botm = group_top - cr * total_thickness
        ds_target["botm"].loc[{"layer": target_name}] = new_botm.values


def _adjust_botm_with_nearest_ratios(
    layer_model_regis_split,
    layer_model_other_split,
    layer_model_regis,
    layer_model_other,
    mask_model_other,
    top,
    kt_coupled,
    koppeltabel_header_regis,
    koppeltabel_header_other,
):
    """Adjust botm of split layers using nearest-neighbor thickness ratios.

    For each split group, compute thickness ratios from the model that has
    actual sublayer data and redistribute the equal-split botm accordingly.

    Two cases:

    - **REGIS split** (1 REGIS -> N OTHER, e.g. HLc -> W11..S13): ratios
      come from the OTHER model.  Cells where any OTHER sublayer is invalid
      receive equal ``1/N`` ratios (no cross-cell extrapolation).
    - **OTHER split** (M REGIS -> 1 OTHER, e.g. BXz1..EEz1 -> W21): ratios
      come from the REGIS model (which covers all cells).

    Groups where REGIS and OTHER are 1:1 are skipped (no split to adjust).

    Modifies ``layer_model_regis_split`` and ``layer_model_other_split``
    **in-place**.

    Parameters
    ----------
    layer_model_regis_split : xr.Dataset
        REGIS layer model after equal-thickness splitting.
    layer_model_other_split : xr.Dataset
        OTHER layer model after equal-thickness splitting (coordinates
        already aligned to REGIS split names).
    layer_model_regis : xr.Dataset
        Original (unsplit) REGIS layer model with 'top' and 'botm'.
    layer_model_other : xr.Dataset
        Original (unsplit) OTHER layer model with 'top' and 'botm'.
    mask_model_other : xr.Dataset
        Boolean mask indicating valid cells in the OTHER model.
    top : xr.DataArray
        Model top elevation.
    kt_coupled : pandas.DataFrame
        Koppeltabel rows for coupled layers (no NaN in the OTHER column).
    koppeltabel_header_regis : str
        Column name for REGIS layer names in the koppeltabel.
    koppeltabel_header_other : str
        Column name for OTHER layer names in the koppeltabel.
    """
    split_layer_names = list(layer_model_regis_split.layer.values)

    # The 1:1 correspondence between kt_coupled rows and split layer names
    # is the central invariant: both are ordered by koppeltabel row order
    # (groupby uses sort=False), so row i maps to split_layer_names[i].
    if len(kt_coupled) != len(split_layer_names):
        msg = f"Koppeltabel row count ({len(kt_coupled)}) does not match split layer count ({len(split_layer_names)})"
        raise ValueError(msg)

    kt = kt_coupled.copy()
    kt["split_layer"] = split_layer_names

    # ── REGIS split groups (1 REGIS → N OTHER) ──
    for regis_name, group_df in kt.groupby(koppeltabel_header_regis, sort=False):
        if len(group_df) <= 1:
            continue

        other_names = list(group_df[koppeltabel_header_other].values)
        target_names = list(group_df["split_layer"].values)

        # Use cells where ALL sublayers have valid data for ratio computation.
        # Cells where the AND fails fall back to equal 1/N ratios — no NN spread.
        mask_valid = mask_model_other["botm"].sel(layer=other_names).all(dim="layer")
        ratios = _compute_thickness_ratios(layer_model_other, other_names, mask_valid, fallback="equal")

        _apply_ratios_to_botm(layer_model_regis_split, top, target_names, ratios)
        logger.info(
            "Adjusted REGIS split group '%s' (%d sublayers) using thickness ratios from OTHER model",
            regis_name,
            len(target_names),
        )

    # ── OTHER split groups (M REGIS → 1 OTHER) ──
    for other_name, group_df in kt.groupby(koppeltabel_header_other, sort=False):
        if len(group_df) <= 1:
            continue

        regis_names = list(group_df[koppeltabel_header_regis].values)
        target_names = list(group_df["split_layer"].values)

        # REGIS covers all cells, so mask_valid is True everywhere; the
        # equal fallback is therefore never invoked here.
        mask_valid = xr.DataArray(
            np.ones(layer_model_regis.sizes["icell2d"], dtype=bool),
            dims=("icell2d",),
            coords={"icell2d": layer_model_regis.coords["icell2d"]},
        )
        ratios = _compute_thickness_ratios(layer_model_regis, regis_names, mask_valid, fallback="equal")

        _apply_ratios_to_botm(layer_model_other_split, top, target_names, ratios)
        logger.info(
            "Adjusted OTHER split group '%s' (%d sublayers) using thickness ratios from REGIS model",
            other_name,
            len(target_names),
        )


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
    """Validate input datasets and parameters for combining layer models."""
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

    # Check no NaN values in layer model other variables inside the declared mask.
    for k in ["kh", "kv", "botm"]:
        assert layer_model_other[k].where(mask_model_other[k], -999).notnull().all(), (
            f"layer_model_other has NaN inside mask_model_other for variable {k}; "
            "ensure upstream builders fill NaN within their declared masks before calling combine_two_layer_models"
        )

    # Validate transition model
    assert all(var in transition_model.variables for var in ["kh", "kv", "botm"]), (
        "Variable 'kh', 'kv', or 'botm' is missing in transition_model"
    )

    assert all(np.issubdtype(dtype, bool) for dtype in transition_model.dtypes.values()), (
        "Variable 'kh', 'kv', and 'botm' in transition_model should be boolean"
    )

    # If REGIS was already merged with another model, the layer names can contain underscores.
    basenames_regis = {layer.split("_")[0] for layer in layer_model_regis.layer.values}

    # Check koppeltabel values are present in layer models
    assert set(dfk[koppeltabel_header_regis]) == basenames_regis, (
        f"All values in koppeltabel[{koppeltabel_header_regis}] should be present in layer_model_regis.layer"
    )

    basenames_other = {layer.split("_")[0] for layer in layer_model_other.layer.values}
    # Check koppeltabel values are present in layer models
    assert basenames_other.issubset(set(dfk[koppeltabel_header_other])), (
        f"All values in koppeltabel[{koppeltabel_header_other}] should be present in layer_model_other.layer"
    )

    # No overlap between mask_model_other and transition_model
    assert all(~(mask_model_other[k] & transition_model[k]).any() for k in ["kh", "kv", "botm"]), (
        "mask_model_other and transition_model should not overlap"
    )

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
