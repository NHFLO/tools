import logging

from pytest import skip

import nlmod
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

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
    logger.info("Combining layer models")

    dfk = df_koppeltabel.copy()

    # Validate input datasets
    _validate_inputs(
        layer_model_regis,
        layer_model_other,
        mask_model_other,
        transition_model,
        top,
        dfk,
        koppeltabel_header_regis,
        koppeltabel_header_other,
    )

    # Apply mask to other layer model
    for var in ["kh", "kv", "botm"]:
        layer_model_other[var] = layer_model_other[var].where(mask_model_other[var], np.nan)
        assert (layer_model_other[var].notnull() == mask_model_other[var]).all(), (
            f"There were nan values present in {var} in cells that should be valid"
        )

    # Basename can occur multiple times if previously combined
    basenames_regis = [layer.split("_")[0] for layer in layer_model_regis.layer.values]
    basenames_other = [layer.split("_")[0] for layer in layer_model_other.layer.values]

    # Only select part of the table that appears in the two layer models
    dfk_mask = dfk[koppeltabel_header_regis].isin(basenames_regis) & dfk[koppeltabel_header_other].isin(basenames_other)
    dfk = dfk[dfk_mask]

    # Names of the layers, including split, in the REGISII and OTHER layer models
    dfk["Regis_split_index"] = (dfk.groupby(koppeltabel_header_regis, sort=False).cumcount() + 1).astype(str)
    dfk["Regis_split"] = dfk[koppeltabel_header_regis].str.cat(dfk["Regis_split_index"], sep="_")
    dfk["Regis_split_n"] = dfk[koppeltabel_header_regis].map(
        dfk.groupby(koppeltabel_header_regis)[koppeltabel_header_regis].count()
    )
    dfk["Other_split_n"] = dfk[koppeltabel_header_other].map(
        dfk.groupby(koppeltabel_header_other)[koppeltabel_header_other].count()
    )

    # Count in how many layers the REGISII and OTHER layers need to be split if previously never combined (default)
    # split_counts_regis_def = (
    #     dfk.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis].count().to_dict()
    # )
    # split_counts_other_def = (
    #     dfk.groupby(koppeltabel_header_other, sort=False)[koppeltabel_header_other].count().to_dict()
    # )

    # Split both layer models
    layer_model_regis_split = layer_model_regis.sel(layer=dfk[koppeltabel_header_regis].values).assign_coords(
        layer=dfk["Regis_split"]
    )
    layer_model_other_split = layer_model_other.sel(layer=dfk[koppeltabel_header_other].values).assign_coords(
        layer=dfk["Regis_split"]
    )
    mask_model_other_split = mask_model_other.sel(layer=dfk[koppeltabel_header_other].values).assign_coords(
        layer=dfk["Regis_split"]
    )
    transition_model_split = transition_model.sel(layer=dfk[koppeltabel_header_other].values).assign_coords(
        layer=dfk["Regis_split"]
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

    # Linearly interpolate transition zone inplace
    _interpolate_ds(out, isvalid=cat != 3, ismissing=cat == 3, method="linear")

    # Add top and remaining metadata
    out["top"] = top

    """All done except for the botm of the newly split layers"""
    # Set botm for new layers in regis and other. First compute thickness of layers
    # Evenly divide the thickness over the new layers in layer_model_regis_split and layer_model_other_split.
    thick_regis = nlmod.dims.layers.calculate_thickness(xr.merge((layer_model_regis[["botm"]], {"top": top})))
    thick_other = nlmod.dims.layers.calculate_thickness(xr.merge((layer_model_other[["botm"]], {"top": top})))

    thick_regis_split = thick_regis.sel(layer=dfk[koppeltabel_header_regis].values).assign_coords({
        "layer": dfk["Regis_split"].values
    })
    thick_other_split = thick_other.sel(layer=dfk[koppeltabel_header_other].values).assign_coords({
        "layer": dfk["Regis_split"].values
    })

    thick_regis_split.coords[koppeltabel_header_other] = ("layer", dfk[koppeltabel_header_other])  # Used by groupby
    thick_other_split.coords[koppeltabel_header_regis] = ("layer", dfk[koppeltabel_header_regis])  # Used by groupby
    thick_regis_split_sum = (
        thick_regis_split.groupby(koppeltabel_header_other)
        .sum(skipna=True)
        .sel(**{koppeltabel_header_other: dfk[koppeltabel_header_other].values})
    )
    thick_other_split_sum = (
        thick_other_split.groupby(koppeltabel_header_regis)
        .sum(skipna=True)
        .sel(**{koppeltabel_header_regis: dfk[koppeltabel_header_regis].values})
    )
    thick_regis_split_sum = thick_regis_split_sum.rename({koppeltabel_header_other: "layer"}).assign_coords(
        layer=dfk["Regis_split"].values
    )
    thick_other_split_sum = thick_other_split_sum.rename({koppeltabel_header_regis: "layer"}).assign_coords(
        layer=dfk["Regis_split"].values
    )

    # Total thickness is zero, leads to division by zero. Division in equal parts if zero-thickness.
    regis_split_n_da = xr.DataArray(dfk["Regis_split_n"].values, coords={"layer": dfk["Regis_split"].values})
    other_split_n_da = xr.DataArray(dfk["Other_split_n"].values, coords={"layer": dfk["Regis_split"].values})
    ratio_regis_split = xr.where(
        thick_regis_split_sum != 0.0, thick_regis_split / thick_regis_split_sum, 1 / other_split_n_da
    )
    ratio_other_split = xr.where(
        thick_other_split_sum != 0.0, thick_other_split / thick_other_split_sum, 1 / regis_split_n_da
    )

    a = ratio_regis_split.groupby(koppeltabel_header_other).sum()
    b = ratio_other_split.groupby(koppeltabel_header_regis).sum()
    xr.where(cat.botm == 1, a, b)
    # Find non-lowest layer of split layers: HLc_1, HLc_2, ..., HLc_5 => True, True, True, True, False
    # Botm of the lowest is already at the correct height.
    # isnonlowestlayer_regis = np.zeros(len(dfk), dtype=bool)
    # isnonlowestlayer_other = np.zeros(len(dfk), dtype=bool)
    # isnonlowestlayer_regis[:-1] = dfk[koppeltabel_header_regis].values[:-1] == dfk[koppeltabel_header_regis].values[1:]
    # isnonlowestlayer_other[:-1] = dfk[koppeltabel_header_other].values[:-1] == dfk[koppeltabel_header_other].values[1:]

    # Extrapolate using nearest neighbor.
    # iter_regis = ratio_regis_split.sel(layer=isnonlowestlayer_regis).groupby("layer")
    # iter_other = ratio_other_split.sel(layer=isnonlowestlayer_other).groupby("layer")
    # for (name_regis, layer_regis), (name_other, layer_other) in zip(iter_regis, iter_other):
    #     print(name_regis, layer_regis, name_other, layer_other)

    # Assign botm to newly split regis layers, using nearest neighbor ratios from layer_model_other
    # other contains the data that needs to be extrapolated
    for name in dfk.loc[dfk["Regis_split_n"] > 1, "Regis_split"].values:
        ratio = ratio_other_split.sel(layer=name)
        isregis = (cat.botm == 1).sel(layer=name).values
        isother = (cat.botm == 2).sel(layer=name).values

        # If all values are valid, no extrapolation is needed
        if not any(isregis) or not any(isother):
            continue

        points = list(
            zip(
                ratio.coords["x"].sel(icell2d=isother).values,
                ratio.coords["y"].sel(icell2d=isother).values,
                strict=True,
            )
        )
        values = ratio.sel(icell2d=isother).values
        qpoints = list(
            zip(
                ratio.coords["x"].sel(icell2d=isregis).values,
                ratio.coords["y"].sel(icell2d=isregis).values,
                strict=True,
            )
        )
        qvalues = griddata(
            points=points,
            values=values,
            xi=qpoints,
            method="nearest",
        )
        ratio_regis_split.loc[{"layer": name}].loc[isregis] = qvalues

    # Enrich ratio_other_split with the values of ratio_regis_split for new layers
    is_new_other_layer = dfk["Other_split_n"].values > 1
    ratio_other_split.loc[{"layer": is_new_other_layer}] = ratio_regis_split.loc[{"layer": is_new_other_layer}]

    assert ratio_regis_split.groupby(koppeltabel_header_regis).sum() == 1.0
    assert ratio_other_split.groupby(koppeltabel_header_regis).sum() == 1.0
    thick_regis_split_sum.coords[koppeltabel_header_regis] = ("layer", dfk[koppeltabel_header_regis])  # Used by groupby
    thick_other_split_sum.coords[koppeltabel_header_other] = ("layer", dfk[koppeltabel_header_other])  # Used by groupby

    thick_split_cumsum_regis = (
        (ratio_regis_split * thick_other_split_sum).groupby(koppeltabel_header_regis).cumsum(dim="layer")
    )

    # REGIS is defined everywhere, so assign botm to newly split other layers, using the ratios
    # merge ratio_other_split and ratio_regis_split
    ratio_split = xr.where(cat.botm == 2, ratio_other_split, ratio_regis_split)
    thick_split = xr.where(cat.botm == 2, thick_other_split, ratio_regis_split * thick_regis_split_sum)

    # # Evenly split new layers
    # nsplit_regis = dfk[koppeltabel_header_regis].map(dfk.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis].size())
    # nsplit_other = dfk[koppeltabel_header_other].map(dfk.groupby(koppeltabel_header_other, sort=False)[koppeltabel_header_other].size())
    # thick_ratio_regis = xr.DataArray(data=(nsplit_regis - dfk.groupby(koppeltabel_header_regis, sort=False).cumcount() - 1) / nsplit_regis, coords={"layer": dfk["Regis_split"].values})
    # thick_ratio_other = xr.DataArray(data=(nsplit_other - dfk.groupby(koppeltabel_header_other, sort=False).cumcount() - 1) / nsplit_other, coords={"layer": dfk["Regis_split"].values})

    # thick_regis_split = thick_regis.sel(
    #     layer=np.concatenate([v * [k] for k, v in split_counts_regis_def.items()])
    # ).assign_coords({"layer": dfk["Regis_split"].values}) * thick_ratio_regis
    # thick_other_split = thick_other.sel(
    #     layer=np.concatenate([v * [k] for k, v in split_counts_other_def.items()])
    # ).assign_coords({"layer": dfk["Regis_split"].values}) * thick_ratio_other

    thick_other_split.coords[koppeltabel_header_regis] = ("layer", dfk[koppeltabel_header_regis])
    thick_regis_split.coords[koppeltabel_header_other] = ("layer", dfk[koppeltabel_header_other])

    # thick_other_split.groupby(koppeltabel_header_regis).sum().sel(**{koppeltabel_header_regis: dfk[koppeltabel_header_regis].values})

    layer_model_regis_split["botm"] += thick_regis_split
    layer_model_other_split["botm"] += thick_other_split

    return out, cat


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
                ds[k].sel(layer=layer), isvalid[k].sel(layer=layer), ismissing[k].sel(layer=layer), method=method
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


# def combine_two_layer_models2(
#     df_koppeltabel,
#     layer_model_regis,
#     layer_model_other,
#     mask_model_other,
#     transition_model=None,
#     koppeltabel_header_regis="Regis II v2.2",
#     koppeltabel_header_other="ASSUMPTION1",
# ):
#     """
#     Combine the layer models of REGISII and OTHER.

#     The values of the OTHER layer model are used where the layer_model_other is not nan. Mask_model_other
#     is used to help set the values of the layer_model_other to nan where the layer model is not defined.
#     The values of the REGISII layer model are used where the layer_model_other is nan
#     and transition_model is False. The remaining values are where the
#     transition_model is True. Those values are linearly interpolated from the
#     REGISII layer model to the OTHER layer model.

#     `layer_model_regis` and `layer_model_other` should have the same grid.
#     The layer names of `layer_model_other` should be present in koppeltabel[`koppeltabel_header_other`].
#     The layer names of `layer_model_regis` should be present in koppeltabel[`koppeltabel_header_regis`].
#     To guarantee the coupling is always valid, the koppeltabel should be defined for all interlaying
#     REGISII layers, this is not enforced.

#     Note that the top variable is required in both layer models to be able to split
#     and combine the top layer.

#     TODO: Check that top is not merged and taken from layer_model 1.

#     Parameters
#     ----------
#     df_koppeltabel : pandas DataFrame
#         DataFrame containing the koppeltabel. koppeltabel[`koppeltabel_header_other`]
#         should contain the layer names of `layer_model_other` and
#         koppeltabel[`koppeltabel_header_regis`] should contain the layer names of
#         `layer_model_regis`.
#     layer_model_regis : xarray Dataset
#         Dataset containing the layer model of REGISII. It should contain the
#         variables 'kh', 'kv', 'botm', and 'top'.
#     layer_model_other : xarray Dataset
#         Dataset containing the layer model of OTHER. It should have nan values
#         where the layer model is not defined. It should contain the variables
#         'kh', 'kv', 'botm', and 'top'.
#     transition_model : xarray Dataset, optional
#         Dataset containing the transition model of OTHER. It should contain
#         the variables 'kh', 'kv', 'botm'. The default is None.
#         It should be True where the transition between layer_model_regis and layer_model_other
#         is defined and False where it is not. Where True, the values of are linearly interpolated
#         from the REGISII layer model to the OTHER layer model. If None, the transition is not used.
#     koppeltabel_header_regis : str, optional
#         Column name of the koppeltabel containing the REGISII layer names.
#         The default is 'Regis II v2.2'.
#     koppeltabel_header_other : str, optional
#         Column name of the koppeltabel containing the OTHER layer names.
#         The default is 'ASSUMPTION1'.

#     Returns
#     -------
#     layer_model_out : xarray Dataset
#         Dataset containing the combined layer model.

#         layer_model_regis contains all layers of REGISII. Only a subset of those layers connect to other. The connection is defined in the koppeltabel.

#     Notes
#     -----
#     layer_model_tophalf, layer_model_other:
#         The top layers are the layers that are connected via the koppeltabel. It requires the koppeltabel to be valid for those layers and those layers should be present in layer_model_other.

#     layer_model_other_split, layer_model_top_split:
#         If multiple other layers are connected to a single regis layer, the regis layer is split into multiple layers. The thickness of the split layers is extrapolated from the other layers.
#         If multiple regis layers are connected to a single other layer, the other layer is split into multiple layers. The thickness of the split layers is extrapolated from the regis layers.

#     Bottom layers:
#     layer_model_bothalf:
#         The bottom layers are the layers that are not connected via the koppeltabel and the layers from layer_model_regis are used.
#         layer_model_other_split is a ds

#     Connection notes:
#     - NaN values in layer_model_other mean that those values are not defined in the other layer model. The values of layer_model_regis are used instead.
#     - A thickness of zero in layer_model_other means that the layer is absent in the other layer model.

#     """
#     dfk = df_koppeltabel.copy()

#     # Validate input datasets
#     _validate_inputs(
#         layer_model_regis,
#         layer_model_other,
#         transition_model,
#         dfk,
#         koppeltabel_header_regis,
#         koppeltabel_header_other
#     )

#     # Apply mask to other layer model
#     for var in ["kh", "kv", "botm"]:
#         layer_model_other[var] = layer_model_other[var].where(mask_model_other[var], np.nan)
#         assert (
#             layer_model_other[var].notnull() == mask_model_other[var]
#         ).all(), f"There were nan values present in {var} in cells that should be valid"


#     assert (
#         not dfk[koppeltabel_header_regis].str.contains("_").any()
#         and not dfk[koppeltabel_header_other].str.contains("_").any()
#     ), "koppeltabel_header_regis and koppeltabel_header_other should not contain '_'"

#     # Basename can occur multiple times if previously combined
#     basenames_regis = [layer.split("_")[0] for layer in layer_model_regis.layer.values]
#     basenames_other = [layer.split("_")[0] for layer in layer_model_other.layer.values]

#     # Only select part of the table that appears in the two layer models
#     dfk_mask = dfk[koppeltabel_header_regis].isin(basenames_regis) & dfk[koppeltabel_header_other].isin(basenames_other)
#     dfk = dfk[dfk_mask]

#     logger.info("Combining layer models")

#     # Construct a new layer index for the split REGIS layers
#     dfk["Regis_split_index"] = (dfk.groupby(koppeltabel_header_regis).cumcount() + 1).astype(str)
#     dfk["Regis_split"] = dfk[koppeltabel_header_regis].str.cat(dfk["Regis_split_index"], sep="_")
#     dfk["OTHER_split_index"] = (dfk.groupby(koppeltabel_header_other).cumcount() + 1).astype(str)
#     dfk["OTHER_split"] = dfk[koppeltabel_header_other].str.cat(dfk["OTHER_split_index"], sep="_")

#     # Leave out lower REGIS layers
#     top_regis_mask = np.array([i in dfk[koppeltabel_header_regis].values for i in basenames_regis])
#     assert np.diff(top_regis_mask).sum() == 1, "REGIS layers should be consequtive from top to bottom."

#     layer_model_tophalf = layer_model_regis.sel(layer=top_regis_mask)
#     layer_model_bothalf = layer_model_regis.sel(layer=~top_regis_mask)

#     # Count in how many layers the REGISII layers need to be split if previously never combined (default)
#     split_counts_regis_def = (
#         dfk.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis].count().to_dict()
#     )
#     # New layer names for the split REGIS layers. If HLc is split in 6 layers,
#     # HLc_1, HLc_2, ..., HLc_5 are created. HLc is renamed to HLc_6, as it has
#     # the correct botm, and is therefore not considered new.
#     layer_names_regis_new = np.concatenate([
#         [f"{k}_{vi}" for vi in range(1, v)] for k, v in split_counts_regis_def.items() if v > 1
#     ])
#     # used for adjusting botm of split layers
#     layer_names_regis_new_dict = {
#         k: [f"{k}_{vi}" for vi in range(1, v)] for k, v in split_counts_regis_def.items() if v > 1
#     }

#     # Count in how many layers the OTHER layers need to be split if previously never combined
#     split_counts_other_def = (
#         dfk.groupby(koppeltabel_header_other, sort=False)[koppeltabel_header_other].count().to_dict()
#     )
#     layer_names_other_new = np.concatenate([
#         [f"{k}_{vi}" for vi in range(1, v)] for k, v in split_counts_other_def.items() if v > 1
#     ])
#     # used for adjusting botm of split layers
#     layer_names_other_new_dict = {
#         k: [f"{k}_{vi}" for vi in range(1, v)] for k, v in split_counts_other_def.items() if v > 1
#     }

#     # Split both layer models with evenly-split thickness
#     layer_model_other_split = layer_model_other.sel(
#         layer=np.concatenate([v * [k] for k, v in split_counts_other_def.items()])
#     )
#     layer_model_other_split = layer_model_other_split.assign_coords(layer=dfk["Regis_split"])
#     # Set botm of new layers to nan
#     mask = dfk["Regis_split"][dfk["OTHER_split"].isin(layer_names_other_new)].values
#     layer_model_other_split["botm"].loc[{"layer": mask}] = np.nan

#     # layer_model_other_split where True, layer_model_top_split where False
#     valid_other_layers = layer_model_other_split["botm"].notnull()

#     if layer_model_regis.layer.str.contains("_").any():
#         # TODO: if previously combined layer_model needs to be split for a second time
#         split_counts_regis_cur = dict(zip(*np.unique(basenames_regis, return_counts=True), strict=False))
#         assert all(
#             v == split_counts_regis_cur[k] for k, v in split_counts_regis_def.items()
#         ), "Previously combined REGIS layers should be split in the same number of layers as before."
#         layer_model_top_split = layer_model_tophalf
#     else:
#         layer_model_top_split = layer_model_tophalf.sel(
#             layer=np.concatenate([v * [k] for k, v in split_counts_regis_def.items()])
#         )
#         # Set botm of new layers to nan
#         layer_model_top_split = layer_model_top_split.assign_coords(layer=dfk["Regis_split"])
#         layer_model_top_split["botm"].loc[{"layer": layer_names_regis_new}] = np.nan

#     # extrapolate thickness of split layers
#     thick_regis_top_split = nlmod.dims.layers.calculate_thickness(layer_model_top_split)
#     thick_other_split = nlmod.dims.layers.calculate_thickness(layer_model_other_split)

#     # assert not (thick_regis_top_split.fillna(0.) < 0.).any(), "Regis thickness of layers should be positive"
#     # assert not (thick_other_split.fillna(0.) < 0.).any(), "Other's thickness of layers should be positive"

#     # best estimate thickness of unsplit regis layers
#     elev_regis = xr.concat(
#         (
#             layer_model_regis["top"].expand_dims(layer=["mv"]),
#             layer_model_regis["botm"],
#         ),
#         dim="layer",
#     )
#     top_regis = elev_regis.isel(layer=slice(-1)).assign_coords(layer=layer_model_regis.layer.values)
#     elev_other = xr.concat(
#         (
#             layer_model_other["top"].expand_dims(layer=["mv"]),
#             layer_model_other["botm"],
#         ),
#         dim="layer",
#     )
#     top_other = elev_other.isel(layer=slice(-1)).assign_coords(layer=layer_model_other.layer.values)

#     botm_regis, layers, layers_other = _split_layers_regis(layer_model_regis, layer_model_other, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_names_regis_new, layer_names_regis_new_dict, layer_model_top_split, thick_other_split, top_regis, top_other)

#     botm_other = _split_layers_other(layer_model_regis, layer_model_other, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_names_other_new_dict, layer_model_other_split, thick_regis_top_split, top_regis, top_other, layers, layers_other)

#     layer_model_out, cat = _merge_layer_models(layer_model_regis, transition_model, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_model_bothalf, layer_names_regis_new_dict, layer_names_other_new_dict, layer_model_other_split, valid_other_layers, layer_model_top_split, botm_regis, botm_other)

#     return layer_model_out, cat

# def _split_layers_regis(layer_model_regis, layer_model_other, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_names_regis_new, layer_names_regis_new_dict, layer_model_top_split, thick_other_split, top_regis, top_other):
#     """
#     Connecting multiple OTHER layers to one REGIS layer
#     -------------------------------------------------
#     In a previous step, extra layers were added to layer_model_top_split so that one REGIS layer
#     can connect to multiple OTHER layers. The botm is already at the correct elevation for where
#     the OTHER layers are present (layer_model_other_split). The botm's of those layers is here set
#     to the is now adjusted so that the thickness
#     The total_thickness_layers is the sum of the thickness of the OTHER layers that are connected
#     to the one REGIS layer. The total thickness of the OTHER layers is used if available, else the
#     total thickness of the REGIS layers is used. The thickness of the OTHER layers is extrapolated
#     into the areas of the REGIS layers.
#     The thick_ratio_other is the ratio of the thickness of the OTHER layers with respect to total thickness
#     that is extrapolated into the REGIS layer. The thick_ratio_other is used to calculate the elevations
#     of the botm of the newly split REGIS layers.
#     """

#     logger.info(f"Adjusting the botm of the newly split REGIS layers: {layer_names_regis_new}")

#     # Modifying layer_model_top_split["botm"] in place.
#     botm_regis = layer_model_top_split["botm"].copy()
#     for name, group in dfk.groupby(koppeltabel_header_regis):
#         if name not in layer_names_regis_new_dict:
#             # This REGIS layer is not split
#             continue

#         layers = group["Regis_split"].values
#         layers_other = dfk[koppeltabel_header_other][dfk["Regis_split"].isin(layers)].values
#         new_layers = layer_names_regis_new_dict[name]

#         if all(i in layer_model_regis.layer.values for i in new_layers):
#             # layer_model_regis is previously combined and already split
#             logger.info(
#                 f"Previously combined REGIS layers: {name} are already split. "
#                 "The botm's for these layers are only adjusted where non-nan other "
#                 "data is provided and in the transition zone."
#             )
#             continue
#         assert  not any(
#             i in layer_model_regis.layer.values for i in new_layers
#         ), "Previously combined REGIS layers should not be split for a second time."

#         logger.info(
#             f"About to adjust the botm of the newly split REGIS layers: {new_layers}. "
#             f"{layers[-1]} already has the correct elevation"
#         )

#         # Top of combined layers
#         top_total = xr.where(
#             top_other.sel(layer=layers_other[0]).notnull(),
#             top_other.sel(layer=layers_other[0]),
#             top_other.sel(layer=layers_other)
#             .max(dim="layer")
#             .where(
#                 top_other.sel(layer=layers_other).max(dim="layer").notnull(),
#                 top_regis.sel(layer=name),
#             ),
#         )

#         botm_total = xr.where(
#             layer_model_other["botm"].sel(layer=layers_other[-1]).notnull(),
#             layer_model_other["botm"].sel(layer=layers_other[-1]),
#             layer_model_other["botm"]
#             .sel(layer=layers_other)
#             .min(dim="layer")
#             .where(
#                 layer_model_other["botm"].sel(layer=layers_other).min(dim="layer").notnull(),
#                 layer_model_regis["botm"].sel(layer=name),
#             ),
#         )

#         if (top_total < botm_total).any():
#             logger.warning("Total thickness of layers should be positive.")

#         total_thickness_layers = top_total - botm_total

#         # thick ratios of other layers that need to be extrapolated into the areas of the regis layers
#         thick_ratio_other = xr.where(
#             total_thickness_layers != 0.0,
#             thick_other_split.sel(layer=layers) / total_thickness_layers,
#             0.0,
#         )

#         for layer in new_layers:
#             mask = thick_ratio_other.sel(layer=layer).notnull()  # locate valid values

#             if mask.sum() == 0:
#                 logger.info(
#                     f"Insufficient data in layer_model_other to extrapolate {layer} thickness into "
#                     f"layer {name}. Splitting layers evenly."
#                 )
#                 continue

#             griddata_points = list(
#                 zip(
#                     thick_ratio_other.coords["x"].sel(icell2d=mask).values,
#                     thick_ratio_other.coords["y"].sel(icell2d=mask).values,
#                     strict=False,
#                 )
#             )
#             gridpoint_values = thick_ratio_other.sel(layer=layer, icell2d=mask).values
#             qpoints = list(
#                 zip(
#                     thick_ratio_other.coords["x"].sel(icell2d=~mask).values,
#                     thick_ratio_other.coords["y"].sel(icell2d=~mask).values,
#                     strict=False,
#                 )
#             )
#             qvalues = griddata(
#                 points=griddata_points,
#                 values=gridpoint_values,
#                 xi=qpoints,
#                 method="nearest",
#             )

#             thick_ratio_other.loc[{"layer": layer, "icell2d": ~mask}] = qvalues

#         # evenly fill up missing thick_ratio values. Same for all layers.
#         fillna = (1 - thick_ratio_other.sum(dim="layer", skipna=True)) / thick_ratio_other.isnull().sum(
#             dim="layer", skipna=True
#         )
#         thick_ratio_other = thick_ratio_other.fillna(fillna)

#         botm_split = top_total - (thick_ratio_other * total_thickness_layers).cumsum(dim="layer", skipna=False)
#         botm_regis.loc[{"layer": layers}] = botm_split
#     return botm_regis,layers,layers_other


# def _split_layers_other(layer_model_regis, layer_model_other, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_names_other_new_dict, layer_model_other_split, thick_regis_top_split, top_regis, top_other, layers, layers_other):
#     """
#     Connecting one OTHER layer to multiple REGIS layers
#     -------------------------------------------------
#     In a previous step, extra layers were added to layer_model_other_split so that one OTHER layer
#     can connect to multiple REGIS layers. Outside of the OTHER layers, the botm of the multiple
#     REGIS layers is already at the correct elevation (layer_model_top_split). Inside of the OTHER
#     layers only the lower botm and the upper top are at the correct elevation. The elevation
#     intermediate botm's inside the OTHER layers is set here.
#     To estimate the thickness of the intermediate layers, the origional layer thickness over
#     total thickness ratio of the REGIS layers at the the location of the OTHER layer is used
#     (thick_ratio_regis). This strategy is chosen so that the transition between OTHER and REGIS
#     layers is smooth.
#     """
#     logger.info("Adjusting the botm of the newly split OTHER layers")
#     del layers_other, layers
#     botm_other = layer_model_other_split["botm"].copy()
#     for name, group in dfk.groupby(koppeltabel_header_other):
#         if name not in layer_names_other_new_dict:
#             # This OTHER layer is not split
#             continue

#         layers = group["Regis_split"].values
#         new_layers = layer_names_other_new_dict[name]

#         if any("_" in i for i in layer_model_regis.layer.values):
#             layers_regis = group["Regis_split"].values
#         else:
#             layers_regis = group[koppeltabel_header_regis].values

#         logger.info(
#             f"About to adjust the botm of the newly split OTHER layers: {new_layers}. "
#             f"{layers[-1]} already has the correct elevation"
#         )

#         # thick ratios of regis layers that need to be extrapolated into the areas of the other layers
#         total_thickness_layers_regis = thick_regis_top_split.sel(layer=layers).sum(dim="layer", skipna=False)
#         thick_ratio_regis = xr.where(
#             total_thickness_layers_regis != 0.0,
#             thick_regis_top_split.sel(layer=layers) / total_thickness_layers_regis,
#             0.0,
#         )

#         top_total = xr.where(
#             top_other.sel(layer=name).notnull(),
#             top_other.sel(layer=name),
#             top_regis.sel(layer=layers_regis[0]),
#         )
#         _top = top_other.sel(layer=slice(name)).min(dim="layer")
#         top_total = top_total.where(~((top_total > _top) & _top.notnull()), _top)
#         _top = top_other.sel(layer=slice(name, None)).max(dim="layer")
#         top_total = top_total.where(~((top_total < _top) & _top.notnull()), _top)

#         # Botm of combined layers
#         botm_total = xr.where(
#             layer_model_other["botm"].sel(layer=name).notnull(),
#             layer_model_other["botm"].sel(layer=name),
#             layer_model_regis["botm"].sel(layer=layers_regis[-1]),
#         )
#         botm_total = botm_total.where(
#             botm_total < top_total,
#             top_total,
#         )
#         total_thickness_layers = top_total - botm_total
#         assert (total_thickness_layers.fillna(0.0) >= 0.0).all(), "Total thickness of layers should be positive"

#         botm_split = top_total - (thick_ratio_regis * total_thickness_layers).cumsum(dim="layer", skipna=False)
#         botm_other.loc[{"layer": layers}] = botm_split
#     return botm_other

# def _merge_layer_models(layer_model_regis, transition_model, koppeltabel_header_regis, koppeltabel_header_other, dfk, layer_model_bothalf, layer_names_regis_new_dict, layer_names_other_new_dict, layer_model_other_split, valid_other_layers, layer_model_top_split, botm_regis, botm_other):
#     """Merge the two layer models"""
#     logger.info("Merging the two layer models")

#     layer_model_top = xr.Dataset(
#         {
#             "botm": xr.where(
#                 valid_other_layers,
#                 layer_model_other_split["botm"],
#                 layer_model_top_split["botm"],
#             ),
#             "kh": xr.where(
#                 valid_other_layers,
#                 layer_model_other_split["kh"],
#                 layer_model_top_split["kh"],
#             ),
#             "kv": xr.where(
#                 valid_other_layers,
#                 layer_model_other_split["kv"],
#                 layer_model_top_split["kv"],
#             ),
#         },
#         attrs={
#             "extent": layer_model_regis.attrs["extent"],
#             "gridtype": layer_model_regis.attrs["gridtype"],
#         },
#     )

#     isadjusted_botm_regis = dfk[koppeltabel_header_regis].isin(list(layer_names_regis_new_dict.keys())).values
#     layer_model_top["botm"].loc[{"layer": isadjusted_botm_regis}] = botm_regis.loc[{"layer": isadjusted_botm_regis}]

#     isadjusted_botm_other = dfk[koppeltabel_header_other].isin(list(layer_names_other_new_dict.keys())).values
#     layer_model_top["botm"].loc[{"layer": isadjusted_botm_other}] = botm_other.loc[{"layer": isadjusted_botm_other}]

#     # introduce transition of layers
#     if transition_model is not None:
#         logger.info("Linear interpolation of transition region inbetween the two layer models")
#         transition_model_split = transition_model.sel(layer=dfk[koppeltabel_header_other].values).assign_coords(
#             layer=dfk["Regis_split"].values
#         )

#         for key in ["botm", "kh", "kv"]:
#             var = layer_model_top[key]
#             trans = transition_model_split[key]

#             for layer in var.layer.values:
#                 vari = var.sel(layer=layer)
#                 transi = trans.sel(layer=layer)
#                 if transi.sum() == 0:
#                     continue

#                 griddata_points = list(
#                     zip(
#                         vari.coords["x"].sel(icell2d=~transi).values,
#                         vari.coords["y"].sel(icell2d=~transi).values,
#                         strict=False,
#                     )
#                 )
#                 gridpoint_values = vari.sel(icell2d=~transi).values
#                 qpoints = list(
#                     zip(
#                         vari.coords["x"].sel(icell2d=transi).values,
#                         vari.coords["y"].sel(icell2d=transi).values,
#                         strict=False,
#                     )
#                 )
#                 qvalues = griddata(
#                     points=griddata_points,
#                     values=gridpoint_values,
#                     xi=qpoints,
#                     method="linear",
#                 )

#                 var.loc[{"layer": layer, "icell2d": transi}] = qvalues
#     else:
#         logger.info(
#             "No transition of the two layer models provided, resulting at sharp changes in kh, kv, and botm, at interface."
#         )

#     layer_model_out = xr.concat((layer_model_top, layer_model_bothalf), dim="layer")
#     layer_model_out["top"] = layer_model_regis["top"]

#     # categorize layers
#     # 1: regis
#     # 2: other
#     # 3: transition
#     cat_top = xr.where(valid_other_layers, 2, 1)

#     if transition_model is not None:
#         cat_top = xr.where(transition_model_split[["botm", "kh", "kv"]], 3, cat_top)

#     cat_botm = xr.ones_like(layer_model_bothalf[["botm", "kh", "kv"]], dtype=int)
#     cat = xr.concat((cat_top, cat_botm), dim="layer")
#     return layer_model_out,cat


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
