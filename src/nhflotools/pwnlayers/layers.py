import logging

import nlmod
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from .layers_read import read_pwn_data2

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

    TODO: Check that top is not merged and taken from layer_model 1.

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

        layer_model_regis contains all layers of REGISII. Only a subset of those layers
    connect to pwn. The connection is defined in the koppeltabel.

    Notes
    -----
    layer_model_tophalf, layer_model_pwn:
        The top layers are the layers that that are connected via the koppeltabel. It
        requires the koppeltabel to be valid for those layers and those layers should
        be present in layer_model_pwn.

    layer_model_pwn_split, layer_model_top_split:
        If multiple pwn layers are connected to a single regis layer, the regis layer
        is split into multiple layers. The thickness of the split layers is extrapolated
        from the pwn layers.
        If multiple regis layers are connected to a single pwn layer, the pwn layer
        is split into multiple layers. The thickness of the split layers is extrapolated
        from the regis layers.

    # Bottom layers
    layer_model_bothalf:
        The bottom layers are the layers that are not connected via the koppeltabel and
        the layers from layer_model_regis are used.
        layer_model_pwn_split is a ds

    # connection notes
    - nan values in layer_model_pwn means that those values are not defined in the pwn layer model.
        The values of layer_model_regis are used instead.
    - A thickness of zero in layer_model_pwn means that the layer is absent in the pwn layer model.

    """
    dfk = df_koppeltabel.copy()

    assert (
        layer_model_regis.attrs["extent"] == layer_model_pwn.attrs["extent"]
    ), "Extent of layer models are not equal"
    assert (
        layer_model_regis.attrs["gridtype"] == layer_model_pwn.attrs["gridtype"]
    ), "Gridtype of layer models are not equal"
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

    assert (
        not dfk[koppeltabel_header_regis].str.contains("_").any()
        and not dfk[koppeltabel_header_pwn].str.contains("_").any()
    ), "koppeltabel_header_regis and koppeltabel_header_pwn should not contain '_'"

    # Basename can occur multiple times if previously combined
    basenames_regis = [layer.split("_")[0] for layer in layer_model_regis.layer.values]
    basenames_pwn = [layer.split("_")[0] for layer in layer_model_pwn.layer.values]

    # Only select part of the table that appears in the two layer models
    dfk_mask = dfk[koppeltabel_header_regis].isin(basenames_regis) & dfk[
        koppeltabel_header_pwn
    ].isin(basenames_pwn)
    dfk = dfk[dfk_mask]

    logger.info("Combining layer models")

    # Construct a new layer index for the split REGIS layers
    dfk["Regis_split_index"] = (
        dfk.groupby(koppeltabel_header_regis).cumcount() + 1
    ).astype(str)
    dfk["Regis_split"] = dfk[koppeltabel_header_regis].str.cat(
        dfk["Regis_split_index"], sep="_"
    )
    dfk["Pwn_split_index"] = (
        dfk.groupby(koppeltabel_header_pwn).cumcount() + 1
    ).astype(str)
    dfk["Pwn_split"] = dfk[koppeltabel_header_pwn].str.cat(
        dfk["Pwn_split_index"], sep="_"
    )

    # Leave out lower REGIS layers
    top_regis_mask = np.array(
        [i in dfk[koppeltabel_header_regis].values for i in basenames_regis]
    )
    assert (
        np.diff(top_regis_mask).sum() == 1
    ), "REGIS layers should be consequtive from top to bottom."

    layer_model_tophalf = layer_model_regis.sel(layer=top_regis_mask)
    layer_model_bothalf = layer_model_regis.sel(layer=~top_regis_mask)

    # Count in how many layers the REGISII layers need to be split if previously never combined (default)
    split_counts_regis_def = (
        dfk.groupby(koppeltabel_header_regis, sort=False)[koppeltabel_header_regis]
        .count()
        .to_dict()
    )
    # New layer names for the split REGIS layers. If HLc is split in 6 layers,
    # HLc_1, HLc_2, ..., HLc_5 are created. HLc is renamed to HLc_6, as it has
    # the correct botm, and is therefore not considered new.
    layer_names_regis_new = np.concatenate(
        [
            [f"{k}_{vi}" for vi in range(1, v)]
            for k, v in split_counts_regis_def.items()
            if v > 1
        ]
    )
    # used for adjusting botm of split layers
    layer_names_regis_new_dict = {
        k: [f"{k}_{vi}" for vi in range(1, v)]
        for k, v in split_counts_regis_def.items()
        if v > 1
    }

    # Count in how many layers the PWN layers need to be split if previously never combined
    split_counts_pwn_def = (
        dfk.groupby(koppeltabel_header_pwn, sort=False)[koppeltabel_header_pwn]
        .count()
        .to_dict()
    )
    layer_names_pwn_new = np.concatenate(
        [
            [f"{k}_{vi}" for vi in range(1, v)]
            for k, v in split_counts_pwn_def.items()
            if v > 1
        ]
    )
    # used for adjusting botm of split layers
    layer_names_pwn_new_dict = {
        k: [f"{k}_{vi}" for vi in range(1, v)]
        for k, v in split_counts_pwn_def.items()
        if v > 1
    }

    # Split both layer models with evenly-split thickness
    layer_model_pwn_split = layer_model_pwn.sel(
        layer=np.concatenate([v * [k] for k, v in split_counts_pwn_def.items()])
    )
    layer_model_pwn_split = layer_model_pwn_split.assign_coords(
        layer=dfk["Regis_split"]
    )
    # Set botm of new layers to nan
    mask = dfk["Regis_split"][dfk["Pwn_split"].isin(layer_names_pwn_new)].values
    layer_model_pwn_split["botm"].loc[{"layer": mask}] = np.nan

    # layer_model_pwn_split where True, layer_model_top_split where False
    valid_pwn_layers = layer_model_pwn_split["botm"].notnull()

    if layer_model_regis.layer.str.contains("_").any():
        # TODO: if previously combined layer_model needs to be split for a second time
        split_counts_regis_cur = dict(
            zip(*np.unique(basenames_regis, return_counts=True))
        )
        assert all(
            v == split_counts_regis_cur[k] for k, v in split_counts_regis_def.items()
        ), "Previously combined REGIS layers should be split in the same number of layers as before."
        layer_model_top_split = layer_model_tophalf

        # # fill missing values in layer_model_pwn_split with layer_model_top_split
        # # After this step
        # layer_model_pwn_split = layer_model_pwn_split.combine_first(
        #     layer_model_top_split
        # )
    else:
        layer_model_top_split = layer_model_tophalf.sel(
            layer=np.concatenate([v * [k] for k, v in split_counts_regis_def.items()])
        )
        # Set botm of new layers to nan
        layer_model_top_split = layer_model_top_split.assign_coords(
            layer=dfk["Regis_split"]
        )
        layer_model_top_split["botm"].loc[{"layer": layer_names_regis_new}] = np.nan

    # extrapolate thickness of split layers
    thick_regis_top_split = nlmod.dims.layers.calculate_thickness(layer_model_top_split)
    thick_pwn_split = nlmod.dims.layers.calculate_thickness(layer_model_pwn_split)

    # best estimate thickness of unsplit regis layers
    elev_regis = xr.concat(
        (
            layer_model_regis["top"].expand_dims(layer=["mv"]),
            layer_model_regis["botm"],
        ),
        dim="layer",
    )
    top_regis = elev_regis.isel(layer=slice(-1)).assign_coords(
        layer=layer_model_regis.layer.values
    )
    elev_pwn = xr.concat(
        (
            layer_model_pwn["top"].expand_dims(layer=["mv"]),
            layer_model_pwn["botm"],
        ),
        dim="layer",
    )
    top_pwn = elev_pwn.isel(layer=slice(-1)).assign_coords(
        layer=layer_model_pwn.layer.values
    )

    """
    Connecting multiple PWN layers to one REGIS layer
    -------------------------------------------------

    In a previous step, extra layers were added to layer_model_top_split so that one REGIS layer
    can connect to multiple PWN layers. The botm is already at the correct elevation for where 
    the PWN layers are present (layer_model_pwn_split). The botm's of those layers is here set
    to the is now adjusted so that the thickness

    The total_thickness_layers is the sum of the thickness of the PWN layers that are connected
    to the one REGIS layer. The total thickness of the PWN layers is used if available, else the 
    total thickness of the REGIS layers is used. The thickness of the PWN layers is extrapolated 
    into the areas of the REGIS layers.
    
    The thick_ratio_pwn is the ratio of the thickness of the PWN layers with respect to total thickness
    that is extrapolated into the REGIS layer. The thick_ratio_pwn is used to calculate the elevations 
    of the botm of the newly split REGIS layers.
    """

    logger.info(
        f"Adjusting the botm of the newly split REGIS layers: {layer_names_regis_new}"
    )

    # Modifying layer_model_top_split["botm"] in place.
    botm_regis = layer_model_top_split["botm"].copy()
    for name, group in dfk.groupby(koppeltabel_header_regis):
        if name not in layer_names_regis_new_dict:
            # This REGIS layer is not split
            continue

        layers = group["Regis_split"].values
        layers_pwn = dfk["ASSUMPTION1"][dfk["Regis_split"].isin(layers)].values
        new_layers = layer_names_regis_new_dict[name]

        if all(i in layer_model_regis.layer.values for i in new_layers):
            # layer_model_regis is previously combined and already split
            logger.info(
                f"Previously combined REGIS layers: {name} are already split. "
                "The botm's for these layers are only adjusted where non-nan pwn "
                "data is provided and in the transition zone."
            )
            continue
        else:
            assert ~any(
                i in layer_model_regis.layer.values for i in new_layers
            ), "Previously combined REGIS layers should not be split for a second time."

        logger.info(
            f"About to adjust the botm of the newly split REGIS layers: {new_layers}. "
            f"{layers[-1]} already has the correct elevation"
        )

        # Top of combined layers
        top_total = xr.where(
            top_pwn.sel(layer=layers_pwn[0]).notnull(),
            top_pwn.sel(layer=layers_pwn[0]),
            xr.concat(
                (
                    top_pwn.sel(layer=layers_pwn).max(dim="layer"),
                    top_regis.sel(layer=name),
                ),
                dim="max",
            ).min(dim="max"),
        )

        botm_total = xr.where(
            layer_model_pwn["botm"].sel(layer=layers_pwn[-1]).notnull(),
            layer_model_pwn["botm"].sel(layer=layers_pwn[-1]),
            xr.concat(
                (
                    layer_model_pwn["botm"].sel(layer=layers_pwn).min(dim="layer"),
                    layer_model_regis["botm"].sel(layer=name),
                ),
                dim="min",
            ).min(dim="min"),
        )
        total_thickness_layers = top_total - botm_total

        # thick ratios of pwn layers that need to be extrapolated into the areas of the regis layers
        thick_ratio_pwn = xr.where(
            total_thickness_layers != 0.0,
            thick_pwn_split.sel(layer=layers) / total_thickness_layers,
            0.0,
        )

        for layer in new_layers:
            mask = thick_ratio_pwn.sel(layer=layer).notnull()  # locate valid values

            if mask.sum() == 0:
                logger.info(
                    f"Insufficient data in layer_model_pwn to extrapolate {layer} thickness into "
                    f"layer {name}. Splitting layers evenly."
                )
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

        # evenly fill up missing thick_ratio values. Same for all layers.
        fillna = (
            1 - thick_ratio_pwn.sum(dim="layer", skipna=True)
        ) / thick_ratio_pwn.isnull().sum(dim="layer", skipna=True)
        thick_ratio_pwn = thick_ratio_pwn.fillna(fillna)

        botm_split = top_total - (thick_ratio_pwn * total_thickness_layers).cumsum(
            dim="layer", skipna=False
        )
        botm_regis.loc[{"layer": layers}] = botm_split

    """
    Connecting one PWN layer to multiple REGIS layers
    -------------------------------------------------

    In a previous step, extra layers were added to layer_model_pwn_split so that one PWN layer
    can connect to multiple REGIS layers. Outside of the PWN layers, the botm of the multiple 
    REGIS layers is already at the correct elevation (layer_model_top_split). Inside of the PWN
    layers only the lower botm and the upper top are at the correct elevation. The elevation 
    intermediate botm's inside the PWN layers is set here.

    To estimate the thickness of the intermediate layers, the origional layer thickness over 
    total thickness ratio of the REGIS layers at the the location of the PWN layer is used 
    (thick_ratio_regis). This strategy is chosen so that the transition between PWN and REGIS
    layers is smooth.
    """
    logger.info("Adjusting the botm of the newly split PWN layers")
    del layers_pwn, layers
    botm_pwn = layer_model_pwn_split["botm"].copy()
    for name, group in dfk.groupby(koppeltabel_header_pwn):
        if name not in layer_names_pwn_new_dict:
            # This PWN layer is not split
            continue

        layers = group["Regis_split"].values
        new_layers = layer_names_pwn_new_dict[name]

        if any("_" in i for i in layer_model_regis.layer.values):
            layers_regis = group["Regis_split"].values
        else:
            layers_regis = group["Regis II v2.2"].values

        logger.info(
            f"About to adjust the botm of the newly split PWN layers: {new_layers}. "
            f"{layers[-1]} already has the correct elevation"
        )

        # thick ratios of regis layers that need to be extrapolated into the areas of the pwn layers
        total_thickness_layers_regis = thick_regis_top_split.sel(layer=layers).sum(
            dim="layer", skipna=False
        )
        thick_ratio_regis = xr.where(
            total_thickness_layers_regis != 0.0,
            thick_regis_top_split.sel(layer=layers) / total_thickness_layers_regis,
            0.0,
        )

        # Top of combined layers
        top_total = xr.where(
            top_pwn.sel(layer=name).notnull(),
            top_pwn.sel(layer=name),
            top_regis.sel(layer=layers_regis[0]),
        )

        botm_total = xr.where(
            layer_model_pwn["botm"].sel(layer=name).notnull(),
            layer_model_pwn["botm"].sel(layer=name),
            layer_model_regis["botm"].sel(layer=layers_regis[-1]),
        )
        total_thickness_layers = top_total - botm_total

        botm_split = top_total - (thick_ratio_regis * total_thickness_layers).cumsum(
            dim="layer", skipna=False
        )
        botm_pwn.loc[{"layer": layers}] = botm_split

    """Merge the two layer models"""
    logger.info("Merging the two layer models")

    layer_model_top = xr.Dataset(
        {
            "botm": xr.where(
                valid_pwn_layers,
                layer_model_pwn_split["botm"],
                layer_model_top_split["botm"],
            ),
            "kh": xr.where(
                valid_pwn_layers,
                layer_model_pwn_split["kh"],
                layer_model_top_split["kh"],
            ),
            "kv": xr.where(
                valid_pwn_layers,
                layer_model_pwn_split["kv"],
                layer_model_top_split["kv"],
            ),
        },
        attrs={
            "extent": layer_model_regis.attrs["extent"],
            "gridtype": layer_model_regis.attrs["gridtype"],
        },
    )

    isadjusted_botm_regis = (
        dfk["Regis II v2.2"].isin(list(layer_names_regis_new_dict.keys())).values
    )
    layer_model_top["botm"].loc[{"layer": isadjusted_botm_regis}] = botm_regis.loc[
        {"layer": isadjusted_botm_regis}
    ]

    isadjusted_botm_pwn = (
        dfk["ASSUMPTION1"].isin(list(layer_names_pwn_new_dict.keys())).values
    )
    layer_model_top["botm"].loc[{"layer": isadjusted_botm_pwn}] = botm_pwn.loc[
        {"layer": isadjusted_botm_pwn}
    ]

    # introduce transition of layers
    if transition_model_pwn is not None:
        logger.info(
            "Linear interpolation of transition region inbetween the two layer models"
        )
        transition_model_pwn_split = transition_model_pwn.sel(
            layer=dfk[koppeltabel_header_pwn].values
        ).assign_coords(layer=dfk["Regis_split"].values)

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
        logger.info(
            "No transition of the two layer models provided, resulting at sharp changes in kh, kv, and botm, at interface."
        )

    layer_model_out = xr.concat((layer_model_top, layer_model_bothalf), dim="layer")
    layer_model_out["top"] = layer_model_regis["top"]

    # categorize layers
    # 1: regis
    # 2: pwn
    # 3: transition
    cat_top = xr.where(valid_pwn_layers, 2, 1)

    if transition_model_pwn is not None:
        cat_top = xr.where(transition_model_pwn_split[["botm", "kh", "kv"]], 3, cat_top)

    cat_botm = xr.ones_like(layer_model_bothalf[["botm", "kh", "kv"]], dtype=int)
    cat = xr.concat((cat_top, cat_botm), dim="layer")

    return layer_model_out, cat


def get_pwn_layer_model(
    ds,
    datadir_mensink=None,
    datadir_bergen=None,
    length_transition=100.0,
    cachedir=None,
):
    pwn = read_pwn_data2(
        ds,
        datadir_mensink=datadir_mensink,
        datadir_bergen=datadir_bergen,
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

    # Bergen
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
    layer_model_bergen = xr.Dataset(
        {
            "top": pwn["top"],
            "kh": get_bergen_kh(pwn),
            "botm": get_bergen_botm(pwn),
            "kv": get_bergen_kv(pwn),
        },
        coords={"layer": list(translate_triwaco_bergen_names_to_index.keys())},
        attrs=layer_model_pwn_attrs,
    )
    transition_model_bergen = xr.Dataset(
        {
            "top": pwn["top_transition"],
            "botm": get_bergen_botm(pwn, mask=False, transition=True),
            "kh": get_bergen_kh(pwn, mask=False, transition=True),
            "kv": get_bergen_kv(pwn, mask=False, transition=True),
        },
        coords={"layer": list(translate_triwaco_bergen_names_to_index.keys())},
    )

    return (
        layer_model_pwn,
        transition_model_pwn,
        layer_model_bergen,
        transition_model_bergen,
    )


def get_bergen_thickness(data, mask=False, transition=False):
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
    botm = get_bergen_botm(data, mask=mask, transition=transition)

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
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    else:
        return out


def get_bergen_kh(data, mask=False, anisotropy=5.0, transition=False):
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
        out = get_bergen_thickness(data, mask=True, transition=False).rename("kv")
        out[dict(layer=1)] *= data["BER_C1A_mask"]
        out[dict(layer=3)] *= data["BER_C1B_mask"]
        out[dict(layer=5)] *= data["BER_C1C_mask"]
        out[dict(layer=7)] *= data["BER_C1D_mask"]
        out[dict(layer=9)] *= data["BER_C2_mask"]

    elif transition:
        # Valid value if valid thickness or valid BER_C
        out = get_bergen_thickness(data, mask=True, transition=False).rename("kv")
        out[dict(layer=1)] |= data["BER_C1A_mask"]
        out[dict(layer=3)] |= data["BER_C1B_mask"]
        out[dict(layer=5)] |= data["BER_C1C_mask"]
        out[dict(layer=7)] |= data["BER_C1D_mask"]
        out[dict(layer=9)] |= data["BER_C2_mask"]

    else:
        thickness = get_bergen_thickness(data, mask=mask, transition=transition)

        # 1 where out_mask is True, nan where out_mask is False
        out_mask = get_bergen_thickness(data, mask=True, transition=False)
        out = thickness.where(~out_mask, other=1.0).rename("kv")

        out[dict(layer=[0, 2, 4, 6, 8])] *= [8.0, 7.0, 12.0, 15.0, 20.0]
        out[dict(layer=1)] = thickness[dict(layer=1)] / data["BER_C1A"] * anisotropy
        out[dict(layer=3)] = thickness[dict(layer=3)] / data["BER_C1B"] * anisotropy
        out[dict(layer=5)] = thickness[dict(layer=5)] / data["BER_C1C"] * anisotropy
        out[dict(layer=7)] = thickness[dict(layer=7)] / data["BER_C1D"] * anisotropy
        out[dict(layer=9)] = thickness[dict(layer=9)] / data["BER_C2"] * anisotropy

    return out


def get_bergen_kv(data, mask=False, anisotropy=5.0, transition=False):
    """
    Calculate the hydraulic conductivity (KV) for different aquifers and aquitards.

    Parameters:
        data (xarray.Dataset): Dataset containing the necessary variables for calculation.
        mask (bool, optional): Flag indicating whether to apply a mask to the data. Defaults to False.
        anisotropy (float, optional): Anisotropy factor for adjusting the hydraulic conductivity. Defaults to 5.0.

    Returns:
        xarray.DataArray: Array containing the calculated hydraulic conductivity values for each layer.

    Example:
        # Calculate hydraulic conductivity values without applying a mask
        kv_values = get_kv(data)

        # Calculate hydraulic conductivity values with a mask applied
        kv_values_masked = get_kv(data, mask=True)

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

    out = kh / anisotropy

    return out


def get_bergen_botm(data, mask=False, transition=False):
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
    )
    if mask:
        return ~np.isnan(out)
    elif transition:
        mask = get_bergen_botm(data, mask=True, transition=False)
        transition = np.isnan(out)
        check = mask.astype(int) + transition.astype(int)
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    else:
        return out


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
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
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
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
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
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
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
        assert (check <= 1).all(), "Transition cells should not overlap with mask."
        return transition
    else:
        return out
