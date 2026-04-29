import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def fix_missings_botms_and_min_layer_thickness(*, top=None, botm=None):
    """
    Fix missing botms and ensure all layers have a positive thickness.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the layer model with top and botm.

    Raises
    ------
    ValueError
        If top contains nan values.
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
    ncell, nisnull = botm.size, botm.isnull().sum()
    nfixed = (~np.isclose(botm, botm_fixed)).sum()
    logger.info(
        "Fixed %.1f%% missing botms using downward fill. Shifted %.1f%% botms to ensure all layers have a positive thickness, assuming more info is in the upper layer.",
        nisnull / ncell * 100.0,
        (nfixed - nisnull) / ncell * 100.0,
    )
    return botm_fixed
