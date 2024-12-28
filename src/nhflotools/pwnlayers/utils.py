import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

def fix_missings_botms_and_min_layer_thickness(ds):
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
    if ds["top"].isnull().any():
        msg = "Top should not contain nan values"
        raise ValueError(msg)

    out = xr.concat((ds["top"].expand_dims(dim={"layer": ["mv"]}, axis=0), ds["botm"]), dim="layer")

    # Use ffill here to fill the nan's with the previous layer. Layer thickness is zero for non existing layers
    out = out.ffill(dim="layer")
    out.values = np.minimum.accumulate(out.values, axis=out.dims.index("layer"))
    topbotm_fixed = out.isel(layer=slice(1, None)).transpose("layer", "icell2d")
    ds["botm"].values = topbotm_fixed.values

    # inform
    ncell, nisnull = ds["botm"].size, ds["botm"].isnull().sum()
    nfixed = (~np.isclose(ds.botm, topbotm_fixed)).sum()

    logger.info(
        "Fixed %.1f%% missing botms using downward fill. Shifted %.1f%% botms to ensure all layers have a positive thickness, assuming more info is in the upper layer.",
        nisnull / ncell * 100.0,
        (nfixed - nisnull) / ncell * 100.0
    )
