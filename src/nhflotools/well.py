"""Functions to get wells data for PWN model."""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_wells_pwn_dataframe(data_path_wells_pwn, flow_product="median"):
    """Get wells dataframe for PWN model.

    Negative is extraction, positive is infiltration, MODFLOW convention. Data is in m3/h and function
    outputs in m3/day.

    Parameters
    ----------
    data_path_wells_pwn : str
        Path to the wells data.
    flow_product : str, optional
        Product to use for the flow, by default "median". Other products are currently not implemented.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with wells metadata and per-well flow ``Q`` [m3/day]. ``Q`` is the
        secundair median flow divided over its ``sec_nput`` wells, so the individual wells
        sum to the secundair total -- correct for the WEL package. For a grouped MAW well
        (``maw_from_df(group="sec_flow_tag")``) multiply ``Q`` by ``sec_nput`` first, because
        MAW applies a single combined RATE per group.
    """
    wdf = gpd.read_file(os.path.join(data_path_wells_pwn, "pumping_infiltration_wells.geojson"))
    wdf.index = wdf.locatie
    wdf["x"] = wdf.geometry.x
    wdf["y"] = wdf.geometry.y
    wdf["rw"] = 0.25  # for maw

    # Add concentration for infiltrations
    wdf["CONCENTRATION"] = 0.0

    # Get flows of entire secundair. (negative is extraction, positive is infiltration. data is in m3/h)
    flows = pd.read_feather(os.path.join(data_path_wells_pwn, "sec_flows.feather"))
    if flow_product == "timeseries":
        msg = "flow_product==timeseries not implemented"
        raise NotImplementedError(msg)
        # wdf["Q"] = wdf.sec_flow_tag and convert to m3/day
        # wdf = wdf.dropna(subset=["Q"])
        # Add `flows` as timeseries using nlmod.dims.time.dataframe_to_flopy_timeseries

        # return wdf, flows

    # If constant
    if flow_product == "median":
        constant_flow = flows.median(axis=0, numeric_only=True)
        wdf["sec_nput"] = pd.to_numeric(wdf["sec_nput"], errors="coerce")
        wdf["Q"] = wdf.sec_flow_tag.map(constant_flow) / wdf.sec_nput * 24.0
        wdf["Q"] = wdf["Q"].where(np.isfinite(wdf["Q"]) & (wdf["Q"] != 0))
        n_dropped = int(wdf["Q"].isna().sum())
        if n_dropped:
            logger.warning(
                "Dropping %d of %d wells without a nonzero secundair flow "
                "(unmapped sec_flow_tag, zero/NaN sec_nput, or zero median flow).",
                n_dropped,
                len(wdf),
            )
        return wdf.dropna(subset=["Q"])

    msg = f"flow_product {flow_product} not implemented"
    raise ValueError(msg)
