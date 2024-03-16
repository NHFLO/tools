import os

import geopandas as gpd
import pandas as pd


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
        GeoDataFrame with wells metadata and flow [m3/day].
    gpd.GeoDataFrame, optional
        GeoDataFrame with flow timeseries [m3/day]. Not implemented yet.
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
        raise NotImplementedError("flow_product==timeseries not implemented")
        # wdf["Q"] = wdf.sec_flow_tag and convert to m3/day
        # wdf = wdf.dropna(subset=["Q"])
        # Add `flows` as timeseries using nlmod.dims.time.dataframe_to_flopy_timeseries

        # return wdf, flows

    # If constant
    if flow_product == "median":
        constant_flow = flows.median(axis=0)
        wdf["sec_nput"] = pd.to_numeric(wdf["sec_nput"], errors="coerce")
        wdf["Q"] = wdf.sec_flow_tag.map(constant_flow) / wdf.sec_nput * 24.0
        wdf = wdf.dropna(subset=["Q"])
        return wdf

    raise ValueError(f"flow_product {flow_product} not implemented")
