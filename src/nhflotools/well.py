"""Functions to get wells data for PWN model."""

import logging
import os

import geopandas as gpd
import nlmod
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


def get_wells_tata_dataframes(
    data_path_wells_tata,
    ds,
    *,
    kd_zoet_layer=100.0,
    cl_max_zoet_layer=1000.0,
):
    """Get Tata Steel saltwater and freshwater wells for the PWN model.

    Parameters
    ----------
    data_path_wells_tata : str or path-like
        Path to the Tata wells data directory.
    ds : xarray.Dataset
        Model dataset with ``x``, ``y``, ``top``, ``botm``, ``kh``, and
        ``chloride`` data. ``thickness`` is added to this dataset, matching the
        original modelscript behavior.
    kd_zoet_layer : float, optional
        Minimum transmissivity-like value ``kh * thickness`` [m2/d] used to
        place freshwater wells, by default 100.0.
    cl_max_zoet_layer : float, optional
        Chloride concentration [mg/l] above which a warning is emitted for the
        selected freshwater well layer, by default 1000.0.

    Returns
    -------
    tuple of gpd.GeoDataFrame
        Saltwater and freshwater Tata well GeoDataFrames. Both include ``x``,
        ``y``, ``Q``, and ``CONCENTRATION``. Freshwater wells also include
        vertically aligned ``top`` and ``botm`` screen elevations.
    """
    gdf_tata_zout = gpd.read_file(
        os.path.join(data_path_wells_tata, "tata_zoutwaterbronnen.geojson"),
        driver="GeoJSON",
    )
    gdf_tata_zout["x"] = gdf_tata_zout["geometry"].x
    gdf_tata_zout["y"] = gdf_tata_zout["geometry"].y
    gdf_tata_zout["Q"] = -gdf_tata_zout["Q_m3/d"] / len(gdf_tata_zout)
    gdf_tata_zout["CONCENTRATION"] = 0.0

    ds["thickness"] = nlmod.dims.calculate_thickness(ds)
    kd = ds["kh"] * ds["thickness"]

    gdf_tata_zoet = gpd.read_file(
        os.path.join(data_path_wells_tata, "tata_zoetwaterbronnen.geojson"),
        driver="GeoJSON",
    )
    gdf_tata_zoet["x"] = gdf_tata_zoet["geometry"].x
    gdf_tata_zoet["y"] = gdf_tata_zoet["geometry"].y
    gdf_tata_zoet["Q"] = -gdf_tata_zoet["Q_m3/d"] / len(gdf_tata_zoet)
    gdf_tata_zoet["CONCENTRATION"] = 0.0

    gdf_tata_zoet["botm"] = np.nan
    gdf_tata_zoet["top"] = np.nan

    for name, row in gdf_tata_zoet.iterrows():
        inearest = np.sqrt((ds.x - row["x"]) ** 2 + (ds.y - row["y"]) ** 2).argmin(dim="icell2d")
        lay = np.where(kd.isel(icell2d=inearest) > kd_zoet_layer)[0][0]
        cl = ds["chloride"].isel(icell2d=inearest)[lay]

        if cl > cl_max_zoet_layer:
            msg = "Unable to place zoetwaterbron in fresh aquifer. => Placing TATA zoetwaterbron in saline aquifer."
            msg += f" Cl: {cl:.2f} mg/l, kd: {kd.isel(icell2d=inearest)[lay]:.2f} m2/d, ilayer: {lay}, "
            msg += f"layername: {ds.layer[lay].item()}, xwell: {row['x']}, ywell: {row['y']}, modelextent: {ds.extent}"
            logger.warning(msg)

        gdf_tata_zoet.loc[name, "botm"] = ds["botm"].isel(icell2d=inearest)[lay] + 0.001
        if lay == 0:
            gdf_tata_zoet.loc[name, "top"] = ds["top"].isel(icell2d=inearest).item() - 0.001
        else:
            gdf_tata_zoet.loc[name, "top"] = ds["botm"].isel(icell2d=inearest)[lay - 1] - 0.001

    return gdf_tata_zout, gdf_tata_zoet
