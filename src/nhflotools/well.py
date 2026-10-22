"""Functions to get wells data for PWN model."""

import logging
import os

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_wells_pwn_dataframe(data_path_wells_pwn, flow_product="median", *, grouped=False):
    """Get wells dataframe for PWN model.

    Negative is extraction, positive is infiltration, MODFLOW convention. Data is in m3/h and function
    outputs in m3/day.

    Parameters
    ----------
    data_path_wells_pwn : str
        Path to the wells data.
    flow_product : str, optional
        ``"median"`` (default) for constant rates or ``"timeseries"`` for TS6 names.
    grouped : bool, optional
        Return full secundair rates for ``maw_from_df(group="sec_flow_tag")``.
        Default False divides rates by ``sec_nput`` for individual WEL wells.

    Returns
    -------
    gpd.GeoDataFrame or tuple of (gpd.GeoDataFrame, pd.DataFrame)
        Wells with numeric ``Q`` [m3/day] for medians. For time series, ``Q`` contains
        names referencing columns in the returned datetime-indexed rates [m3/day].
        Use :func:`resample_well_flows` before attaching these to a MODFLOW package.
        Time series retain temporarily idle and zero-median wells.
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
    wdf["sec_nput"] = pd.to_numeric(wdf["sec_nput"], errors="coerce")
    if flow_product == "timeseries":
        return _get_pwn_flow_timeseries(wdf, flows, grouped=grouped)

    # If constant
    if flow_product == "median":
        constant_flow = flows.median(axis=0, numeric_only=True)
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
        wdf = wdf.dropna(subset=["Q"])
        if grouped:
            wdf["Q"] *= wdf["sec_nput"]
        return wdf

    msg = f"flow_product {flow_product} not implemented"
    raise ValueError(msg)


def _get_pwn_flow_timeseries(wdf, flows, *, grouped):
    """Prepare PWN rates and safe TS6 names without filtering on median flow."""
    flows = flows.set_index(pd.to_datetime(flows["ophaal tijdstip"])).select_dtypes(include="number").sort_index()
    if flows.index.hasnans or flows.index.has_duplicates:
        msg = "PWN flow timestamps must be nonmissing and unique"
        raise ValueError(msg)
    if np.isinf(flows.to_numpy()).any():
        msg = "PWN flow rates must not contain infinite values"
        raise ValueError(msg)
    valid = (
        wdf["sec_flow_tag"].isin(flows.columns[flows.notna().any()])
        & np.isfinite(wdf["sec_nput"])
        & (wdf["sec_nput"] > 0)
    )
    if not valid.all():
        logger.warning(
            "Dropping %d of %d wells without a usable secundair time series or positive sec_nput.",
            int((~valid).sum()),
            len(wdf),
        )
    wdf = wdf.loc[valid].copy()
    wdf["Q"] = ""
    rates = pd.DataFrame(index=flows.index)
    for number, ((tag, count), wells) in enumerate(wdf.groupby(["sec_flow_tag", "sec_nput"], sort=False)):
        name = f"pwn_{number}"
        if grouped:
            # One RATE per MAW group, even if the metadata have different well counts.
            name = f"pwn_{flows.columns.get_loc(tag)}"
        wdf.loc[wells.index, "Q"] = name
        rates[name] = flows[tag] * (24.0 if grouped else 24.0 / count)
    return wdf, rates


def resample_well_flows(flows, ds):
    """Average PWN flow observations over model stress periods.

    Parameters
    ----------
    flows : pd.DataFrame
        Datetime-indexed well rates [m3/day], as returned by
        ``get_wells_pwn_dataframe(..., flow_product="timeseries")``.
    ds : xr.Dataset
        Model dataset with datetime period ends in ``time`` and ``time.start``.

    Returns
    -------
    pd.DataFrame
        Arithmetic means of available observations in each half-open
        ``[period start, period end)`` interval, labelled at ``ds.time``.
        Empty periods use the observation nearest their end, independently
        per column, including extrapolation beyond the source record.
        Applies equally to the steady spin-up and transient periods.
        Attach using ``dataframe_to_flopy_timeseries`` with
        ``shift_label_from_right_to_left=True`` and stepwise interpolation.
    """
    ends = pd.DatetimeIndex(ds.time.values)
    starts = ends.insert(0, pd.Timestamp(ds.time.start))[:-1]
    means = pd.DataFrame(
        [
            flows.loc[(flows.index >= start) & (flows.index < end)].mean()
            for start, end in zip(starts, ends, strict=True)
        ],
        index=ends,
        columns=flows.columns,
    )
    for name in flows:
        observed = flows[name].dropna().sort_index()
        if observed.empty or not np.isfinite(observed).all():
            msg = f"Well time series {name!r} must contain finite observations"
            raise ValueError(msg)
        means[name] = means[name].fillna(observed.reindex(ends, method="nearest"))
    return means


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
