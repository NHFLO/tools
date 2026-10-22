"""Read FluxPark drainage and groundwater ET demand onto an NHFLO model grid."""

import logging
from itertools import pairwise
from pathlib import Path

import nlmod
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


@nlmod.cache.cache_netcdf(coords_2d=True, coords_time=True)
def get_fluxpark_recharge(ds, data_path_fluxpark, start, extrapolate_in_time=True):  # noqa: FBT002
    """Crop FluxPark fluxes and average onto model periods and grid cells.

    Parameters
    ----------
    ds : xarray.Dataset
        Model grid and time coordinates, including quadtree vertex grids.
    data_path_fluxpark : str or pathlib.Path
        Directory containing ``rch_fluxpark.nc`` with ``drain_mm_d`` and
        ``trans_def_pot_mm_d``, in mm/d, on a regular EPSG:28992 grid with
        dimensions ``time``, ``y`` and ``x``.
    start : str or pandas.Timestamp
        Model start date (``ds.time.start``). Passed explicitly because nlmod's
        cache hash excludes time attributes, including the model start date.
    extrapolate_in_time : bool, optional
        Fill periods containing no source timestamps with the source rate nearest
        their period-end timestamp. Applies to internal gaps and dates outside
        the source record. Default is True; False leaves empty periods as NaN.
        Missing spatial values in observed periods are not filled.

    Returns
    -------
    xarray.Dataset
        ``recharge`` (soil drainage) and ``evaporation`` (potential transpiration
        deficit), in m/d, at ``ds.time`` (period ends). Daily source timestamps
        denote interval starts and are averaged in left-closed, right-open model
        periods. Partly observed periods average only available measurements.
        Source/model cell overlap areas provide the spatial weights.

    Notes
    -----
    The transpiration deficit is the maximum additional demand on groundwater,
    not total potential evaporation. Land-fraction scaling, masks and the
    head-dependent EVT response belong in the model's package configuration.
    These soil fluxes must not be used as lake rainfall or open-water evaporation.
    Cropping retains source pixels overlapping the model extent, including pixels
    whose centers lie just outside it. Each variable and model period is reduced
    separately, using chunks of at most five days by 256 by 256 pixels and a
    synchronous scheduler. Only one period's source raster is materialized for
    spatial resampling; the final model-grid output remains in memory.
    """
    logger.info("Read FluxPark forcing from %s and average onto model periods and grid", data_path_fluxpark)
    fluxpark_file = Path(data_path_fluxpark) / "rch_fluxpark.nc"
    if not fluxpark_file.exists():
        msg = f"FluxPark path {fluxpark_file} does not exist"
        raise ValueError(msg)

    with xr.open_dataset(fluxpark_file, cache=False) as source:
        xmin, xmax, ymin, ymax = nlmod.grid.get_extent(ds)
        fluxpark_raw = (
            source[["drain_mm_d", "trans_def_pot_mm_d"]]
            .transpose("time", "y", "x")
            .sortby("x")
            .sortby("y", ascending=False)
            .rio.write_crs(28992)
            .rio.clip_box(xmin, ymin, xmax, ymax, allow_one_dimensional_raster=True)
            .sortby("time")
        )
        for name, flux in fluxpark_raw.data_vars.items():
            if flux.attrs.get("units") != "mm/d":
                msg = f"Expected FluxPark {name} in mm/d, got {flux.attrs.get('units')!r}"
                raise ValueError(msg)

        source_times = fluxpark_raw.indexes["time"]
        boundaries = source_times.searchsorted(pd.DatetimeIndex([pd.Timestamp(start), *ds.time.values]))
        rates = {}
        for input_name, output_name in {"drain_mm_d": "recharge", "trans_def_pot_mm_d": "evaporation"}.items():
            for index, (left, right) in enumerate(pairwise(boundaries)):
                if left < right:
                    period = (
                        fluxpark_raw[input_name]
                        .isel(time=slice(left, right))
                        .chunk({"time": 5, "y": 256, "x": 256})
                        .mean("time", keep_attrs=True)
                        .compute(scheduler="synchronous")
                    )
                elif extrapolate_in_time:
                    period = (
                        fluxpark_raw[input_name]
                        .sel(time=ds.time.values[index], method="nearest")
                        .chunk({"y": 256, "x": 256})
                        .compute(scheduler="synchronous")
                    )
                else:
                    period = xr.full_like(fluxpark_raw[input_name].isel(time=0), float("nan"), dtype=float)
                grid = ds.isel(time=slice(index, index + 1))
                period = (period.drop_vars("time", errors="ignore") / 1000.0).expand_dims(time=grid.time)
                resampled = nlmod.resample.structured_da_to_ds(period, grid, method="average")
                if index == 0:
                    rates[output_name] = resampled.reindex(time=ds.time)
                else:
                    rates[output_name][{"time": slice(index, index + 1)}] = resampled
            rates[output_name].attrs = {"units": "m/d"}
    return xr.Dataset(rates)
