"""Read FluxPark drainage and groundwater ET demand onto an NHFLO model grid."""

import logging
from pathlib import Path

import nlmod
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


@nlmod.cache.cache_netcdf(coords_2d=True, coords_time=True)
def get_fluxpark_recharge(ds, data_path_fluxpark, start):
    """Read daily FluxPark fluxes for the model dates and area-average onto its grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Model grid and time coordinates, including quadtree vertex grids.
    data_path_fluxpark : str or pathlib.Path
        NetCDF file containing ``drain_mm_d`` and ``trans_def_pot_mm_d``, in mm/d,
        on a regular EPSG:28992 grid with dimensions ``time``, ``y`` and ``x``.
    start : str or pandas.Timestamp
        Model start date (``ds.time.start``). Passed explicitly because nlmod's
        cache hash excludes time attributes, including the model start date.

    Returns
    -------
    xarray.Dataset
        ``recharge`` (soil drainage) and ``evaporation`` (potential transpiration
        deficit), in m/d, with daily interval-start timestamps in
        ``[start, ds.time[-1])``. Source/model cell overlap areas provide the
        spatial weights. Dates outside the source record use the nearest observed
        daily rate. No temporal averaging is performed.

    Notes
    -----
    The transpiration deficit is the maximum additional demand on groundwater,
    not total potential evaporation. Land-fraction scaling, masks and the
    head-dependent EVT response belong in the model's package configuration.
    These soil fluxes must not be used as lake rainfall or open-water evaporation.
    Keep the daily output separate from the model's stress-period time coordinate.
    """
    logger.info("Read daily FluxPark forcing from %s and area-average onto the model grid", data_path_fluxpark)
    fluxpark_file = Path(data_path_fluxpark)
    if not fluxpark_file.exists():
        msg = f"FluxPark path {fluxpark_file} does not exist"
        raise ValueError(msg)

    with xr.open_dataset(fluxpark_file) as fluxpark_raw:
        days = pd.date_range(start, ds.time.values[-1], freq="D", inclusive="left")
        fluxpark = (
            fluxpark_raw[["drain_mm_d", "trans_def_pot_mm_d"]]
            .reindex(time=days, method="nearest")
            .transpose("time", "y", "x")
            .sortby("x")
            .sortby("y", ascending=False)
        )
        for name, flux in fluxpark.data_vars.items():
            if flux.attrs.get("units") != "mm/d":
                msg = f"Expected FluxPark {name} in mm/d, got {flux.attrs.get('units')!r}"
                raise ValueError(msg)

        daily_grid = ds.drop_dims("time").assign_coords(time=fluxpark.time)
        rates = {}
        for input_name, output_name in {"drain_mm_d": "recharge", "trans_def_pot_mm_d": "evaporation"}.items():
            rates[output_name] = nlmod.resample.structured_da_to_ds(
                fluxpark[input_name] / 1000.0, daily_grid, method="average"
            )
            rates[output_name].attrs = {"units": "m/d"}
    return xr.Dataset(rates)
