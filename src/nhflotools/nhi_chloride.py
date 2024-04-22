import logging
import os

import nlmod
import xarray as xr

logger = logging.getLogger(__name__)


@nlmod.cache.cache_netcdf(coords_3d=True)
def get_nhi_chloride_concentration(ds, data_path_nhi_chloride):
    """
    Get NHI chloride concentration and interpolate to modelgrid.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset
    data_path_nhi_chloride : str
        file path to chloride data

    Returns
    -------
    xr.DataArray
        interpolated chloride concentration
    """
    logger.info(f"Get NHI chloride concentration from {data_path_nhi_chloride} and interpolate to modelgrid")
    fp_cl = os.path.join(data_path_nhi_chloride, "chloride_p50.nc")

    with xr.open_dataset(fp_cl) as fh:
        cl = fh["chloride_p50"].transpose("layer", "y", "x").load()

    # cli has x and y of ds but layer of cl
    cli = cl.interp(x=ds.x, y=ds.y, method="nearest").drop_vars(["dy", "dx", "percentile"])

    da = nlmod.layers.aggregate_by_weighted_mean_to_ds(
        ds, 
        xr.Dataset({"p50": cli}), 
        "p50")

    da.values[0] = xr.where(ds["northsea"] == 1, 18_000, da.values[0])

    for ilay in range(da.layer.size):
        da.values[ilay] = nlmod.resample.fillnan_da(ds=ds, da=da.isel(layer=ilay), method="nearest")

    da.attrs = {
        "description": "Chloride concentration interpolated from NHI data",
        "units": "mg/l",
    }

    if da.isnull().any():
        logger.warning(f"Interpolated NHI chloride concentration contains NaNs")

    return da
