import logging
import os

import nlmod
import numpy as np
import scipy.interpolate
import xarray as xr

logger = logging.getLogger(__name__)


@nlmod.cache.cache_netcdf
def get_nhi_chloride_concentration(ds, data_path_nhi_chloride, interp_method="nearest"):
    """
    Get NHI chloride concentration and interpolate to modelgrid

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset
    data_path_nhi_chloride : str
        file path to chloride data
    interp_method : str, optional
        interpolation method, by default "nearest"

    Returns
    -------
    xr.DataArray
        interpolated chloride concentration
    """
    logger.info(
        f"Get NHI chloride concentration from {data_path_nhi_chloride} and interpolate to modelgrid"
    )
    fp_cl = os.path.join(data_path_nhi_chloride, "chloride_p50.nc")
    assert os.path.isfile(fp_cl), f"file {fp_cl} not found"
    cl = xr.open_dataset(fp_cl)["chloride_p50"].swap_dims(dict(layer="z"))

    # interpolate to modelgrid using nearest
    zc = cl.bottom + (cl.top - cl.bottom) / 2
    points = (
        zc.values,
        cl.y.values,
        cl.x.values,
    )
    values = cl.transpose("z", "y", "x").values

    thickness = nlmod.layers.calculate_thickness(ds)
    zci = ds.botm + thickness / 2
    xi = np.vstack(
        (
            zci.values.flatten(),
            zci.y.broadcast_like(zci).values.flatten(),
            zci.x.broadcast_like(zci).values.flatten(),
        )
    ).T

    qi = scipy.interpolate.interpn(
        points, values, xi, method=interp_method, bounds_error=False
    ).reshape(zci.shape)
    attrs = {
        "description": "Chloride concentration interpolated from NHI data",
        "units": "mg/l",
    }
    out = xr.DataArray(qi, coords=zci.coords, dims=zci.dims, attrs=attrs)

    # interpolate NaNs with method nearest (for Noord Holland only 1e-4% of cells)
    assert out.dims[0] == "layer", "Rewrite the following code with a transpose"
    for ilay in range(out.shape[0]):
        out.values[ilay] = nlmod.resample.fillnan_da(
            ds=ds, da=out.isel(layer=ilay), method="nearest"
        )

    return out
