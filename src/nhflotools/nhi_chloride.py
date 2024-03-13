import os

import nlmod
import numpy as np
import scipy.interpolate
import xarray as xr


@nlmod.cache.cache_netcdf
def get_nhi_chloride_concentration(ds, fp_cl, interp_method="nearest"):
    """
    Get NHI chloride concentration and interpolate to modelgrid

    Parameters
    ----------
    ds : xarray.Dataset
        model dataset
    fp_cl : str
        file path to chloride data
    interp_method : str, optional
        interpolation method, by default "nearest"

    Returns
    -------
    xr.DataArray
        interpolated chloride concentration
    """
    assert os.path.isfile(fp_cl), f"file {fp_cl} not found"
    cl = xr.open_dataset(fp_cl)["chloride_p50"].swap_dims(dict(layer="z"))

    # interpolate to modelgrid using nearest
    zc = cl.bottom + (cl.top - cl.bottom) / 2
    points = np.vstack(
        (
            cl.x.broadcast_like(cl).values.flatten(),
            cl.y.broadcast_like(cl).values.flatten(),
            zc.broadcast_like(cl).values.flatten(),
        )
    ).T
    values = cl.values.flatten()

    thickness = nlmod.layers.calculate_thickness(ds)
    zci = ds.botm + thickness / 2
    xi = np.vstack(
        (
            zci.x.broadcast_like(zci).values.flatten(),
            zci.x.broadcast_like(zci).values.flatten(),
            zci.values.flatten(),
        )
    ).T

    qi = scipy.interpolate.griddata(points, values, xi, method=interp_method).reshape(
        zci.shape
    )
    out = xr.DataArray(qi, coords=zci.coords, dims=zci.dims)
    return out
