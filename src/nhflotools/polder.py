import itertools
from collections import Counter, defaultdict

import nlmod
import numpy as np
import scipy.interpolate as si
import xarray as xr


def drn_from_waterboard_data(ds, gwf, wb="Hollands Noorderkwartier", cbot=1.0):
    """Create DRN package from waterboard data.

    Het oppervlaktewater in de polders is vlakdekkend geschematiseerd op basis van
    peilgebieden van Hoogheemraadschap Hollands Noorderkwartier (HHNK). Hierbij
    zijn dus geen afzonderlijke sloten beschouwd. Voor het peil is het gemiddelde
    van het zomer- en winterpeil aangenomen. De conductance is bepaald met het
    oppervlak van een polder per cel, gedeeld door een bodemweerstand van 1 dag. Dit
    oppervlaktewater is via de Drain package (DRN) in het model verwerkt en kan dus
    alleen water afvoeren. Daar waar geen polderpeilen zijn vastgelegd door HHNK, is
    het gemiddelde niveau van het maaiveld toegepast om drainage op maaiveld in het
    model op te nemen.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    wb : str, optional
        Waterboard, by default "Holl
    cbot : float, optional
        Conductance of the drains, by default 1.0.

    Returns
    -------
    flopy.modflow.ModflowGwfdrn
        DRN package.
    """
    gdf = nlmod.read.waterboard.download_data(wb=wb, data_kind="level_areas", extent=ds.extent, verify=False)

    if gdf.empty:
        return None

    # Rename duplicate indices
    counts = Counter(gdf.index)
    suffix_counter = defaultdict(lambda: itertools.count(1))
    index2 = [elem if counts[elem] == 1 else elem + f"_{next(suffix_counter[elem])}" for elem in gdf.index]
    gdf.index = index2

    gdf_grid = nlmod.grid.gdf_to_grid(gdf.loc[:, ["summer_stage", "winter_stage", "geometry"]], gwf)
    fields_methods = {
        "summer_stage": "area_weighted",
        "winter_stage": "area_weighted",
    }
    celldata = nlmod.grid.aggregate_vector_per_cell(gdf_grid, fields_methods=fields_methods)
    celldata["x"] = ds.sel(icell2d=celldata.index)["x"]
    celldata["y"] = ds.sel(icell2d=celldata.index)["y"]
    celldata["area"] = ds.sel(icell2d=celldata.index)["area"]
    celldata["cond"] = celldata["area"] / cbot
    celldata["elev"] = celldata.loc[:, ["summer_stage", "winter_stage"]].mean(axis=1)

    # There are some gaps in the summer_stage and winter_stage data, fill these with nearest
    nv = celldata["elev"].isnull().values
    celldata.loc[nv, "elev"] = si.griddata(
        celldata[["x", "y"]][~nv],
        celldata["elev"][~nv],
        xi=celldata[["x", "y"]][nv],
        method="nearest",
    )

    drn_elev = xr.full_like(ds.top, np.nan)
    drn_elev.loc[{"icell2d": celldata.index}] = celldata.elev.values
    drn_elev = xr.where(drn_elev.isnull(), ds["ahn"], drn_elev)
    drn_cond = xr.full_like(ds.top, 0.0)
    drn_cond.loc[{"icell2d": celldata.index}] = celldata.cond.values
    drn_cond = xr.where(drn_cond.isnull() & ds["ahn"].notnull(), ds.area / cbot, drn_cond)
    drn_cond = drn_cond.clip(max=ds.area.max())
    ds["drn_elev"] = drn_elev
    ds["drn_cond"] = drn_cond

    return nlmod.gwf.drn(ds, gwf, elev="drn_elev", cond="drn_cond")
