"""Regional polder drainage (HHNK peilgebieden) as a MODFLOW 6 DRN package."""

import itertools
from collections import Counter, defaultdict

import nlmod
import numpy as np
import scipy.interpolate as si
import xarray as xr


def drn_from_waterboard_data(ds, gwf, wb="Hollands Noorderkwartier", cbot=1.0, exclude=None):
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
        Waterboard, by default "Hollands Noorderkwartier".
    cbot : float, optional
        Bottom resistance of the drains [days], by default 1.0. The per-cell conductance
        is ``cell_area / cbot``.
    exclude : xarray.DataArray or None, optional
        Boolean mask or open-water fraction (0-1) over ``icell2d``. The conductance is
        scaled by the drainable land share ``1 - exclude`` (a boolean ``True`` scales by
        0), so a reach is dropped at fully open-water cells — whether its stage comes from
        HHNK peilgebied data or from the maaiveld fallback (``nlmod.gwf.drn`` masks on
        ``cond > 0``) — and only drains the land share elsewhere. The default None applies
        no exclusion and is fully backward-compatible. Used to hand carved lake cells over
        to a dedicated stage boundary and to keep open water out of the drained area.

    Returns
    -------
    flopy.mf6.ModflowGwfdrn or None
        DRN package, or None when no level-area data intersects the model extent.
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
    drn_cond = xr.full_like(ds.top, np.nan)
    drn_cond.loc[{"icell2d": celldata.index}] = celldata.cond.values
    fallback_mask = drn_cond.isnull() & ds["ahn"].notnull()
    if "northsea" in ds:
        fallback_mask &= ds["northsea"] == 0
    drn_cond = xr.where(fallback_mask, ds.area / cbot, drn_cond)
    ds["drn_elev"] = drn_elev
    ds["drn_cond"] = drn_cond

    if exclude is not None:
        # Scale the conductance by the drainable land share; nlmod.gwf.drn builds a reach
        # wherever cond > 0, so a zero land share drops the reach whether its stage came
        # from the celldata assignment or the maaiveld fallback above.
        land = (1.0 - exclude.astype(float)).clip(min=0.0)
        ds["drn_cond"] *= land
        ds["drn_elev"] = ds["drn_elev"].where(land > 0)

    return nlmod.gwf.drn(ds, gwf, elev="drn_elev", cond="drn_cond")
