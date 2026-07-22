"""Tests for nhflotools.polder.

The polder DRN gains an optional ``exclude`` mask so carved lake cells can be handed over
to a dedicated stage boundary. The reach must be dropped whether its stage comes from real
HHNK peilgebied data or from the maaiveld fallback, which is why the exclusion nulls the
conductance after the celldata assignment rather than only editing the fallback mask.
"""

import geopandas as gpd
import nlmod
import xarray as xr
from shapely.geometry import box

from nhflotools import polder

# Cell ids of the shared 3x3 disv grid.
CELL_REAL_STAGE = 4  # centre cell, covered by the mocked HHNK level area (real peilgebied stage)
CELL_FALLBACK = 0  # a cell with no HHNK stage -> maaiveld fallback drain


def _reach_cells(drn):
    """Return the set of icell2d values that carry a DRN reach."""
    if drn is None:
        return set()
    return {rec["cellid"][-1] for rec in drn.stress_period_data.data[0]}


def _run_drn(monkeypatch, ds, gwf, exclude):
    """Run drn_from_waterboard_data with a real HHNK level area over the centre cell mocked in."""
    ds["ahn"] = xr.full_like(ds["top"], 1.0)
    ds["northsea"] = xr.zeros_like(ds["top"]).astype(int)
    level_areas = gpd.GeoDataFrame(
        {"summer_stage": [0.5], "winter_stage": [0.3]},
        geometry=[box(200, 200, 400, 400)],  # exactly the centre cell of the 3x3 grid
        index=["area_a"],
    )
    monkeypatch.setattr(nlmod.read.waterboard, "download_data", lambda **_kw: level_areas.copy())
    return polder.drn_from_waterboard_data(ds=ds, gwf=gwf, cbot=1.0, exclude=exclude)


def test_drn_fallback_excludes_lake_cells(disv_grid, monkeypatch):
    """Exclude drops the reach at a lake cell carrying real HHNK stage; baseline keeps it."""
    # Baseline (no exclude): the centre cell carries real HHNK peilgebied stage, so it gets a reach.
    ds, gwf, _geoms = disv_grid()
    drn_base = _run_drn(monkeypatch, ds, gwf, exclude=None)
    base_cells = _reach_cells(drn_base)
    assert CELL_REAL_STAGE in base_cells  # pin: without exclude the real-stage lake cell IS drained

    # With exclude: the reach at the centre cell is dropped even though its stage is real (not
    # fallback), while a non-excluded fallback cell still gets a reach.
    ds2, gwf2, _geoms2 = disv_grid()
    exclude = xr.zeros_like(ds2["top"]).astype(bool)
    exclude.loc[{"icell2d": CELL_REAL_STAGE}] = True
    drn_excl = _run_drn(monkeypatch, ds2, gwf2, exclude=exclude)
    excl_cells = _reach_cells(drn_excl)

    assert CELL_REAL_STAGE not in excl_cells
    assert CELL_FALLBACK in excl_cells  # a fallback (maaiveld) cell is unaffected
    assert excl_cells == base_cells - {CELL_REAL_STAGE}
