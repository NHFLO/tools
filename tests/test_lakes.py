"""Tests for nhflotools.lakes.

The Bergen pond/lake carve, the per-lake stage RIV and the recharge pond-mask are all
derived from the single aggregator ``_aggregate_lake_cells``, so the carved-cell set and the
stage-reach set are equal by construction. These tests pin that identity, the strict 50%
coverage threshold, the per-cell collapse of overlapping lake pieces, and the recharge mask.
"""

import logging
import types

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from nhflotools import lakes

# Cell ids of the shared 3x3 disv grid used throughout (icell2d, row-major from the top-left).
CELL_40 = 0  # 40% lake coverage -> not carved
CELL_60 = 1  # 60% lake coverage -> carved
CELL_50 = 2  # exactly 50% coverage -> not carved (strict '>')
CELL_FULL = 4  # centre cell, fully covered -> carved
PANDEN_A = 3  # stub panden RIV cell
PANDEN_B = 5  # stub panden RIV cell

DEEPEST_BOTM = 0.2  # minimum botm across the overlapping pieces of CELL_FULL


def _piece(cellid, geom, strt, botm, clake=10.0, ident="lake"):
    """Build one lake-piece record already assigned to a grid cell."""
    return {
        "cellid": cellid,
        "geometry": geom,
        "strt": strt,
        "botm": botm,
        "clake": clake,
        "identificatie": ident,
    }


def _lake_gdf(rows):
    """Build a lake gdf (as gdf_to_grid would return) from piece dicts."""
    return gpd.GeoDataFrame(rows, geometry="geometry")


def test_carve_selection_min_area_fraction(disv_grid):
    """Coverage uses a strict '>' threshold and the carved bottom is the per-cell minimum."""
    ds, _gwf, _geoms = disv_grid()
    # Cell 0 at 40%, cell 1 at 60%, cell 2 at exactly 50%, cell 4 fully covered by two pieces.
    rows = [
        _piece(CELL_40, box(0, 400, 80, 600), 2.0, 1.0),  # 16000 / 40000 = 0.40 -> excluded
        _piece(CELL_60, box(200, 400, 320, 600), 2.0, 1.0),  # 24000 / 40000 = 0.60 -> carved
        _piece(CELL_50, box(400, 400, 500, 600), 2.0, 1.0),  # 20000 / 40000 = 0.50 -> excluded
        _piece(CELL_FULL, box(200, 200, 400, 300), 2.0, 1.0, ident="deep"),  # 20000, botm 1.0
        _piece(CELL_FULL, box(200, 300, 400, 400), 2.0, DEEPEST_BOTM, ident="deeper"),  # 20000, botm 0.2
    ]
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, _lake_gdf(rows), min_area_fraction=0.5)

    assert set(lake_cellids.tolist()) == {CELL_60, CELL_FULL}
    assert CELL_40 not in lake_cellids  # 40% not carved
    assert CELL_50 not in lake_cellids  # exactly 50% not carved -> pins strict '>'
    # the carved top is the minimum botm across the overlapping pieces of the cell
    assert ds_carved["top"].sel(icell2d=CELL_FULL).item() == pytest.approx(DEEPEST_BOTM)


def test_carve_and_stage_sets_identical(disv_grid):
    """A >50%-covered cell whose only piece lacks a stage is dropped from BOTH sets."""
    ds, gwf, geoms = disv_grid(transport=0)
    rows = [
        _piece(CELL_FULL, geoms[CELL_FULL], 2.5, 0.0, ident="lake_full"),  # has stage -> both sets
        _piece(CELL_40, geoms[CELL_40], np.nan, 1.0, ident="botm_only"),  # NO stage -> neither set
    ]
    gdf = _lake_gdf(rows)

    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)
    riv = lakes.riv_from_lakes_pwn(ds_carved, gwf, gdf, min_area_fraction=0.5)
    stage_cells = {rec["cellid"][-1] for rec in riv.stress_period_data.data[0]}

    # The unified dropna(['strt', 'botm']) filter drops the botm-only cell from both sets, so the
    # script's `stage_cells == lake_cellids` consistency assertion cannot fire on a valid build.
    assert stage_cells == set(lake_cellids.tolist())
    assert CELL_40 not in stage_cells
    assert CELL_40 not in set(lake_cellids.tolist())
    assert set(lake_cellids.tolist()) == {CELL_FULL}


def test_lake_riv_stage_boundary_on_synthetic_grid(disv_grid, caplog):
    """Single- and multi-piece cells yield exactly one reach each with the expected values."""
    ds, gwf, geoms = disv_grid(transport=1)
    # cell 1: two overlapping pieces, areas 24000 (strt 2.0, botm 0.5) and 12000 (strt 3.0, botm 0.0)
    area_a, area_b = 24000.0, 12000.0
    strt_a, strt_b = 2.0, 3.0
    clake = 10.0
    rows = [
        _piece(CELL_FULL, geoms[CELL_FULL], 2.5, 0.0, clake=clake, ident="lake_single"),
        _piece(CELL_60, box(200, 400, 320, 600), strt_a, 0.5, clake=clake, ident="piece_a"),
        _piece(CELL_60, box(320, 400, 400, 550), strt_b, 0.0, clake=clake, ident="piece_b"),
    ]
    gdf = _lake_gdf(rows)
    ds_carved, _lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)

    with caplog.at_level(logging.WARNING):
        riv = lakes.riv_from_lakes_pwn(ds_carved, gwf, gdf, min_area_fraction=0.5)

    recs = {rec["cellid"][-1]: rec for rec in riv.stress_period_data.data[0]}
    assert set(recs) == {CELL_60, CELL_FULL}  # one reach per cell, no duplicate cellids

    # single-piece cell: stage/rbot/cond and top-active-layer placement
    single = recs[CELL_FULL]
    assert single["cellid"][0] == 0  # lands in the top active layer -> pins lay_of_rbot
    assert single["stage"] == pytest.approx(2.5)
    assert single["rbot"] == pytest.approx(0.0)
    assert single["cond"] == pytest.approx(40000.0 / clake)  # piece_area / clake

    # two-piece cell: min rbot, summed cond, area-weighted stage
    two = recs[CELL_60]
    assert two["rbot"] == pytest.approx(0.0)  # min(0.5, 0.0)
    assert two["cond"] == pytest.approx(area_a / clake + area_b / clake)  # summed
    assert two["stage"] == pytest.approx((area_a * strt_a + area_b * strt_b) / (area_a + area_b))

    # no "stage below bottom elevation" / "records without a stage" warnings on valid data
    messages = " ".join(rec.getMessage().lower() for rec in caplog.records)
    assert "stage below bottom" not in messages
    assert "without a stage" not in messages

    # transport build registers the RIV as an SSM source exactly once
    assert riv.package_name in ds_carved.attrs["ssm_sources"]
    assert ds_carved.attrs["ssm_sources"].count(riv.package_name) == 1


def test_pond_mask_excludes_recharge_cells(disv_grid):
    """The mask equals lake_cell in the None branch and adds panden cells via cid[-1]."""
    ds, _gwf, geoms = disv_grid()
    gdf = _lake_gdf([_piece(CELL_60, geoms[CELL_60], 2.0, 1.0), _piece(CELL_FULL, geoms[CELL_FULL], 2.0, 1.0)])
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)
    assert set(lake_cellids.tolist()) == {CELL_60, CELL_FULL}

    # None branch (the default Bergen build): mask is exactly lake_cell
    mask_none = lakes.recharge_pond_mask(ds_carved, None)
    assert bool((mask_none == ds_carved["lake_cell"]).all())

    # stub panden RIV whose cellid entries are (layer, icell2d) tuples (with a duplicate cell)
    stub = types.SimpleNamespace(
        stress_period_data=types.SimpleNamespace(data={0: {"cellid": [(0, PANDEN_A), (0, PANDEN_B), (0, PANDEN_A)]}})
    )
    mask = lakes.recharge_pond_mask(ds_carved, stub)
    pond_cells = set(ds_carved["icell2d"].values[mask.values].tolist())
    expected = {CELL_60, CELL_FULL, PANDEN_A, PANDEN_B}
    assert pond_cells == expected  # lake cells + panden cells via cid[-1]

    northsea = ds_carved["top"].astype(int) * 0  # northsea == 0 everywhere -> all land
    masked_in = (northsea == 0) & ~mask
    # no lake or panden cell survives the recharge mask
    assert not bool(masked_in.sel(icell2d=sorted(expected)).any())
    # masked-in count drops by exactly |lake union panden| relative to the northsea-only mask
    assert int((northsea == 0).sum()) - int(masked_in.sum()) == len(expected)
