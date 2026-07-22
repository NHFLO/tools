"""Tests for nhflotools.lakes.

The Bergen pond/lake carve, the per-lake stage RIV and the recharge pond-mask are all
derived from the single aggregator ``_aggregate_lake_cells``, so the carved-cell set and the
stage-reach set are equal by construction. These tests pin that identity, the strict 50%
coverage threshold, the per-lake-per-cell aggregation (pieces merge within a lake, never
across lakes), and the boolean and fractional recharge masks.
"""

import logging
import types

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import pytest
import xarray as xr
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

# Shared by the RIV and LAK equivalence tests: one single-piece lake covering the centre
# cell (strt 2.5, botm 0.0) plus one two-piece lake in CELL_60. Both tests assert on the
# same aggregated values, so they must consume the identical input.
CLAKE = 10.0
AREA_A, AREA_B = 24000.0, 12000.0  # piece areas of the two-piece lake
STRT_A, STRT_B = 2.0, 3.0  # stages of the two pieces (area-weighted in aggregation)


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


def _single_and_two_piece_lakes(geoms):
    """Build the lake gdf shared by the RIV and LAK equivalence tests."""
    return _lake_gdf([
        _piece(CELL_FULL, geoms[CELL_FULL], 2.5, 0.0, clake=CLAKE, ident="lake_single"),
        _piece(CELL_60, box(200, 400, 320, 600), STRT_A, 0.5, clake=CLAKE, ident="lake_two_pieces"),
        _piece(CELL_60, box(320, 400, 400, 550), STRT_B, 0.0, clake=CLAKE, ident="lake_two_pieces"),
    ])


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
    gdf = _single_and_two_piece_lakes(geoms)
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
    assert single["cond"] == pytest.approx(40000.0 / CLAKE)  # piece_area / clake

    # two-piece cell: min rbot, summed cond, area-weighted stage
    two = recs[CELL_60]
    assert two["rbot"] == pytest.approx(0.0)  # min(0.5, 0.0)
    assert two["cond"] == pytest.approx(AREA_A / CLAKE + AREA_B / CLAKE)  # summed
    assert two["stage"] == pytest.approx((AREA_A * STRT_A + AREA_B * STRT_B) / (AREA_A + AREA_B))

    # no "stage below bottom elevation" / "records without a stage" warnings on valid data
    messages = " ".join(rec.getMessage().lower() for rec in caplog.records)
    assert "stage below bottom" not in messages
    assert "without a stage" not in messages

    # transport build registers the RIV as an SSM source exactly once
    assert riv.package_name in ds_carved.attrs["ssm_sources"]
    assert ds_carved.attrs["ssm_sources"].count(riv.package_name) == 1


def test_two_lakes_sharing_a_cell_get_separate_reaches(disv_grid):
    """Two different lakes overlapping one cell each keep their own reach, stage and boundname.

    Regression for the per-cell collapse: merging lakes within a cell blends their stages
    (each lake holds its own prescribed stage) and destroys the per-lake boundname needed to
    configure MVR movers and weir outlets between individual lakes.
    """
    ds, gwf, _geoms = disv_grid(transport=0)
    clake = 10.0
    strt_a, strt_b = 2.0, 3.0
    # Lake A covers the west half of the centre cell, lake B the east half; both also cover a
    # cell of their own so each lake exists independently of the shared cell.
    rows = [
        _piece(CELL_FULL, box(200, 200, 300, 400), strt_a, 1.0, clake=clake, ident="lake_a"),
        _piece(CELL_FULL, box(300, 200, 400, 400), strt_b, 0.5, clake=clake, ident="lake_b"),
        _piece(CELL_60, box(200, 400, 400, 600), strt_a, 1.0, clake=clake, ident="lake_a"),
        _piece(PANDEN_B, box(400, 200, 600, 400), strt_b, 0.5, clake=clake, ident="lake_b"),
    ]
    gdf = _lake_gdf(rows)
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)

    # the shared cell is carved once (combined coverage 100%), to the deepest lake bed
    assert CELL_FULL in lake_cellids
    assert ds_carved["top"].sel(icell2d=CELL_FULL).item() == pytest.approx(0.5)

    riv = lakes.riv_from_lakes_pwn(ds_carved, gwf, gdf, min_area_fraction=0.5)
    recs = riv.stress_period_data.data[0]
    shared = [rec for rec in recs if rec["cellid"][-1] == CELL_FULL]

    # one reach per lake in the shared cell, not one blended reach per cell
    assert sorted(rec["boundname"] for rec in shared) == ["lake_a", "lake_b"]
    by_name = {rec["boundname"]: rec for rec in shared}
    # each reach keeps its own lake's stage and conductance (half a 40000 m2 cell each)
    assert by_name["lake_a"]["stage"] == pytest.approx(strt_a)
    assert by_name["lake_b"]["stage"] == pytest.approx(strt_b)
    assert by_name["lake_a"]["cond"] == pytest.approx(20000.0 / clake)
    assert by_name["lake_b"]["cond"] == pytest.approx(20000.0 / clake)
    # each reach also keeps its own lake's bed: the cell is carved to the deepest bed (0.5),
    # but lake_a's perched-infiltration cap stays at its own higher bed (rbot 1.0)
    assert by_name["lake_a"]["rbot"] == pytest.approx(1.0)
    assert by_name["lake_b"]["rbot"] == pytest.approx(0.5)
    # the stage-cell set still equals the carved-cell set (identity by construction)
    assert {rec["cellid"][-1] for rec in recs} == set(lake_cellids.tolist())


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


def test_pond_mask_fractional(disv_grid):
    """``fractional=True`` returns 1.0 at carved/panden cells and the coverage fraction below threshold."""
    ds, _gwf, geoms = disv_grid()
    rows = [
        _piece(CELL_40, box(0, 400, 80, 600), 2.0, 1.0),  # 40% -> not carved, fraction 0.4
        _piece(CELL_60, box(200, 400, 320, 600), 2.0, 1.0),  # 60% -> carved, fraction 1.0
        _piece(CELL_FULL, geoms[CELL_FULL], 2.0, 1.0),  # 100% -> carved, fraction 1.0
    ]
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, _lake_gdf(rows), min_area_fraction=0.5)
    assert set(lake_cellids.tolist()) == {CELL_60, CELL_FULL}

    frac = lakes.recharge_pond_mask(ds_carved, None, fractional=True)
    # carved cells are fully stage-carried (whole cell modeled as lake bed), even at 60% coverage
    assert frac.sel(icell2d=CELL_60).item() == pytest.approx(1.0)
    assert frac.sel(icell2d=CELL_FULL).item() == pytest.approx(1.0)
    # below-threshold cell keeps its open-water fraction; untouched cells keep 0.0
    assert frac.sel(icell2d=CELL_40).item() == pytest.approx(0.4)
    assert frac.sel(icell2d=PANDEN_A).item() == pytest.approx(0.0)

    # the boolean default is unchanged: the 40% cell is not excluded
    mask = lakes.recharge_pond_mask(ds_carved, None)
    assert not bool(mask.sel(icell2d=CELL_40).item())

    # a panden RIV cell is fully stage-carried in both modes
    stub = types.SimpleNamespace(stress_period_data=types.SimpleNamespace(data={0: {"cellid": [(0, PANDEN_A)]}}))
    frac_panden = lakes.recharge_pond_mask(ds_carved, stub, fractional=True)
    assert frac_panden.sel(icell2d=PANDEN_A).item() == pytest.approx(1.0)


def test_carve_lake_cells_empty_when_no_cell_clears_threshold(disv_grid):
    """An empty / all-below-threshold lake gdf carves nothing instead of crashing.

    Regression for the empty-geometry ``.area`` crash: ``carve_lake_cells`` previously
    raised ``TypeError: can only use area methods with polygon geometries`` whenever the
    coverage filter (or an empty input) left no surviving lake piece.
    """
    ds, _gwf, _geoms = disv_grid()
    sliver = _lake_gdf([_piece(CELL_40, box(0, 400, 20, 420), 2.0, 1.0)])  # 400/40000 = 1% < 50%
    empty = _lake_gdf([_piece(CELL_40, box(0, 400, 80, 600), 2.0, 1.0)]).iloc[0:0]
    for gdf in (sliver, empty):
        ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)
        assert lake_cellids.tolist() == []
        assert not bool(ds_carved["lake_cell"].any())
        assert bool((ds_carved["top"] == ds["top"]).all())  # nothing carved


def test_riv_from_lakes_pwn_returns_none_when_no_lake_cells(disv_grid):
    """The per-lake RIV is ``None`` (no package added) when no cell clears the threshold."""
    ds, gwf, _geoms = disv_grid()
    sliver = _lake_gdf([_piece(CELL_40, box(0, 400, 20, 420), 2.0, 1.0)])
    ds_carved, _ = lakes.carve_lake_cells(ds, sliver, min_area_fraction=0.5)
    assert lakes.riv_from_lakes_pwn(ds_carved, gwf, sliver, min_area_fraction=0.5) is None


def _build_lak(ds, gwf, gdf_lake_grid):
    """Mirror the model script's LAK branch: prepare the frame, then build via nlmod."""
    gdf = lakes.lak_gdf_from_lakes_pwn(ds, gdf_lake_grid, min_area_fraction=0.5)
    rainfall, evaporation = nlmod.gwf.copy_meteorological_data_from_ds(gdf, ds, boundname_column="identificatie")
    return nlmod.gwf.lake_from_gdf(
        gwf, gdf, ds, rainfall=rainfall, evaporation=evaporation, boundname_column="identificatie"
    )


def _add_time_and_recharge(ds, recharge=0.0007):
    """Extend the synthetic ds with the time axis and recharge that LAK needs."""
    ds = ds.assign_coords(time=pd.to_datetime(["2023-01-01"]))
    ds["time"].attrs["time_units"] = "days"
    ds["recharge"] = xr.DataArray(
        np.full((1, ds.sizes["icell2d"]), recharge), dims=("time", "icell2d"), attrs={"units": "m/d"}
    )
    return ds


def test_lak_connections_match_carved_cells_with_equivalent_conductance(disv_grid):
    """LAK connects exactly the carved cells, with the same lakebed conductance as the RIV.

    The effective bed resistance is chosen such that ``bedleak * cell_area`` equals the
    summed piece conductance (piece_area / clake) of the corresponding RIV reach. The total
    MF6 exchange is still somewhat weaker than the RIV's, because VERTICAL connections add
    the half-cell vertical resistance in series with the lakebed.
    """
    ds, gwf, geoms = disv_grid()
    ds = _add_time_and_recharge(ds)
    gdf = _single_and_two_piece_lakes(geoms)
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)

    lak = _build_lak(ds_carved, gwf, gdf)

    conns = lak.connectiondata.array
    assert {int(cid[-1]) for cid in conns["cellid"]} == set(lake_cellids.tolist())
    assert all(cid[0] == 0 for cid in conns["cellid"])  # first active layer

    cell_area = 40000.0
    bedleak = {int(cid[-1]): bl for cid, bl in zip(conns["cellid"], conns["bedleak"], strict=True)}
    assert bedleak[CELL_FULL] * cell_area == pytest.approx(cell_area / CLAKE)
    assert bedleak[CELL_60] * cell_area == pytest.approx((AREA_A + AREA_B) / CLAKE)

    strt_by_name = {rec[-1]: rec[1] for rec in lak.packagedata.array}
    assert strt_by_name["lake_single"] == pytest.approx(2.5)
    assert strt_by_name["lake_two_pieces"] == pytest.approx((AREA_A * STRT_A + AREA_B * STRT_B) / (AREA_A + AREA_B))

    # copy_meteorological_data_from_ds feeds ds['recharge'] to the lakes as RAINFALL
    settings = {(int(rec[0]), rec[1]): rec[2] for rec in lak.perioddata.data[0]}
    lakeno_by_name = {rec[-1]: int(rec[0]) for rec in lak.packagedata.array}
    assert settings[lakeno_by_name["lake_single"], "RAINFALL"] == pytest.approx(0.0007)


def test_lak_two_lakes_sharing_a_cell_stay_separate(disv_grid):
    """Two lakes overlapping one cell each keep their own lake, connection and strt.

    Pins the strt collapse's grouping key: grouping by cell instead of by lake would blend
    the two lakes' stages in the shared cell and crash nlmod's single-strt-per-lake check.
    """
    ds, gwf, _geoms = disv_grid()
    ds = _add_time_and_recharge(ds)
    clake = 10.0
    strt_a, strt_b = 2.0, 3.0
    rows = [
        _piece(CELL_FULL, box(200, 200, 300, 400), strt_a, 1.0, clake=clake, ident="lake_a"),
        _piece(CELL_FULL, box(300, 200, 400, 400), strt_b, 0.5, clake=clake, ident="lake_b"),
        _piece(CELL_60, box(200, 400, 400, 600), strt_a, 1.0, clake=clake, ident="lake_a"),
        _piece(PANDEN_B, box(400, 200, 600, 400), strt_b, 0.5, clake=clake, ident="lake_b"),
    ]
    gdf = _lake_gdf(rows)
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)

    lak = _build_lak(ds_carved, gwf, gdf)

    strt_by_name = {rec[-1]: rec[1] for rec in lak.packagedata.array}
    assert strt_by_name["lake_a"] == pytest.approx(strt_a)
    assert strt_by_name["lake_b"] == pytest.approx(strt_b)
    assert {int(rec[0]) for rec in lak.packagedata.array} == {0, 1}  # two distinct lakes

    conns = lak.connectiondata.array
    shared = [conn for conn in conns if int(conn["cellid"][-1]) == CELL_FULL]
    assert len(shared) == len(strt_by_name)  # one connection per lake in the shared cell
    # each lake's lakebed conductance in the shared cell equals its RIV cond (half cell each)
    for conn in shared:
        assert conn["bedleak"] * 40000.0 == pytest.approx(20000.0 / clake)
    assert {int(conn["cellid"][-1]) for conn in conns} == set(lake_cellids.tolist())


def test_lak_single_strt_for_lake_spanning_multiple_cells(disv_grid):
    """A lake over several cells with differing per-cell stages gets one exact strt.

    Without the per-lake strt collapse, nlmod's single-strt-per-lake check raises on any
    per-cell difference (deterministic proxy for float-level aggregation noise).
    """
    ds, gwf, geoms = disv_grid()
    ds = _add_time_and_recharge(ds)
    rows = [
        _piece(CELL_FULL, geoms[CELL_FULL], 2.0, 0.0, ident="one_lake"),
        _piece(CELL_60, geoms[CELL_60], 3.0, 0.5, ident="one_lake"),
    ]
    gdf = _lake_gdf(rows)
    ds_carved, lake_cellids = lakes.carve_lake_cells(ds, gdf, min_area_fraction=0.5)

    lak = _build_lak(ds_carved, gwf, gdf)

    assert len(lak.packagedata.array) == 1
    assert lak.packagedata.array[0][1] == pytest.approx(2.5)  # mean over equal-area cells
    assert {int(conn["cellid"][-1]) for conn in lak.connectiondata.array} == set(lake_cellids.tolist())


def test_lak_gdf_from_lakes_pwn_returns_none_when_no_lake_cells(disv_grid):
    """The LAK input frame is ``None`` when no cell clears the threshold (no package built)."""
    ds, _gwf, _geoms = disv_grid()
    ds = _add_time_and_recharge(ds)
    sliver = _lake_gdf([_piece(CELL_40, box(0, 400, 20, 420), 2.0, 1.0)])
    ds_carved, _ = lakes.carve_lake_cells(ds, sliver, min_area_fraction=0.5)
    assert lakes.lak_gdf_from_lakes_pwn(ds_carved, sliver, min_area_fraction=0.5) is None
