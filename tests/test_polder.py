"""Tests for nhflotools.polder.drn_from_waterboard_data.

Only the download is faked (``nlmod.read.waterboard.download_data`` hits a live, uncached
and flaky HHNK ArcGIS REST service at polder.py:40); the grid intersection, the
area-weighted aggregation, the nearest-neighbour stage fill and the flopy DRN package are
all built for real.

Note on conductance: polder.py:59 deliberately reads the FULL cell area, not the
intersected area (issue #51), so nothing here asserts an intersected-area conductance.
"""

import numpy as np
import xarray as xr
from shapely.geometry import box

from nhflotools.polder import drn_from_waterboard_data
from tests.util import cell_polygon, make_gdf, make_gwf_disv, make_rect_vertex_ds

# make_rect_vertex_ds() default: 2x2 cells of 100 m, row-major from the north-west corner,
# so cell 0=(50,150), 1=(150,150), 2=(50,50), 3=(150,50) and every cell area is 100*100.
CELL_AREA = 100.0 * 100.0


def _patch_download(monkeypatch, gdf):
    """Replace the live waterboard download with ``gdf`` and record the call kwargs."""
    calls = {}

    def _fake(**kwargs):
        calls.update(kwargs)
        return gdf

    monkeypatch.setattr("nlmod.read.waterboard.download_data", _fake)
    return calls


def _reclist(drn):
    """Return the DRN stress period data as sorted ``(layer, icell2d, elev, cond)`` tuples."""
    return sorted((int(r[0][0]), int(r[0][1]), float(r[1]), float(r[2])) for r in drn.stress_period_data.get_data(0))


def test_drn_partition_over_polder_land_and_sea(tmp_path, monkeypatch):
    """Polder cells, uncovered land, sea cells and blocked layers each land in the right bin.

    Regression for the zero-initialised conductance array, which gave uncovered land
    zero-conductance (i.e. inert) drains and simultaneously drained the North Sea.

    ``cbot`` is 2.0, not 1.0: dividing the cell area by 1.0 is an identity, so a
    conductance that ignored ``cbot`` altogether would pass unnoticed.
    """
    cbot = 2.0
    ds = make_rect_vertex_ds()
    # Zero-thickness layer 0 in cell 3 -> nlmod derives idomain 0 there, first active layer 1.
    ds["botm"][0, 3] = float(ds["top"][3])
    ds, gwf = make_gwf_disv(ds, tmp_path)
    ds["ahn"] = xr.DataArray(np.array([0.0, 4.25, 1.0, 2.5]), dims="icell2d")
    ds["northsea"] = xr.DataArray(np.array([0, 0, 1, 0]), dims="icell2d")

    gdf = make_gdf([cell_polygon(ds, 0)], summer_stage=1.0, winter_stage=3.0)
    _patch_download(monkeypatch, gdf)

    drn = drn_from_waterboard_data(ds, gwf, cbot=cbot)

    # cell 0 covered -> mean(1.0, 3.0); cells 1-3 uncovered -> ahn.
    np.testing.assert_array_equal(ds["drn_elev"].values, [2.0, 4.25, 1.0, 2.5])
    # Every non-sea cell gets area/cbot, whether or not a polder polygon covers it.
    np.testing.assert_array_equal(ds["drn_cond"].values, [CELL_AREA / cbot, CELL_AREA / cbot, np.nan, CELL_AREA / cbot])
    # The historical failure mode was a *zero* fallback conductance, not a NaN one.
    assert ds["drn_cond"].values[1] > 0.0

    # The sea cell is dropped; the blocked cell drains from its first active layer.
    assert _reclist(drn) == [
        (0, 0, 2.0, CELL_AREA / cbot),
        (0, 1, 4.25, CELL_AREA / cbot),
        (1, 3, 2.5, CELL_AREA / cbot),
    ]


def test_drn_elev_skips_nan_stages_and_fills_all_nan_cells_from_nearest(gwf_disv, monkeypatch):
    """A missing stage is skipped in the mean; a cell missing both is filled by its nearest peer."""
    ds, gwf = gwf_disv
    # ahn differs from every stage, so an elevation equal to a stage proves the stage path won.
    ds["ahn"] = xr.DataArray(np.array([9.0, 7.0, 8.0, 5.0]), dims="icell2d")

    gdf = make_gdf(
        [cell_polygon(ds, 0), cell_polygon(ds, 1), cell_polygon(ds, 2)],
        summer_stage=[1.0, np.nan, 2.0],
        winter_stage=[np.nan, np.nan, 6.0],
    )
    _patch_download(monkeypatch, gdf)

    drn_from_waterboard_data(ds, gwf)

    # cell 0: one-sided mean of (1.0, NaN) is 1.0, not 0.5.
    # cell 1 (150,150): both stages NaN -> nearest valid donor is cell 0 at (50,150), 100 m
    #   away, versus cell 2 at (50,50), 100*sqrt(2) m away; so 1.0, not 4.0 and not ahn.
    # cell 2: mean(2.0, 6.0) = 4.0. cell 3: no polygon -> ahn.
    np.testing.assert_array_equal(ds["drn_elev"].values, [1.0, 1.0, 4.0, 5.0])


def test_drn_empty_download_returns_none_and_leaves_ds_untouched(gwf_disv, monkeypatch):
    """No level areas in the extent short-circuits before any variable is written to ds."""
    ds, gwf = gwf_disv
    ds["ahn"] = xr.DataArray(np.zeros(4), dims="icell2d")
    calls = _patch_download(monkeypatch, make_gdf([], summer_stage=[], winter_stage=[]))

    assert drn_from_waterboard_data(ds, gwf) is None
    assert "drn_elev" not in ds
    assert "drn_cond" not in ds
    # The download must be scoped to the model, not to the whole waterboard.
    assert calls["extent"] == ds.extent


def test_drn_duplicate_download_index_is_deduplicated(gwf_disv, monkeypatch):
    """Repeated level-area identifiers are made unique so gdf_to_grid accepts the frame.

    ``nlmod.grid.gdf_to_grid`` raises ``ValueError: gdf should not have duplicate columns
    or index``, so reaching an area-weighted elevation at all proves the renaming ran.
    """
    ds, gwf = gwf_disv
    ds["ahn"] = xr.DataArray(np.zeros(4), dims="icell2d")

    # Two features share the id "A": one fills cell 0, one fills the west half of cell 1.
    # "B" fills the east half of cell 1.
    gdf = make_gdf(
        [cell_polygon(ds, 0), box(100.0, 100.0, 150.0, 200.0), box(150.0, 100.0, 200.0, 200.0)],
        summer_stage=[2.0, 6.0, 10.0],
        winter_stage=[2.0, 6.0, 10.0],
    )
    gdf.index = ["A", "A", "B"]
    _patch_download(monkeypatch, gdf)

    drn_from_waterboard_data(ds, gwf)

    # cell 1 keeps both halves: (5000*6.0 + 5000*10.0) / 10000 = 8.0. Losing either
    # duplicate would give 6.0 or 10.0 instead.
    np.testing.assert_array_equal(ds["drn_elev"].values, [2.0, 8.0, 0.0, 0.0])
