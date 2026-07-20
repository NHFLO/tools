"""Tests for :mod:`nhflotools.major_surface_waters` (RWS sea/lake CHD + GHB)."""

import nlmod
import numpy as np
import pytest
from shapely.geometry import box

from nhflotools.major_surface_waters import (
    chd_ghb_from_major_surface_waters,
    get_chd_ghb_data_from_major_surface_waters,
)
from tests.util import cell_polygon, make_gdf, make_gwf_disv, make_rect_vertex_ds

# One of the five OWMNAAM values nlmod.read.rws.discretize_northsea treats as sea
# (nlmod/read/rws.py:214-224). Any other name is a GHB water body, not sea.
SEA_NAME = "Hollandse kust (kustwater)"


def _patch_gdf(monkeypatch, gdf):
    """Make ``nlmod.read.rws.get_gdf_surface_water`` return ``gdf``.

    The single patch feeds both ``discretize_northsea`` and
    ``discretize_surface_water``, because the function under test threads one
    GeoDataFrame into both.
    """

    def _fake_get_gdf_surface_water(*_args, **_kwargs):
        return gdf

    monkeypatch.setattr(nlmod.read.rws, "get_gdf_surface_water", _fake_get_gdf_surface_water, raising=True)


def _inset_cell(ds, icell2d, inset=10.0):
    """Return a box strictly inside one cell, shrunk by ``inset`` on every side.

    A polygon snapped to the cell edges also *touches* its neighbours, and
    ``discretize_northsea`` rasterises with a plain ``intersects`` predicate, so an
    edge-snapped sea polygon would flag the whole grid as sea. Insetting keeps the
    sea confined to one cell while the intersected area stays exact.
    """
    xmin, ymin, xmax, ymax = cell_polygon(ds, icell2d).bounds
    return box(xmin + inset, ymin + inset, xmax - inset, ymax - inset)


def test_discretized_cond_is_area_over_bweerstand_and_dry_cells_are_zero(monkeypatch, vertex_ds):
    """Conductance equals intersected area / bweerstand; untouched cells are 0.0."""
    ds = vertex_ds
    # Sea over cell 0, inset by 10 m -> 80 x 80 m; IJsselmeer covering cell 3 exactly.
    gdf = make_gdf(
        [_inset_cell(ds, 0), cell_polygon(ds, 3)],
        OWMNAAM=[SEA_NAME, "IJsselmeer"],
        peil=[0.0, -0.25],
        bweerstand=[1.0, 2.0],
    )
    _patch_gdf(monkeypatch, gdf)

    rws_ds = get_chd_ghb_data_from_major_surface_waters(ds, cachedir=None)

    area = rws_ds["rws_oppwater_area"].values
    cond = rws_ds["rws_oppwater_cond"].values
    stage = rws_ds["rws_oppwater_stage"].values

    # 80 m x 80 m inset box in cell 0; the full 100 m x 100 m cell 3.
    np.testing.assert_array_equal(area, [6400.0, 0.0, 0.0, 10000.0])
    # cond = intersected area / bweerstand, per water body.
    np.testing.assert_array_equal(cond, [6400.0 / 1.0, 0.0, 0.0, 10000.0 / 2.0])
    np.testing.assert_array_equal(stage, [0.0, 0.0, 0.0, -0.25])

    # Cells without surface water must be exactly zero, never NaN: downstream
    # chd_ghb_from_major_surface_waters masks the GHB on ``cond > 0``.
    assert not np.isnan(cond).any()


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_ijsselmeer_stage_override_applies_only_where_it_intersects(monkeypatch, vertex_ds, reverse_rows):
    """The IJsselmeer peil overwrites the stage only in cells it intersects.

    Row order is permuted as well: the winner-take-all loop in
    ``discretize_surface_water`` compares intersected areas strictly, so the result
    must not depend on the order the water bodies appear in.
    """
    ds = vertex_ds
    # Noordzeekanaal covers the whole southern row (cells 2 and 3) -> 10000 m2 each.
    # IJsselmeer covers the western half of cell 2 only -> 5000 m2, so it loses the
    # area contest there and never reaches cell 3.
    rows = [
        (box(0.0, 0.0, 200.0, 100.0), "Noordzeekanaal", -0.5, 1.0),
        (box(0.0, 0.0, 50.0, 100.0), "IJsselmeer", -0.25, 4.0),
    ]
    if reverse_rows:
        rows.reverse()
    geoms, names, peilen, bweerstanden = zip(*rows, strict=True)
    _patch_gdf(monkeypatch, make_gdf(geoms, OWMNAAM=list(names), peil=list(peilen), bweerstand=list(bweerstanden)))

    rws_ds = get_chd_ghb_data_from_major_surface_waters(ds, cachedir=None)

    # Cell 2: overridden to the IJsselmeer peil. Cell 3: keeps the canal peil.
    np.testing.assert_array_equal(rws_ds["rws_oppwater_stage"].values, [0.0, 0.0, -0.25, -0.5])
    # The override touches the stage only; the canal keeps winning area and cond.
    np.testing.assert_array_equal(rws_ds["rws_oppwater_area"].values, [0.0, 0.0, 10000.0, 10000.0])
    np.testing.assert_array_equal(rws_ds["rws_oppwater_cond"].values, [0.0, 0.0, 10000.0, 10000.0])


def test_input_ds_is_mutated_with_northsea_and_extrapolated_under_sea(monkeypatch):
    """``ds`` gains ``northsea`` and its all-NaN sea column is filled in place.

    ``nlmod.dims.extrapolate_ds(ds)`` is called for its side effect only -- the
    return value is discarded -- so this pins the in-place mutation contract.
    """
    # 3x1 grid so the nearest valid neighbour of the sea cell is unique:
    # cell 0 (x=50) -> cell 1 (x=150, 100 m) beats cell 2 (x=250, 200 m).
    ds = make_rect_vertex_ds(nx=3, ny=1)
    ds["botm"].values[:, 0] = np.nan
    ds["kh"].values[:, 0] = np.nan
    ds["botm"].values[:, 1] = [-11.0, -21.0]
    ds["botm"].values[:, 2] = [-12.0, -22.0]
    ds["kh"].values[:, 1] = 8.0
    ds["kh"].values[:, 2] = 16.0

    _patch_gdf(
        monkeypatch,
        make_gdf([_inset_cell(ds, 0)], OWMNAAM=[SEA_NAME], peil=[0.0], bweerstand=[1.0]),
    )

    get_chd_ghb_data_from_major_surface_waters(ds, cachedir=None)

    assert "northsea" in ds
    np.testing.assert_array_equal(ds["northsea"].values, [True, False, False])
    # The NaN column is replaced by an exact copy of its nearest neighbour, cell 1.
    np.testing.assert_array_equal(ds["botm"].values[:, 0], [-11.0, -21.0])
    np.testing.assert_array_equal(ds["kh"].values[:, 0], [8.0, 8.0])
    # Neighbours are untouched.
    np.testing.assert_array_equal(ds["botm"].values[:, 2], [-12.0, -22.0])


def _sea_and_lake_ds(tmp_path, pinch_layer0=()):
    """Build a 3x2 vertex ds carrying ``northsea`` and the two ``rws_oppwater`` fields.

    Cells 0 and 1 are sea, cells 3 and 4 carry a GHB water body, cells 2 and 5 are
    dry. Cells in ``pinch_layer0`` get a zero-thickness top layer, which makes
    nlmod derive ``idomain[0] == 0`` there.
    """
    ds = make_rect_vertex_ds(nx=3, ny=2)
    for icell2d in pinch_layer0:
        ds["botm"].values[0, icell2d] = float(ds["top"].values[icell2d])
    ds["northsea"] = ("icell2d", np.array([1, 1, 0, 0, 0, 0]))
    ds["rws_oppwater_stage"] = ("icell2d", np.array([0.5, 0.5, 0.0, -1.0, -2.0, 0.0]))
    # Sea cells carry a non-zero cond so the sea-zeroing is observable.
    ds["rws_oppwater_cond"] = ("icell2d", np.array([100.0, 100.0, 0.0, 250.0, 500.0, 0.0]))
    return make_gwf_disv(ds, tmp_path)


def test_chd_ghb_float_branch_masks_layers_and_disjointness(tmp_path):
    """Float sea_stage: sfw masks, GHB/CHD disjointness, heads, aux and layers."""
    # Cell 1 (sea) and cell 4 (GHB) have a pinched-out layer 0.
    ds, gwf = _sea_and_lake_ds(tmp_path, pinch_layer0=(1, 4))
    sea_stage = 0.0625  # exactly representable

    ghb, chd, ts_sea = chd_ghb_from_major_surface_waters(ds, gwf, sea_stage=sea_stage)

    is_sea = ds["northsea"].values.astype(bool)
    sfw_stage = ds["sfw_stage"].values
    sfw_cond = ds["sfw_cond"].values
    # Stage is NaN exactly on the sea, and the raw RWS stage everywhere else.
    np.testing.assert_array_equal(np.isnan(sfw_stage), is_sea)
    np.testing.assert_array_equal(sfw_stage[~is_sea], ds["rws_oppwater_stage"].values[~is_sea])
    # Conductance is zeroed on the sea, untouched elsewhere.
    np.testing.assert_array_equal(sfw_cond, np.where(is_sea, 0.0, ds["rws_oppwater_cond"].values))

    ghb_rec = ghb.stress_period_data.get_data(0)
    chd_rec = chd.stress_period_data.get_data(0)
    ghb_cells = {cellid[1] for cellid in ghb_rec["cellid"]}
    chd_cells = {cellid[1] for cellid in chd_rec["cellid"]}

    assert ghb_cells == {3, 4}
    # Cell 1 is sea but has no active layer 0; CHD is a layer-0-only package, so it
    # drops out while cell 0 remains.
    assert chd_cells == {0}
    # A cell must never be both a general head boundary and a fixed head.
    assert ghb_cells.isdisjoint(chd_cells)

    # GHB goes into the first active layer: 0 for cell 3, 1 for the pinched cell 4.
    assert {(cellid[1], cellid[0]) for cellid in ghb_rec["cellid"]} == {(3, 0), (4, 1)}
    np.testing.assert_array_equal(ghb_rec["bhead"], [-1.0, -2.0])
    np.testing.assert_array_equal(ghb_rec["cond"], [250.0, 500.0])
    # The GHB water bodies are fresh.
    np.testing.assert_array_equal(ghb_rec["CONCENTRATION"], [0.0, 0.0])

    np.testing.assert_array_equal(chd_rec["head"], [sea_stage])
    # The sea is fixed at 18000 mg Cl-/l in the transport model (SEA_CHLORIDE_MG_L).
    np.testing.assert_array_equal(chd_rec["CONCENTRATION"], [18000.0])
    assert ts_sea is None


def test_chd_time_series_branch_wiring(tmp_path):
    """A list sea_stage wires a linear time series named ``sea_stage`` into the CHD."""
    ds, gwf = _sea_and_lake_ds(tmp_path)
    series = [(0.0, 0.25), (365.0, 0.5)]

    _ghb, chd, _ts_sea = chd_ghb_from_major_surface_waters(ds, gwf, sea_stage=series)

    chd_rec = chd.stress_period_data.get_data(0)
    # One record per active sea cell, each head the literal time-series name, not a number.
    n_sea = int(ds["northsea"].sum())
    assert list(chd_rec["head"]) == ["sea_stage"] * n_sea
    assert chd.ts.time_series_namerecord.get_data().tolist() == [("sea_stage",)]
    assert chd.ts.interpolation_methodrecord.get_data().tolist() == [("linear",)]
    np.testing.assert_array_equal(
        np.array(chd.ts.timeseries.get_data().tolist()),
        np.array(series),
    )


def test_chd_none_guard_returns_three_tuple(tmp_path):
    """With no active sea cell the CHD is absent and the call still returns a triple."""
    # Both sea cells lose their layer 0, so nlmod.gwf.chd returns None.
    ds, gwf = _sea_and_lake_ds(tmp_path, pinch_layer0=(0, 1))

    result = chd_ghb_from_major_surface_waters(ds, gwf, sea_stage=[(0.0, 0.25), (365.0, 0.5)])

    # Unpacking enforces the documented (ghb, chd, ts_sea) arity.
    ghb, chd, ts_sea = result
    assert chd is None
    assert ts_sea is None
    # The GHB is unaffected by the missing sea.
    assert {cellid[1] for cellid in ghb.stress_period_data.get_data(0)["cellid"]} == {3, 4}
