"""Tests for :mod:`nhflotools.panden` (the PWN infiltration-pond RIV package)."""

import logging

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from nhflotools.panden import get_oppervlakte_pwn_shapes, riv_from_oppervlakte_pwn
from tests.util import make_gwf_disv, make_rect_vertex_ds

# Stages hardcoded in panden.py; rbot is stage - 2.0 m.
ICAS_STAGE = 2.8
IKIEF_STAGE = 5.8
DEPTH = 2.0


def _write_panden_shp(directory, geometries, names):
    """Write the real ``Panden_ICAS_IKIEF.shp`` the reader expects.

    A genuine shapefile (rather than a stubbed GeoDataFrame) is written so the
    ``gpd.read_file`` path, the DBF round-trip of ``Naam`` and ``make_valid`` are all
    exercised.

    Parameters
    ----------
    directory : pathlib.Path
        Directory that becomes ``data_path_panden``.
    geometries : sequence of shapely.geometry.Polygon
        One polygon per pand.
    names : sequence of str
        Value of the ``Naam`` attribute of each pand.

    Returns
    -------
    str
        ``directory`` as a string, ready to pass as ``data_path_panden``.
    """
    gdf = gpd.GeoDataFrame({"Naam": list(names)}, geometry=list(geometries), crs="EPSG:28992")
    gdf.to_file(directory / "Panden_ICAS_IKIEF.shp")
    return str(directory)


@pytest.fixture
def panden_ds_gwf(tmp_path):
    """``(ds, gwf)`` on the default 2x2 100 m grid, transport enabled.

    Cell 0 spans x in [0, 100], y in [100, 200]; cell 1 spans x in [100, 200], y in
    [100, 200]. Transport is on so the SSM registration branch is reachable.
    """
    ds = make_rect_vertex_ds(transport=1)
    return make_gwf_disv(ds, tmp_path / "model")


def test_get_oppervlakte_pwn_shapes_assigns_stages_and_drops_other_names(tmp_path, caplog):
    """Only ICAS/IKIEF panden survive, with their stage, resistance and rbot set.

    A "VIJVER" pond carries neither name: it must be dropped with a warning rather than
    reaching the RIV package with a NaN stage (regression, see #41).
    """
    path = _write_panden_shp(
        tmp_path,
        [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
        ["ICAS-noord", "IKIEF-3", "VIJVER"],
    )

    with caplog.at_level(logging.WARNING, logger="nhflotools.panden"):
        shapes = get_oppervlakte_pwn_shapes(data_path_panden=path)

    assert list(shapes["Naam"]) == ["ICAS-noord", "IKIEF-3"]
    np.testing.assert_array_equal(shapes["stage"].to_numpy(), [ICAS_STAGE, IKIEF_STAGE])
    np.testing.assert_array_equal(shapes["c"].to_numpy(), [1.0, 1.0])
    # rbot is 2 m below the stage of that same pand.
    np.testing.assert_array_equal(shapes["rbot"].to_numpy(), [ICAS_STAGE - DEPTH, IKIEF_STAGE - DEPTH])
    assert any("VIJVER" in rec.getMessage() for rec in caplog.records)


def test_riv_single_cell_known_answer(tmp_path, panden_ds_gwf):
    """One 50x50 m pand inside one cell yields one fully derived RIV record."""
    ds, gwf = panden_ds_gwf
    # 50 m x 50 m, wholly inside cell 0 (x in [0, 100], y in [100, 200]).
    path = _write_panden_shp(tmp_path, [box(10, 110, 60, 160)], ["ICAS-noord"])

    riv = riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path)
    spd = riv.stress_period_data.get_data(0)

    # cond = intersected area / c = (50 * 50) / 1.0. rbot = 2.8 - 2.0 is above botm[0]
    # = -10, so lay_of_rbot puts the record in layer 0. The aux (CONCENTRATION) value
    # precedes the boundname, which is the order flopy reads them in.
    assert len(spd) == 1
    cellid, stage, cond, rbot, aux, boundname = tuple(spd[0])
    assert tuple(cellid) == (0, 0)
    assert stage == ICAS_STAGE
    assert cond == 50.0 * 50.0
    assert rbot == ICAS_STAGE - DEPTH
    assert aux == 0.0
    assert boundname.strip() == "ICAS-noord"


def test_riv_conductance_is_conserved_and_stage_area_weighted(tmp_path, panden_ds_gwf):
    """Splitting a pand over cells partitions its conductance; a shared cell blends stages.

    The pre-fix code aggregated the resistance ``c`` instead of ``area / c``, a silent
    ~1e4x conductance error that this partition identity pins down.
    """
    ds, gwf = panden_ds_gwf
    path = _write_panden_shp(
        tmp_path,
        # Straddles the cell 0 / cell 1 boundary at x = 100: 25 m x 50 m in cell 0 and
        # 75 m x 50 m in cell 1. The second pand sits wholly in cell 0.
        [box(75, 110, 175, 160), box(10, 110, 60, 160)],
        ["ICAS-noord", "IKIEF-3"],
    )

    riv = riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path)
    records = {tuple(rec[0]): rec for rec in riv.stress_period_data.get_data(0)}

    icas_cell0 = 25.0 * 50.0
    icas_cell1 = 75.0 * 50.0
    ikief_cell0 = 50.0 * 50.0
    assert set(records) == {(0, 0), (0, 1)}

    # c == 1.0 everywhere, so cond per cell is exactly the intersected area, and the
    # two pieces of the straddling pand must add back up to its full area.
    assert records[(0, 0)][2] == icas_cell0 + ikief_cell0
    assert records[(0, 1)][2] == icas_cell1
    total_cond = records[(0, 0)][2] + records[(0, 1)][2]
    assert total_cond == icas_cell0 + icas_cell1 + ikief_cell0

    # Cell 0 holds both panden: area-weighted stage, minimum rbot.
    expected_stage = (ICAS_STAGE * icas_cell0 + IKIEF_STAGE * ikief_cell0) / (icas_cell0 + ikief_cell0)
    # rel tolerance accommodates the pandas groupby accumulation order of the
    # area * stage products, which need not match the order used here.
    assert records[(0, 0)][1] == pytest.approx(expected_stage, rel=1e-12)
    assert records[(0, 0)][3] == ICAS_STAGE - DEPTH
    assert records[(0, 1)][1] == ICAS_STAGE
    assert records[(0, 1)][3] == ICAS_STAGE - DEPTH


def test_riv_rbot_selects_first_layer_it_reaches_into(tmp_path):
    """A pand bottom below the first layer bottom is placed in the layer that holds it."""
    # Layer 0 spans [5, 1], layer 1 spans [1, -10]; rbot = 0.8 lies inside layer 1.
    ds = make_rect_vertex_ds(top=5.0, botm=(1.0, -10.0))
    ds, gwf = make_gwf_disv(ds, tmp_path / "model")
    path = _write_panden_shp(tmp_path, [box(10, 110, 60, 160)], ["ICAS-noord"])

    riv = riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path)
    spd = riv.stress_period_data.get_data(0)

    assert len(spd) == 1
    assert tuple(spd[0][0]) == (1, 0)


def test_riv_outside_extent_returns_none_and_leaves_ds_untouched(tmp_path, panden_ds_gwf):
    """No intersection means no RIV package and no SSM source registered."""
    ds, gwf = panden_ds_gwf
    # Model extent is [0, 200] x [0, 200].
    path = _write_panden_shp(tmp_path, [box(1000, 1000, 1100, 1100)], ["ICAS-noord"])

    assert riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path) is None
    assert "ssm_sources" not in ds.attrs
    assert not [p for p in gwf.get_package_list() if p.lower().startswith("riv")]


@pytest.mark.parametrize("transport", [0, 1])
def test_riv_ssm_registration_is_transport_gated_and_idempotent(tmp_path, transport):
    """The RIV aux is registered as an SSM source once, and only under transport.

    Rebuilding the flow model on an existing dataset (the model script re-run against a
    cached ``ds``) must not append the same package a second time.
    """
    ds = make_rect_vertex_ds(transport=transport)
    path = _write_panden_shp(tmp_path, [box(10, 110, 60, 160)], ["ICAS-noord"])

    ds, gwf = make_gwf_disv(ds, tmp_path / "run1")
    riv = riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path)
    ds, gwf2 = make_gwf_disv(ds, tmp_path / "run2")
    riv_from_oppervlakte_pwn(ds, gwf2, data_path_panden=path)

    if transport:
        assert ds.attrs["ssm_sources"] == [riv.package_name]
    else:
        assert "ssm_sources" not in ds.attrs


def test_riv_stores_panden_coverage(tmp_path, panden_ds_gwf):
    """The panden-covered fraction of each cell is stored for the fractional recharge mask."""
    ds, gwf = panden_ds_gwf
    # 50 m x 50 m pand wholly inside cell 0 of the 2x2 grid of 100 m x 100 m cells.
    path = _write_panden_shp(tmp_path, [box(10, 110, 60, 160)], ["ICAS-noord"])

    riv_from_oppervlakte_pwn(ds, gwf, data_path_panden=path)

    np.testing.assert_allclose(ds["panden_coverage"].values, [2500.0 / 10000.0, 0.0, 0.0, 0.0])
