"""Tests for nhflotools.postprocessing.

The interface-elevation helper supersedes the old layer-discretised ``grensvlak`` with
``nlmod.dims.get_isosurface``. These tests pin the intended behaviour: it agrees with the old
logic at the fresh/salt limits (per column) and interpolates the crossing in between. The
remaining tests cover the budget guard against a real MODFLOW listing, the derived fields
``add_output_to_ds`` builds, and the exact set of figures ``plot_result_maps`` writes.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import nhflotools.postprocessing as pp
from nhflotools.postprocessing import interface_elevation
from tests.util import make_rect_vertex_ds, write_mf6_listing

# col0 crosses the fresh threshold, col1 stays fresh throughout, col2 stays salt throughout.
_CONC = [[100, 500, 2000, 9000], [50, 80, 90, 120], [3000, 4000, 5000, 6000]]
_THRESHOLD = 1000.0
_FRESH_COL, _SALT_COL = 1, 2


def _old_grensvlak(concentration, threshold, ds):
    """Reimplement the previous layer-discretised grensvlak as a regression baseline."""
    exceed = concentration > threshold
    any_exceed = exceed.any(dim="layer")
    ilay = exceed.argmax(dim="layer") - 1
    gv = ds["botm"].isel(layer=ilay)
    gv = xr.where(ilay == -1, ds["top"], gv)
    return xr.where(~any_exceed, ds["botm"].isel(layer=-1), gv)


def _synthetic_ds():
    """Build three columns x four layers with distinct tops so per-column fills are exercised."""
    layer = [0, 1, 2, 3]
    top = xr.DataArray([0.0, 5.0, -2.0], dims=["icell2d"])
    botm = xr.DataArray(
        np.array([
            [-10.0, -5.0, -12.0],
            [-20.0, -15.0, -22.0],
            [-30.0, -25.0, -32.0],
            [-40.0, -35.0, -42.0],
        ]),
        dims=["layer", "icell2d"],
        coords={"layer": layer},
    )
    return xr.Dataset({"top": top, "botm": botm})


def _conc(values):
    """Build a concentration DataArray from rows-per-column values."""
    return xr.DataArray(np.array(values, float).T, dims=["layer", "icell2d"], coords={"layer": [0, 1, 2, 3]})


def test_interface_matches_old_at_limits():
    """Fresh-throughout and salt-throughout columns match the old grensvlak (model bottom/top)."""
    ds = _synthetic_ds()
    conc = _conc(_CONC)
    gv = interface_elevation(ds, conc, _THRESHOLD)
    old = _old_grensvlak(conc, _THRESHOLD, ds)

    # fresh column -> model bottom, salt column -> model top, both matching the old logic
    assert gv.isel(icell2d=_FRESH_COL).item() == old.isel(icell2d=_FRESH_COL).item()
    assert gv.isel(icell2d=_FRESH_COL).item() == ds["botm"].isel(layer=-1, icell2d=_FRESH_COL).item()
    assert gv.isel(icell2d=_SALT_COL).item() == old.isel(icell2d=_SALT_COL).item()
    assert gv.isel(icell2d=_SALT_COL).item() == ds["top"].isel(icell2d=_SALT_COL).item()
    assert gv.attrs["threshold"] == _THRESHOLD


def test_interface_interpolates_crossing():
    """Between samples the interface is interpolated, not snapped to a layer boundary."""
    ds = _synthetic_ds()
    conc = _conc(_CONC)
    gv = interface_elevation(ds, conc, _THRESHOLD).isel(icell2d=0).item()

    # linear interpolation between cell centre -15 (conc 500) and -25 (conc 2000)
    z_upper, z_lower, c_upper, c_lower = -15.0, -25.0, 500.0, 2000.0
    expected = z_upper + (_THRESHOLD - c_upper) / (c_lower - c_upper) * (z_lower - z_upper)
    np.testing.assert_allclose(gv, expected)

    # strictly shallower than the old stepped value (the layer-1 botm) and below the layer-1 centre
    old = _old_grensvlak(conc, _THRESHOLD, ds).isel(icell2d=0).item()
    assert old == ds["botm"].isel(layer=1, icell2d=0).item()
    assert old < gv < z_upper


def test_interface_with_time_dim():
    """The helper broadcasts over a leading time dimension."""
    ds = _synthetic_ds()
    base = _conc(_CONC)
    conc = xr.concat([base, base], dim="time").assign_coords(time=[0, 1])
    gv = interface_elevation(ds, conc, _THRESHOLD)

    assert gv.sizes["time"] == conc.sizes["time"]
    # the interpolated column is broadcast identically across the (identical) time steps
    assert gv.isel(time=1, icell2d=0).item() == gv.isel(time=0, icell2d=0).item()
    assert gv.isel(time=0, icell2d=_FRESH_COL).item() == ds["botm"].isel(layer=-1, icell2d=_FRESH_COL).item()
    assert gv.isel(time=0, icell2d=_SALT_COL).item() == ds["top"].isel(icell2d=_SALT_COL).item()


def test_interface_salt_topped_returns_top():
    """A column salt at the surface but fresher below returns the model top (first up-crossing)."""
    ds = _synthetic_ds()
    # col0 is salt at the surface, then fresh, then salt again -> get_isosurface's first *down*-crossing
    # would place it deep; the physical interface (and the old logic) is the surface.
    conc = _conc([[8000, 4000, 500, 1200], [50, 80, 90, 120], [3000, 4000, 5000, 6000]])
    gv = interface_elevation(ds, conc, _THRESHOLD)
    old = _old_grensvlak(conc, _THRESHOLD, ds)
    assert gv.isel(icell2d=0).item() == ds["top"].isel(icell2d=0).item()
    assert gv.isel(icell2d=0).item() == old.isel(icell2d=0).item()


def test_interface_first_up_crossing_when_fresh_topped():
    """Fresh at the surface with a fresh/salt/fresh/salt profile takes the shallowest up-crossing."""
    ds = _synthetic_ds()
    conc = _conc([[500, 2000, 500, 2000], [50, 80, 90, 120], [3000, 4000, 5000, 6000]])
    gv = interface_elevation(ds, conc, _THRESHOLD).isel(icell2d=0).item()
    # first crossing is between the layer-0 centre -5 (conc 500) and the layer-1 centre -15 (conc 2000),
    # so the layer-0 centre - and hence ds["top"] - is exercised
    z_upper, z_lower, c_upper, c_lower = -5.0, -15.0, 500.0, 2000.0
    expected = z_upper + (_THRESHOLD - c_upper) / (c_lower - c_upper) * (z_lower - z_upper)
    np.testing.assert_allclose(gv, expected)


def test_interface_trailing_nan_column():
    """Columns whose deepest cells are inactive (NaN) still resolve to model bottom/top, not NaN."""
    ds = _synthetic_ds()
    conc = _conc([[100, 200, np.nan, np.nan], [3000, 4000, np.nan, np.nan], [50, 80, 90, 120]])
    gv = interface_elevation(ds, conc, _THRESHOLD)
    assert gv.isel(icell2d=0).item() == ds["botm"].isel(layer=-1, icell2d=0).item()  # fresh -> bottom
    assert gv.isel(icell2d=1).item() == ds["top"].isel(icell2d=1).item()  # salt -> top


def test_interface_deepens_with_threshold():
    """On a downward-increasing profile the interface deepens as the threshold rises (incl. 8000)."""
    ds = _synthetic_ds()
    profile = [200, 2000, 6000, 9000]
    conc = _conc([profile, profile, profile])
    depths = [interface_elevation(ds, conc, thr).isel(icell2d=0).item() for thr in (1000.0, 4000.0, 8000.0)]
    assert depths[0] > depths[1] > depths[2]


# --- check_budget_discrepancy against a real MODFLOW listing -------------------------------


@pytest.mark.parametrize(
    ("pct_disc", "max_gwf_pct", "expect_raise"),
    [
        (2.0, 4.0, False),  # comfortably below the threshold
        (2.0, 2.0, True),  # exactly at it: the guard is inclusive (>=)
        (-2.0, 2.0, True),  # a negative discrepancy is compared on its absolute value
    ],
)
def test_check_budget_real_listing(tmp_path, pct_disc, max_gwf_pct, expect_raise):
    """The guard trips at (not just above) the threshold when flopy really parses the listing.

    Values are integral so the float32 column flopy produces holds them exactly and the
    ``>=`` boundary is unambiguous.
    """
    write_mf6_listing(tmp_path / "test.lst", pct_disc)

    def run():
        pp.check_budget_discrepancy(str(tmp_path), "test", transport=False, max_gwf_pct=max_gwf_pct)

    if expect_raise:
        with pytest.raises(RuntimeError, match="discrepancy too large"):
            run()
    else:
        run()


def test_check_budget_unparsable_listing(tmp_path):
    """A listing flopy cannot parse raises rather than silently passing the guard."""
    (tmp_path / "test.lst").write_text("this is not a MODFLOW listing\n")
    with pytest.raises(RuntimeError, match="Could not parse"):
        pp.check_budget_discrepancy(str(tmp_path), "test", transport=False)


# --- add_output_to_ds ----------------------------------------------------------------------

# Three layers of thickness 4, 4 and 8 m below top=0: cell centres at -2, -6 and -12 m NAP and
# a thickness sum of 16, so every weighted mean below is exactly representable.
_BOTM = (-4.0, -8.0, -16.0)
_Z_CENTRE = np.array([-2.0, -6.0, -12.0])
_THICKNESS = np.array([4.0, 4.0, 8.0])
_THRESHOLD_BRAK = 8000.0
# chloride 500/1500/14500 over thicknesses 4/4/8: (4*500 + 4*1500 + 8*14500) / 16
_PROFILE = np.array([500.0, 1500.0, 14500.0])
_MEAN_T0 = 7750.0


def _output_ds(ntime=2):
    """Two cells x three layers, with the density attrs ``freshwater_head`` reads from ds."""
    ds = make_rect_vertex_ds(nx=2, ny=1, botm=_BOTM, transport=1)
    ds.attrs["drhodc"] = 0.5
    return ds, pd.to_datetime([f"2022-{m + 1:02d}-01" for m in range(ntime)])


def _da(values, time, layer):
    """Broadcast a per-layer sequence to a ``(time, layer, icell2d)`` DataArray of 2 cells."""
    arr = np.broadcast_to(np.asarray(values, float)[None, :, None], (len(time), len(layer), 2))
    return xr.DataArray(arr.copy(), dims=("time", "layer", "icell2d"), coords={"time": time, "layer": layer})


def _patch_loaders(monkeypatch, head, conc):
    """Stub the three nlmod readers of the .hds/.ucn binaries (named seam, no MODFLOW run).

    Returns the list the transport readers append to, so a test can assert they stayed unused.
    """
    calls = []
    monkeypatch.setattr(pp.nlmod.gwf.output, "get_heads_da", lambda *_a, **_kw: head)

    def _conc(*_a, **_kw):
        calls.append("conc")
        return conc

    def _ctop(c, **_kw):
        calls.append("ctop")
        return c.isel(layer=0)

    monkeypatch.setattr(pp.nlmod.gwt.output, "get_concentration_da", _conc)
    monkeypatch.setattr(pp.nlmod.gwt.output, "get_concentration_at_gw_surface", _ctop)
    return calls


def test_add_output_freshwater_head_and_head_fill(monkeypatch):
    """The density correction is an identity at zero chloride and exact and signed below it.

    With denseref=1024 and drhodc=0.5, chloride 1024 gives density 1536, so the two factors
    density/denseref = 1.5 and (density - denseref)/denseref = 0.5 are exact binary fractions.
    """
    ds, time = _output_ds()
    layer = ds.layer.values
    # layer 0 is chloride-free (rho == rho_ref -> hf must equal the head itself)
    conc = _da([0.0, 1024.0, 1024.0], time, layer)
    head = _da([2.0, 2.0, 2.0], time, layer)
    head[0, 0, 0] = np.nan  # one dry cell, filled from the layer below by bfill
    _patch_loaders(monkeypatch, head, conc)

    ds, _ = pp.add_output_to_ds(ds, "ws", "test", denseref=1024.0)

    # bfill must leave every finite head untouched and fill the planted gap from layer 1
    finite = np.isfinite(head.values)
    np.testing.assert_array_equal(ds["head_filled"].values[finite], head.values[finite])
    assert ds["head_filled"].values[0, 0, 0] == head.values[0, 1, 0]

    # hf = rho/rho_ref * h - (rho - rho_ref)/rho_ref * z, with h = 2 everywhere after the fill
    expected = np.array([2.0, 1.5 * 2.0 - 0.5 * _Z_CENTRE[1], 1.5 * 2.0 - 0.5 * _Z_CENTRE[2]])
    np.testing.assert_array_equal(ds["freshwater_head"].values, _da(expected, time, layer).values)
    # the zero-chloride layer is the pure identity: any scaling of the density term breaks it
    np.testing.assert_array_equal(ds["freshwater_head"].isel(layer=0).values, ds["head_filled"].isel(layer=0).values)


def test_add_output_concentration_mean_and_grensvlak(monkeypatch):
    """The mean is thickness-weighted and the two thresholds reach the right grensvlak.

    Chloride 500/1500/14500 crosses 1000 exactly halfway between the layer-0 and layer-1
    centres and 8000 exactly halfway between the layer-1 and layer-2 centres.
    """
    ds, time = _output_ds()
    layer = ds.layer.values
    conc = _da(_PROFILE, time, layer)
    conc[1] *= 2.0  # a second time step so the change vs. t=0 is not trivially zero
    _patch_loaders(monkeypatch, _da([1.0, 1.0, 1.0], time, layer), conc)

    ds, ctop = pp.add_output_to_ds(ds, "ws", "test")

    # thickness-weighted: 7750; an unweighted mean of the same profile would give 5500 instead
    mean0 = float(_PROFILE @ _THICKNESS) / _THICKNESS.sum()
    assert mean0 == _MEAN_T0
    expected_mean = np.full((2, 2), mean0)
    expected_mean[1] = 2.0 * mean0
    np.testing.assert_array_equal(ds["concentration_mean"].values, expected_mean)
    np.testing.assert_array_equal(ds["dconcentration_mean"].values, expected_mean - mean0)

    # midpoint crossings: fresh between centres -2 and -6, brackish between -6 and -12
    zoet = (_Z_CENTRE[0] + _Z_CENTRE[1]) / 2.0
    brak = (_Z_CENTRE[1] + _Z_CENTRE[2]) / 2.0
    assert (zoet, brak) == (-4.0, -9.0)
    np.testing.assert_array_equal(ds["grensvlak_zoet"].isel(time=0).values, np.full(2, zoet))
    np.testing.assert_array_equal(ds["grensvlak_brak"].isel(time=0).values, np.full(2, brak))
    # the fresh interface is never deeper than the brackish one, and the thresholds are not swapped
    assert (ds["grensvlak_zoet"] >= ds["grensvlak_brak"]).all()
    assert ds["grensvlak_zoet"].attrs["threshold"] == _THRESHOLD
    assert ds["grensvlak_brak"].attrs["threshold"] == _THRESHOLD_BRAK

    np.testing.assert_array_equal(ctop.values, conc.isel(layer=0).values)


def test_add_output_without_transport_skips_the_ucn(monkeypatch):
    """``transport=False`` returns ``ctop=None`` and never reads the transport output."""
    ds, time = _output_ds()
    layer = ds.layer.values
    calls = _patch_loaders(monkeypatch, _da([1.0, 1.0, 1.0], time, layer), _da([0.0, 0.0, 0.0], time, layer))

    ds, ctop = pp.add_output_to_ds(ds, "ws", "test", transport=False)

    assert ctop is None
    assert calls == []
    assert "concentration" not in ds
    assert "freshwater_head" not in ds


# --- plot_result_maps ----------------------------------------------------------------------

_GRID_MAP = {"doorsnedelijnen.png"}
_DRN_MAP = {"oppervlaktewater.png"}
# nper=3 with iper=-1 must normalise to 2; a sign flip instead of nper+iper would give 1
_TRANSPORT_MAPS = {
    "map_head_L0_t2.png",
    "map_conc_L0_t2.png",
    "grensvlak_zoet_t2.png",
    "grensvlak_brak_t2.png",
}


def _plot_ds(*, transport, drn):
    """Build a 2x2 vertex ds with three stress periods and the fields plot_result_maps consumes."""
    ds = make_rect_vertex_ds()
    time = pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"])
    ds = ds.assign_coords(time=time)
    ncell = ds.sizes["icell2d"]
    ctop = None
    if drn:
        ds["drn_elev"] = xr.DataArray(np.full(ncell, -1.0), dims=("icell2d",))
    if transport:
        ds["freshwater_head"] = xr.DataArray(
            np.zeros((3, ds.sizes["layer"], ncell)), dims=("time", "layer", "icell2d"), coords={"time": time}
        )
        for name, threshold in (("zoet", _THRESHOLD), ("brak", _THRESHOLD_BRAK)):
            da = xr.DataArray(np.full((3, ncell), -10.0), dims=("time", "icell2d"), coords={"time": time})
            da.attrs["threshold"] = threshold
            ds[f"grensvlak_{name}"] = da
        ctop = xr.DataArray(np.full((3, ncell), 100.0), dims=("time", "icell2d"), coords={"time": time})
    return ds, ctop


@pytest.mark.parametrize(
    ("transport", "drn", "expected"),
    [
        (True, False, _GRID_MAP | _TRANSPORT_MAPS),
        (True, True, _GRID_MAP | _TRANSPORT_MAPS | _DRN_MAP),
        (False, False, _GRID_MAP),  # early return: no freshwater_head, ctop=None
    ],
)
def test_plot_result_maps_filenames(tmp_path, transport, drn, expected):
    """Exactly the expected figures are written, with iper=-1 normalised against nper=3."""
    ds, ctop = _plot_ds(transport=transport, drn=drn)
    # named seam: add_background_map fetches contextily tiles, the module's only network call
    with mock.patch.object(pp.nlmod.plot, "add_background_map"):
        pp.plot_result_maps(ds, ctop, str(tmp_path))

    assert {p.name for p in tmp_path.iterdir()} == expected
