"""Tests for nhflotools.postprocessing.

The interface-elevation helper supersedes the old layer-discretised ``grensvlak`` with
``nlmod.dims.get_isosurface``. These tests pin the intended behaviour: it agrees with the old
logic at the fresh/salt limits (per column) and interpolates the crossing in between.
"""

from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

import nhflotools.postprocessing as pp
from nhflotools.postprocessing import interface_elevation

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


def _run_budget_check(values, max_pct):
    """Call check_budget_discrepancy with Mf6ListBudget stubbed to a known discrepancy column."""
    fake = mock.Mock()
    fake.get_dataframes.return_value = [pd.DataFrame({"PERCENT_DISCREPANCY": values})]
    with mock.patch.object(pp.flopy.utils, "Mf6ListBudget", return_value=fake):
        pp.check_budget_discrepancy("ws", "m", transport=False, max_gwf_pct=max_pct)


def test_check_budget_discrepancy_uses_absolute_value():
    """A large negative discrepancy trips the guard (absolute value), a small one passes."""
    raised = False
    try:
        _run_budget_check([0.1, -5.0, 0.2], max_pct=1.0)
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError when |discrepancy| exceeds the threshold"

    _run_budget_check([0.1, -0.2, 0.3], max_pct=1.0)  # below threshold -> must not raise
