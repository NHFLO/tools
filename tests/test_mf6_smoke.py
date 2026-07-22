"""End-to-end MODFLOW 6 run behind ``postprocessing``'s output loaders.

Every other test in the suite monkeypatches ``nlmod.gwf.output`` and hands
``check_budget_discrepancy`` a synthetic listing. This module is the only one that
writes real ``.lst``/``.hds``/``.grb`` files with the MODFLOW 6 executable and reads them
back through the production code path, so a change in nlmod/flopy output handling that
the mocked tests cannot see shows up here.
"""

import nlmod
import pytest
import xarray as xr

from nhflotools.postprocessing import add_output_to_ds, check_budget_discrepancy
from tests.util import make_structured_ds


pytestmark = pytest.mark.mf6


@pytest.fixture(scope="session")
def mf6_exe():
    """Full path of the MODFLOW 6 executable, downloading it once if it is missing.

    ``nlmod.util.get_exe_path`` searches nlmod's own ``bin`` directory and then flopy's
    metadata, so a previously downloaded executable is reused and the download happens at
    most once per machine -- including on a fresh CI runner, which is why this test needs
    no binaries to be installed beforehand.
    """
    return nlmod.util.get_exe_path(exe_name="mf6", download_if_not_found=True)


# A 3x3 x 2-layer grid of 100 m cells on extent [0, 300] x [0, 300]. Cell centres are at
# x, y in {50, 150, 250}, so (150, 150) is the one cell not on the perimeter.
_NX = _NY = 3
_NLAY = 2
_DELR = 100.0
_CENTRE = 150.0
# h = (2*150 + 150) / 50, which is also the mean of the four neighbours: (5 + 13 + 7 + 11) / 4.
_CENTRE_HEAD = 9.0


def _analytic_head(ds):
    """Prescribed head field ``h = (2x + y) / 50`` [m NAP] on the cell centres.

    A field linear in x and y is harmonic, so it is the exact steady-state solution of
    the finite-volume equations once the whole perimeter is held at its own value. The
    coefficients are deliberately unequal in x and y: a swapped or mirrored axis in the
    ``.grb``/``.hds`` reading path changes the field, whereas a symmetric field would
    survive it. With ``x`` in {50, 150, 250} and ``y`` in {250, 150, 50} every head is a
    small integer, so the comparison needs no tolerance for representation error.
    """
    return (2.0 * ds["x"] + ds["y"]) / 50.0


def test_tiny_run_budget_and_output_loading(tmp_path, mf6_exe):
    """A real MF6 run closes its budget and reloads as the analytic head field.

    Builds the smallest model that still has an unknown: 3x3 cells x 2 layers, the eight
    perimeter cells held constant at ``h = (2x + y) / 50`` in both layers, one free
    interior column. Then, on the files MODFLOW actually wrote,
    :func:`check_budget_discrepancy` must accept the listing and
    :func:`add_output_to_ds` must return the analytic field.
    """
    ws = str(tmp_path)
    model_name = "smoke"
    ds = make_structured_ds(
        extent=(0.0, _NX * _DELR, 0.0, _NY * _DELR),
        delr=_DELR,
        top=0.0,
        botm=[-10.0, -20.0],
        kh=10.0,
        kv=1.0,
        model_name=model_name,
        model_ws=ws,
    )
    ds = nlmod.time.set_ds_time(ds, start="2022-01-01", time=[1.0], steady=True)

    analytic = _analytic_head(ds)
    ds["chd_head"] = analytic.broadcast_like(ds["botm"]).transpose("layer", "y", "x")
    # Everything but the single centre cell is a constant-head cell.
    perimeter = (ds["x"] != _CENTRE) | (ds["y"] != _CENTRE)
    ds["chd_mask"] = perimeter.broadcast_like(ds["botm"]).transpose("layer", "y", "x").astype(int)

    sim = nlmod.sim.sim(ds, exe_name=mf6_exe)
    nlmod.sim.tdis(ds, sim)
    # Tighter than the default so the residual is far below the 1e-8 comparison below.
    nlmod.sim.ims(sim, complexity="SIMPLE", outer_dvclose=1e-9, inner_dvclose=1e-10)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)
    nlmod.gwf.npf(ds, gwf)
    # Start far from the answer so the interior column is genuinely solved for.
    nlmod.gwf.ic(ds, gwf, starting_head=0.0)
    nlmod.gwf.chd(ds, gwf)
    nlmod.gwf.oc(ds, gwf)
    nlmod.sim.write_and_run(sim, ds, write_ds=False, silent=True)

    # Only constant-head cells exchange water, so the volumetric budget must close; a
    # wrong budget key or an inverted threshold turns this into a RuntimeError.
    check_budget_discrepancy(ws, model_name, transport=False)

    ds, ctop = add_output_to_ds(ds, ws, model_name, transport=False)
    assert ctop is None
    assert "concentration" not in ds

    # The constant-head cells must come back at the head they were given, and the free
    # centre cell at the mean of its four neighbours. ``expected`` is aligned on the x/y
    # coordinates, so a mirrored or swapped axis in the .grb/.hds path is a mismatch.
    # atol accommodates the IMS iterative solver, which stops at outer_dvclose rather
    # than at algebraic exactness.
    expected = analytic.broadcast_like(ds["head"])
    assert float(expected.isel(time=0, layer=0).sel(x=_CENTRE, y=_CENTRE)) == _CENTRE_HEAD
    xr.testing.assert_allclose(ds["head"], expected, rtol=0.0, atol=1e-8)
    assert ds["head"].sizes == {"time": 1, "layer": _NLAY, "y": _NY, "x": _NX}
