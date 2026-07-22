"""Shared synthetic disv fixtures for the lake/polder tests.

These build a tiny vertex (disv) MODFLOW 6 model plus a matching xarray dataset, so the
lake carve/stage/mask helpers can be exercised without a solver, real PWN data, or a
network connection.
"""

import flopy
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box


def _build_disv(nrow=3, ncol=3, top=5.0, dz=10.0, transport=0):
    """Build an independent (ds, gwf, cell_geometries) synthetic disv model.

    Parameters
    ----------
    nrow, ncol : int, optional
        Number of rows/columns of 200 m square cells. The default is a 3x3 grid.
    top : float, optional
        Uniform model top, by default 5.0 m NAP.
    dz : float, optional
        Uniform layer thickness, by default 10.0 m (two layers).
    transport : int, optional
        Value stored under ``ds.attrs['transport']``, by default 0 (no transport).

    Returns
    -------
    ds : xarray.Dataset
        Dataset with ``top``, ``botm``, ``kh``, ``idomain`` and ``area``.
    gwf : flopy.mf6.ModflowGwf
        Groundwater flow model with a matching disv grid and a single stress period.
    geoms : list of shapely.geometry.Polygon
        Cell footprints, indexed by ``icell2d``.
    """
    delr = delc = 200.0
    nlay = 2
    ncpl = nrow * ncol

    verts = []
    vid = {}
    k = 0
    for j in range(nrow + 1):
        for i in range(ncol + 1):
            vid[(j, i)] = k
            verts.append([k, i * delr, (nrow - j) * delc])
            k += 1

    cell2d = []
    xc, yc, geoms = [], [], []
    for r in range(nrow):
        for c in range(ncol):
            icpl = r * ncol + c
            v = [vid[(r, c)], vid[(r, c + 1)], vid[(r + 1, c + 1)], vid[(r + 1, c)]]
            cx, cy = (c + 0.5) * delr, (nrow - r - 0.5) * delc
            xc.append(cx)
            yc.append(cy)
            cell2d.append([icpl, cx, cy, 4, *v])
            geoms.append(box(c * delr, (nrow - r - 1) * delc, (c + 1) * delr, (nrow - r) * delc))

    botm = np.array([[top - dz * (lay + 1)] * ncpl for lay in range(nlay)], dtype=float)

    sim = flopy.mf6.MFSimulation(sim_name="t", exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="t")
    flopy.mf6.ModflowGwfdisv(
        gwf, nlay=nlay, ncpl=ncpl, nvert=len(verts), top=top, botm=botm, vertices=verts, cell2d=cell2d
    )

    ds = xr.Dataset(
        data_vars={
            "top": ("icell2d", np.full(ncpl, top, dtype=float)),
            "botm": (("layer", "icell2d"), botm),
            "kh": (("layer", "icell2d"), np.full((nlay, ncpl), 10.0)),
            "idomain": (("layer", "icell2d"), np.ones((nlay, ncpl), dtype=int)),
            "area": ("icell2d", np.full(ncpl, delr * delc, dtype=float)),
        },
        coords={
            "layer": np.arange(nlay),
            "icell2d": np.arange(ncpl),
            "x": ("icell2d", np.array(xc)),
            "y": ("icell2d", np.array(yc)),
        },
    )
    ds.attrs["gridtype"] = "vertex"
    ds.attrs["transport"] = transport
    ds.attrs["extent"] = [0.0, ncol * delr, 0.0, nrow * delc]
    ds.attrs["ssm_sources"] = []
    return ds, gwf, geoms


@pytest.fixture
def disv_grid():
    """Return the :func:`_build_disv` factory (each call builds a fresh, independent model)."""
    return _build_disv
