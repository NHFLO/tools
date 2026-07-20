"""Builders for tiny synthetic model datasets.

nlmod's own ``tests/util.py`` is not importable from an installed nlmod, so the few
helpers the nhflotools suite needs are built here. Everything is hand-built: no gridgen,
no MODFLOW binaries and no network, which keeps the unit tests to milliseconds.
"""

import flopy
import geopandas as gpd
import nlmod
import numpy as np
import xarray as xr
from shapely.geometry import box


def make_rect_vertex_ds(
    nx=2,
    ny=2,
    delr=100.0,
    botm=(-10.0, -20.0),
    top=0.0,
    kh=5.0,
    transport=0,
):
    """Build a rectangular vertex (DISV) model dataset.

    Cells are numbered row-major from the north-west corner, matching MODFLOW's
    convention. The grid is regular, so every cell area is ``delr**2`` exactly.

    Parameters
    ----------
    nx, ny : int
        Number of cells in the x and y direction.
    delr : float
        Cell size [m]; cells are square.
    botm : sequence of float
        Layer bottom elevations [mNAP], one per layer.
    top : float
        Surface elevation [mNAP].
    kh : float
        Horizontal conductivity [m/day].
    transport : int
        Value of the ``transport`` attribute nlmod's package builders read.

    Returns
    -------
    xarray.Dataset
        Vertex dataset with ``top``, ``botm``, ``kh``, ``kv``, ``area`` and ``idomain``,
        the ``xv``/``yv``/``icvert`` grid geometry and the attributes nlmod requires.
    """
    botm = np.asarray(botm, dtype=float)
    nlay = botm.size
    ncell = nx * ny
    extent = [0.0, nx * delr, 0.0, ny * delr]

    # Cell centres, row-major from the north-west corner.
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (ix.ravel() + 0.5) * delr
    y = extent[3] - (iy.ravel() + 0.5) * delr

    # Vertices of the (nx+1) x (ny+1) lattice, numbered the same way.
    vx, vy = np.meshgrid(np.arange(nx + 1) * delr, extent[3] - np.arange(ny + 1) * delr)
    vx, vy = vx.ravel(), vy.ravel()

    def vertex_id(row, col):
        return row * (nx + 1) + col

    icvert = np.array(
        [
            [
                vertex_id(r, c),
                vertex_id(r, c + 1),
                vertex_id(r + 1, c + 1),
                vertex_id(r + 1, c),
            ]
            for r in range(ny)
            for c in range(nx)
        ],
        dtype=int,
    )

    ds = xr.Dataset(
        data_vars={
            "top": ("icell2d", np.full(ncell, float(top))),
            "botm": (("layer", "icell2d"), np.tile(botm[:, None], (1, ncell))),
            "kh": (("layer", "icell2d"), np.full((nlay, ncell), float(kh))),
            "kv": (("layer", "icell2d"), np.full((nlay, ncell), float(kh) / 10.0)),
            "area": ("icell2d", np.full(ncell, delr**2)),
            "idomain": (("layer", "icell2d"), np.ones((nlay, ncell), dtype=int)),
            "xv": ("iv", vx),
            "yv": ("iv", vy),
            "icvert": (("icell2d", "nvert"), icvert),
        },
        coords={
            "layer": np.arange(nlay),
            "icell2d": np.arange(ncell),
            "iv": np.arange(vx.size),
            "x": ("icell2d", x),
            "y": ("icell2d", y),
        },
        attrs={
            "gridtype": "vertex",
            "extent": extent,
            "model_name": "test",
            "mfversion": "mf6",
            "model_ws": ".",
            "transport": transport,
        },
    )
    ds["icvert"].attrs["nodata"] = -1
    return ds


def add_time(ds, start="2022-01-01"):
    """Add a single steady-state stress period, required before building sim/tdis."""
    return nlmod.time.set_ds_time(ds, start=start, time=[1.0], steady=True)


def make_gwf_disv(ds, model_ws):
    """Build an in-memory sim/gwf/disv from a vertex ds. Never written, never run."""
    ds = add_time(ds)
    ds.attrs["model_ws"] = str(model_ws)
    # Packages are only built in memory, never written or run, so the exe need not exist.
    sim = nlmod.sim.sim(ds, exe_name="mf6")
    nlmod.sim.tdis(ds, sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.disv(ds, gwf)
    return ds, gwf


def cell_polygon(ds, icell2d):
    """Return the square polygon of one cell of a ``make_rect_vertex_ds`` grid."""
    delr = np.sqrt(float(ds["area"].isel(icell2d=icell2d)))
    x = float(ds["x"].isel(icell2d=icell2d))
    y = float(ds["y"].isel(icell2d=icell2d))
    return box(x - delr / 2, y - delr / 2, x + delr / 2, y + delr / 2)


def make_gdf(geometries, crs="EPSG:28992", **columns):
    """Build a GeoDataFrame from geometries plus scalar-or-sequence columns."""
    n = len(geometries)
    data = {k: (v if isinstance(v, (list, tuple, np.ndarray)) else [v] * n) for k, v in columns.items()}
    return gpd.GeoDataFrame(data, geometry=list(geometries), crs=crs)


def write_mf6_listing(path, pct_disc, budgetkey="VOLUME BUDGET FOR ENTIRE MODEL"):
    """Write a minimal MF6 listing file that flopy's Mf6ListBudget can parse.

    The layout mirrors a real MF6 listing: a volume budget block with matching IN/OUT
    entries, the percent-discrepancy line, and the time-summary block flopy needs to
    attach a time index to the budget.
    """
    inflow, outflow = 100.0, 100.0 * (1.0 - pct_disc / 100.0)
    text = f"""
  {budgetkey} AT END OF TIME STEP    1, STRESS PERIOD   1
  ------------------------------------------------------------------------------

     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T
     ------------------                 ------------------------

           IN:                                      IN:
           ---                                      ---
                 CHD =           0.0000                   CHD ={inflow:>17.4f}

            TOTAL IN =           0.0000            TOTAL IN ={inflow:>17.4f}

          OUT:                                     OUT:
          ----                                     ----
                 CHD =           0.0000                   CHD ={outflow:>17.4f}

           TOTAL OUT =           0.0000           TOTAL OUT ={outflow:>17.4f}

            IN - OUT =           0.0000            IN - OUT ={inflow - outflow:>17.4f}

 PERCENT DISCREPANCY =           0.00     PERCENT DISCREPANCY ={pct_disc:>17.2f}


  TIME SUMMARY AT END OF TIME STEP    1 IN STRESS PERIOD    1
                       SECONDS     MINUTES      HOURS       DAYS        YEARS
                       -----------------------------------------------------------
   TIME STEP LENGTH  86400.      1440.0      24.000      1.0000     2.73785E-03
 STRESS PERIOD TIME  86400.      1440.0      24.000      1.0000     2.73785E-03
         TOTAL TIME  86400.      1440.0      24.000      1.0000     2.73785E-03
"""
    path.write_text(text)
    return path


def make_structured_ds(extent=(0.0, 200.0, 0.0, 200.0), delr=100.0, **kwargs):
    """Tiny structured ds via nlmod's own builder, with executable download suppressed."""
    return nlmod.get_ds(list(extent), delr=delr, download_exe=False, **kwargs)


__all__ = [
    "add_time",
    "cell_polygon",
    "flopy",
    "make_gdf",
    "make_gwf_disv",
    "make_rect_vertex_ds",
    "make_structured_ds",
    "write_mf6_listing",
]
