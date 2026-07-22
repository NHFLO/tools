"""Tests for nhflotools.pwnlayers.layers.get_top_from_ahn.

The function fills the gaps in the AHN surface in a fixed priority order: surface-water
peil first, then a constant over the North Sea, then interpolation from the remaining
valid cells. These tests pin that order, the full-coverage gate on the peil fill and the
(y, x) axis order handed to ``scipy.interpolate.griddata``.
"""

import nlmod
import numpy as np
import pytest
import xarray as xr

from nhflotools.pwnlayers.layers import get_top_from_ahn
from tests.util import make_rect_vertex_ds


def _ds_with_ahn(ahn, x=None, y=None):
    """Build a one-row vertex ds carrying an ``ahn`` variable.

    With both replacement flags off, ``get_top_from_ahn`` reads nothing but ``ds['ahn']``
    and its ``x``/``y`` coordinates, so the cell centres may be moved off the regular
    lattice to create a layout with unique nearest neighbours.

    Parameters
    ----------
    ahn : sequence of float
        AHN value per cell; NaN marks a gap to be filled.
    x, y : sequence of float, optional
        Replacement cell centres. Defaults to the regular grid of ``make_rect_vertex_ds``.

    Returns
    -------
    xarray.Dataset
        Vertex dataset with ``ahn`` on the ``icell2d`` dimension.
    """
    ahn = np.asarray(ahn, dtype=float)
    ds = make_rect_vertex_ds(nx=ahn.size, ny=1)
    if x is not None:
        ds = ds.assign_coords(
            x=("icell2d", np.asarray(x, dtype=float)),
            y=("icell2d", np.asarray(y, dtype=float)),
        )
    ds["ahn"] = ("icell2d", ahn)
    return ds


# Four donors and one gap, all off the x == y diagonal so that reversing one of the two
# coordinate tuples handed to griddata is detectable. Distances from the gap at
# (200, 600), in units of m**2:
#   A (0, 800):     200**2 + 200**2 =  80000  <- true nearest, value 1.0
#   B (800, 0):     600**2 + 600**2 = 720000
#   C (1000, 600):  800**2 +   0**2 = 640000
#   D (200, 0):       0**2 + 600**2 = 360000
# The implementation builds both donor and query tuples as (y, x). Reversing only the
# donors compares (x_d, y_d) against (600, 200) instead, giving
#   A: 600**2 + 600**2 = 720000    B: 200**2 + 200**2 =  80000  <- swapped nearest, 2.0
#   C: 400**2 + 400**2 = 320000    D: 400**2 + 200**2 = 200000
# so a one-sided axis swap moves the answer from 1.0 to 2.0.
_X = [0.0, 800.0, 1000.0, 200.0, 200.0]
_Y = [800.0, 0.0, 600.0, 0.0, 600.0]
_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]


def test_nearest_fill_is_euclidean_and_valid_cells_are_untouched():
    """The gap takes its Euclidean-nearest donor and valid cells pass through unchanged."""
    ahn = np.array(_VALUES, dtype=float)
    ahn[4] = np.nan

    top = get_top_from_ahn(
        _ds_with_ahn(ahn, x=_X, y=_Y),
        replace_surface_water_with_peil=False,
        replace_northsea_with_constant=None,
    )

    # Donor A at (0, 800) is nearest; the swapped metric would pick donor B at 2.0.
    expected = np.array([1.0, 2.0, 4.0, 8.0, 1.0])
    np.testing.assert_array_equal(top.values, expected)

    # No gaps at all: griddata is handed an empty query set and the field is returned
    # bit-identical (pins the scipy empty-xi edge).
    full = get_top_from_ahn(
        _ds_with_ahn(_VALUES, x=_X, y=_Y),
        replace_surface_water_with_peil=False,
        replace_northsea_with_constant=None,
    )
    np.testing.assert_array_equal(full.values, np.array(_VALUES))


def test_fill_priority_peil_then_sea_constant_then_nearest(monkeypatch):
    """Peil wins over the sea constant, 0.0 still fills, partial cover falls through.

    Five cells in a row at x = 50, 150, 250, 350, 450 (100 m cells, area 10000 m2):

    ===== ======================================== ================================
    cell  input                                    expected
    ===== ======================================== ================================
    0     NaN, water over the whole cell + sea     2.0 (peil, not the sea constant)
    1     NaN, sea only                            0.0 (falsy constant still fills)
    2     4.0                                      4.0
    3     8.0                                      8.0
    4     NaN, water over half the cell            8.0 (nearest donor, cell 3)
    ===== ======================================== ================================

    Cell 4 lies at the end of the row, so cell 3 at 100 m is its unique nearest donor
    (cell 2 is 200 m away); its stage of 99.0 would surface instead if the full-coverage
    gate were dropped.
    """
    ds = _ds_with_ahn([np.nan, np.nan, 4.0, 8.0, np.nan])
    icell2d = ds["icell2d"]
    seen = {}

    def fake_get_gdf_surface_water(extent=None, **_kwargs):
        seen["extent"] = extent
        return "gdf-sentinel"

    def fake_discretize_surface_water(_ds, gdf=None, **_kwargs):
        seen["gdf"] = gdf
        return xr.Dataset(
            {
                "rws_oppwater_area": ("icell2d", np.array([10000.0, 0.0, 0.0, 0.0, 5000.0])),
                "rws_oppwater_stage": ("icell2d", np.array([2.0, np.nan, np.nan, np.nan, 99.0])),
            },
            coords={"icell2d": icell2d},
        )

    def fake_discretize_northsea(_ds, **_kwargs):
        return xr.Dataset(
            {"northsea": ("icell2d", np.array([True, True, False, False, False]))},
            coords={"icell2d": icell2d},
        )

    # The rws readers download from live web services; they are the only seam mocked here.
    monkeypatch.setattr(nlmod.read.rws, "get_gdf_surface_water", fake_get_gdf_surface_water)
    monkeypatch.setattr(nlmod.read.rws, "discretize_surface_water", fake_discretize_surface_water)
    monkeypatch.setattr(nlmod.read.rws, "discretize_northsea", fake_discretize_northsea)

    top = get_top_from_ahn(
        ds,
        replace_surface_water_with_peil=True,
        replace_northsea_with_constant=0.0,
    )

    np.testing.assert_array_equal(top.values, np.array([2.0, 0.0, 4.0, 8.0, 8.0]))
    assert seen["extent"] == ds.extent
    assert seen["gdf"] == "gdf-sentinel"


def test_missing_ahn_raises_valueerror():
    """A ds without AHN is rejected up front rather than silently substituted."""
    with pytest.raises(ValueError, match="AHN"):
        get_top_from_ahn(make_rect_vertex_ds(), replace_surface_water_with_peil=False)
