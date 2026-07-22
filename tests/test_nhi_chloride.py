"""Tests for nhflotools.nhi_chloride.get_nhi_chloride_concentration.

The source data is a tiny hand-written ``chloride_p50.nc`` with descending y (as the real
NHI file has), so the nearest-neighbour interpolation onto the model grid is exact and the
thickness-weighted mean per model layer has an integer-valued known answer.
"""

import numpy as np
import pytest
import xarray as xr

from nhflotools.nhi_chloride import SEA_CHLORIDE_MG_L, get_nhi_chloride_concentration
from tests.util import make_rect_vertex_ds

# Source voxels of the default fixture: [0, -4] m at 100 mg/l over [-4, -8] m at 300 mg/l.
_VOXEL_TOP = (0.0, -4.0)
_VOXEL_BOTTOM = (-4.0, -8.0)
_VOXEL_C = (100.0, 300.0)

# Cell centres of the default 2x2 vertex grid (extent 0-200 m, row-major from the NW corner).
_CELL_X = (50.0, 150.0)
_CELL_Y = (150.0, 50.0)


def _write_chloride_nc(directory, values, top=_VOXEL_TOP, bottom=_VOXEL_BOTTOM, x=_CELL_X, y=_CELL_Y):
    """Write a minimal ``chloride_p50.nc`` and return the directory holding it.

    Parameters
    ----------
    directory : pathlib.Path
        Directory the file is written into.
    values : array_like
        Either one concentration per source layer (constant in x and y) or a full
        ``(layer, y, x)`` array.
    top, bottom : sequence of float
        Voxel top/bottom elevations [mNAP], one per source layer.
    x, y : sequence of float
        Source cell centres; ``y`` is descending, matching the real NHI file.

    Returns
    -------
    str
        Path of the directory, ready to pass as ``data_path_nhi_chloride``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = np.tile(values[:, None, None], (1, len(y), len(x)))
    da = xr.DataArray(
        values,
        dims=("layer", "y", "x"),
        coords={
            "layer": np.arange(len(top)),
            "y": list(y),
            "x": list(x),
            "top": ("layer", list(top)),
            "bottom": ("layer", list(bottom)),
        },
    )
    xr.Dataset({"chloride_p50": da}).to_netcdf(directory / "chloride_p50.nc")
    return str(directory)


def _model_ds(top=0.0, botm=(-4.0, -8.0), northsea=None):
    """Build a 2x2 vertex model dataset with the ``northsea`` variable the function requires."""
    ds = make_rect_vertex_ds(top=top, botm=botm)
    n = ds.sizes["icell2d"]
    ds["northsea"] = ("icell2d", np.zeros(n, dtype=int) if northsea is None else np.asarray(northsea, dtype=int))
    return ds


@pytest.mark.parametrize(
    ("top", "botm", "expected"),
    [
        # One model layer spanning both voxels: (4*100 + 4*300) / 8 = 200.
        (0.0, (-8.0,), (200.0,)),
        # Model layers coincide with the voxels, so each layer reproduces its voxel exactly.
        (0.0, (-4.0, -8.0), (100.0, 300.0)),
        # Model layer [-1, -6] clips both voxels: (3*100 + 2*300) / 5 = 180.
        (-1.0, (-6.0,), (180.0,)),
    ],
    ids=["spans-both-voxels", "voxel-aligned", "clips-both-voxels"],
)
def test_thickness_weighted_mean_per_model_layer(tmp_path, top, botm, expected):
    """The per-layer value is the voxel mean weighted by the clipped overlap thickness."""
    path = _write_chloride_nc(tmp_path, _VOXEL_C)
    ds = _model_ds(top=top, botm=botm)

    da = get_nhi_chloride_concentration(ds, path)

    assert da.dims == ("layer", "icell2d")
    assert da.attrs["units"] == "mg/l"
    # The source is uniform in x and y, so every cell of a layer carries the same value.
    np.testing.assert_array_equal(da.values, np.repeat(np.array(expected)[:, None], ds.sizes["icell2d"], axis=1))


def test_nan_voxel_leaves_denominator_untouched(tmp_path):
    """A NaN voxel drops out of numerator *and* denominator, so the answer is undiluted."""
    path = _write_chloride_nc(tmp_path, [np.nan, 300.0])
    ds = _model_ds(botm=(-8.0,))

    da = get_nhi_chloride_concentration(ds, path)

    # Both voxels are 4 m thick; keeping the NaN voxel's thickness in the denominator
    # would halve the result to 150.0.
    np.testing.assert_array_equal(da.values, np.full((1, 4), 300.0))


def test_sea_override_applies_to_layer_zero_only(tmp_path):
    """``northsea == 1`` forces the sea concentration in layer 0 and nowhere else."""
    path = _write_chloride_nc(tmp_path, _VOXEL_C)
    ds = _model_ds(northsea=[1, 0, 0, 0])

    da = get_nhi_chloride_concentration(ds, path)

    expected = np.array([
        [SEA_CHLORIDE_MG_L, 100.0, 100.0, 100.0],
        [300.0, 300.0, 300.0, 300.0],
    ])
    np.testing.assert_array_equal(da.values, expected)


def test_model_layers_outside_source_stack_are_filled_vertically(tmp_path):
    """Layers above and below the voxel stack are rescued by the bfill/ffill pass."""
    path = _write_chloride_nc(tmp_path, _VOXEL_C)
    # Layer 0 = [4, 0] sits entirely above the voxels and layer 3 = [-8, -12] entirely
    # below, so both aggregate to 0/0 = NaN before the fill.
    ds = _model_ds(top=4.0, botm=(0.0, -4.0, -8.0, -12.0))

    da = get_nhi_chloride_concentration(ds, path)

    assert not da.isnull().any()
    # bfill carries layer 1 up into layer 0; ffill carries layer 2 down into layer 3.
    expected = np.repeat(np.array([100.0, 100.0, 300.0, 300.0])[:, None], 4, axis=1)
    np.testing.assert_array_equal(da.values, expected)


def test_cells_outside_source_extent_take_the_nearest_neighbour(tmp_path):
    """Uncovered cells are filled horizontally from the nearest covered cell, not extrapolated."""
    # The source only covers y = 150 and y = 100, and its two rows differ, so the southern
    # model cells (y = 50) fall outside its extent and interpolate to NaN.
    values = np.array([
        [[100.0, 900.0], [500.0, 700.0]],
        [[300.0, 1100.0], [600.0, 800.0]],
    ])
    path = _write_chloride_nc(tmp_path, values, y=(150.0, 100.0))
    ds = _model_ds()

    da = get_nhi_chloride_concentration(ds, path)

    # Cell 2 (50, 50) is 100 m from cell 0 and 141 m from cell 1, so it inherits cell 0;
    # cell 3 likewise inherits cell 1. Extrapolating the source instead would yield the
    # y = 100 row, i.e. 500/700 and 600/800.
    expected = np.array([
        [100.0, 900.0, 100.0, 900.0],
        [300.0, 1100.0, 300.0, 1100.0],
    ])
    np.testing.assert_array_equal(da.values, expected)


def test_cached_call_reproduces_the_uncached_result(tmp_path):
    """The cache_netcdf round trip returns the same values and attrs as a direct call."""
    path = _write_chloride_nc(tmp_path, _VOXEL_C)
    cachedir = tmp_path / "cache"
    cachedir.mkdir()

    reference = get_nhi_chloride_concentration(_model_ds(northsea=[1, 0, 0, 0]), path)
    first = get_nhi_chloride_concentration(
        _model_ds(northsea=[1, 0, 0, 0]), path, cachedir=str(cachedir), cachename="chloride"
    )
    second = get_nhi_chloride_concentration(
        _model_ds(northsea=[1, 0, 0, 0]), path, cachedir=str(cachedir), cachename="chloride"
    )

    assert (cachedir / "chloride.nc").exists()
    assert (cachedir / "chloride.pklz").exists()
    np.testing.assert_array_equal(first.values, reference.values)
    np.testing.assert_array_equal(second.values, reference.values)
    assert second.attrs["units"] == "mg/l"
