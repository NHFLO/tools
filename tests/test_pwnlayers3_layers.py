"""Tests for the PWN layer-model seams of ``nhflotools.pwnlayers3.layers``.

The pure helpers are pinned with hand-derived arithmetic: the botm repair (which lives as
two divergent copies), thickness telescoping, the zero-thickness guard, the in-place
griddata fill, and the REGIS/PWN merge that splits, routes and interpolates layers.
"""

import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from nhflotools.pwnlayers.merge_layer_models import (
    _apply_ratios_to_botm,
    _compute_thickness_ratios,
    _interpolate_da,
    combine_two_layer_models,
)
from nhflotools.pwnlayers.utils import (
    fix_missings_botms_and_min_layer_thickness as fix_botms_utils,
)
from nhflotools.pwnlayers3.layers import _guard_zero_thickness, get_kv, get_pwn_layer_model, get_thickness
from nhflotools.pwnlayers3.layers import (
    fix_missings_botms_and_min_layer_thickness as fix_botms_pwnlayers3,
)
from nhflotools.pwnlayers3.layers import layer_names as PWN_LAYER_NAMES

from .util import make_rect_vertex_ds

# Header names combine_two_layer_models defaults to; the koppeltabel columns must match.
_H_REGIS = "Regis II v2.2"
_H_OTHER = "ASSUMPTION1"
_CRS = "EPSG:28992"


def _cell_coords(ds):
    """Coordinate mapping shared by every DataArray built on the grid of ``ds``."""
    return {"icell2d": ds.icell2d, "x": ds.x, "y": ds.y}


def _spread(values, ncell):
    """Turn a per-layer sequence of scalars into a ``(layer, icell2d)`` array."""
    return np.asarray(values, dtype=float)[:, None] * np.ones((1, ncell))


def _layer_model(ds, layers, botm, kh, kv):
    """Build a REGIS-shaped layer model with uniform values per layer."""
    ncell = ds.sizes["icell2d"]
    return xr.Dataset(
        {
            "botm": (("layer", "icell2d"), _spread(botm, ncell)),
            "kh": (("layer", "icell2d"), _spread(kh, ncell)),
            "kv": (("layer", "icell2d"), _spread(kv, ncell)),
        },
        coords={"layer": list(layers), **_cell_coords(ds)},
        attrs={"extent": ds.attrs["extent"], "gridtype": ds.attrs["gridtype"]},
    )


def _flags(ds, layers, cells_true):
    """Boolean mask/transition Dataset that is True on ``cells_true`` for every layer."""
    data = np.zeros((len(layers), ds.sizes["icell2d"]), dtype=bool)
    data[:, list(cells_true)] = True
    return xr.Dataset(
        {var: (("layer", "icell2d"), data.copy()) for var in ("botm", "kh", "kv")},
        coords={"layer": list(layers), **_cell_coords(ds)},
    )


def _top(ds, value=0.0):
    return xr.DataArray(np.full(ds.sizes["icell2d"], float(value)), dims="icell2d", coords=_cell_coords(ds), name="top")


def _koppeltabel(rows):
    return pd.DataFrame(list(rows), columns=[_H_REGIS, _H_OTHER])


@pytest.mark.parametrize(
    "fix_botms",
    [fix_botms_pwnlayers3, fix_botms_utils],
    ids=["pwnlayers3.layers", "pwnlayers.utils"],
)
def test_fix_missings_botms_is_pure_monotone_and_idempotent(fix_botms):
    """Both divergent copies fill NaNs downward, clip crossings and leave the input alone.

    Parametrized over the two copies on purpose: editing one and not the other is the
    standing maintenance trap in this codebase.
    """
    top = xr.DataArray(np.zeros(3), dims="icell2d", coords={"icell2d": [0, 1, 2]})
    botm = xr.DataArray(
        np.array([
            [-10.0, -10.0, np.nan],
            [np.nan, -5.0, np.nan],
            [-30.0, -20.0, -8.0],
        ]),
        dims=("layer", "icell2d"),
        coords={"layer": ["a", "b", "c"], "icell2d": [0, 1, 2]},
    )
    # ffill down the column with `top` prepended, then a running minimum:
    #   cell 0: [0, -10, nan, -30] -> ffill [0, -10, -10, -30] -> already decreasing.
    #   cell 1: [0, -10,  -5, -20] -> -5 lies above -10, so it is pulled down to -10.
    #   cell 2: [0, nan, nan,  -8] -> leading NaNs are filled from the top elevation, 0.
    expected = np.array([
        [-10.0, -10.0, 0.0],
        [-10.0, -10.0, 0.0],
        [-30.0, -20.0, -8.0],
    ])
    before = botm.copy(deep=True)

    out = fix_botms(top=top, botm=botm)

    np.testing.assert_array_equal(out.values, expected)
    assert out.dims == ("layer", "icell2d")
    assert not out.isnull().any()
    assert (out.diff(dim="layer") <= 0.0).all(), "layer bottoms must be non-increasing downward"
    assert (out <= top).all()
    # Purity: the caller keeps only the return value, so mutating the argument would corrupt
    # the source model silently.
    np.testing.assert_array_equal(botm.values, before.values)
    # Idempotence: re-running the repair on repaired botms must be a no-op.
    np.testing.assert_array_equal(fix_botms(top=top, botm=out).values, expected)

    with pytest.raises(ValueError, match="nan"):
        fix_botms(top=top.where(top.icell2d != 1), botm=botm)


def test_get_thickness_telescopes_and_labels_the_lower_layer():
    """thickness[k] == botm[k-1] - botm[k], labelled with the lower layer; the top layer drops out."""
    botm = xr.DataArray(
        np.array([[-2.0, -2.0], [-6.0, -10.0], [-14.0, -12.0]]),
        dims=("layer", "icell2d"),
        coords={"layer": ["W11", "S11", "W12"], "icell2d": [0, 1]},
    )
    thickness = get_thickness(botm=botm)

    # W11 needs the model top and is therefore absent; the label is the *lower* of each pair.
    assert list(thickness.layer.values) == ["S11", "W12"]
    np.testing.assert_array_equal(thickness.values, np.array([[4.0, 8.0], [8.0, 2.0]]))
    # Telescoping: the column sums back to the distance between the first and last bottom.
    np.testing.assert_array_equal(
        thickness.sum(dim="layer").values, botm.isel(layer=0).values - botm.isel(layer=-1).values
    )


def test_guard_zero_thickness_replaces_only_the_isclose_zero_cells():
    """Cells within np.isclose of zero thickness take the fill value; the rest are untouched."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    # 1e-12 is inside np.isclose's default atol of 1e-8, 0.5 and 2.0 are not.
    guarded = _guard_zero_thickness(values.copy(), np.array([0.0, 1e-12, 0.5, 2.0]), 7.0, "S11")
    np.testing.assert_array_equal(guarded, [7.0, 7.0, 3.0, 4.0])

    identity = _guard_zero_thickness(values.copy(), np.array([0.5, 1.0, 2.0, 4.0]), 7.0, "S11")
    np.testing.assert_array_equal(identity, values)


def test_interpolate_da_writes_through_to_the_parent_dataset():
    """The fill reaches the parent Dataset, honours the method, and no-ops when nothing is missing.

    ``_interpolate_ds`` hands ``_interpolate_da`` a ``.sel(layer=...)`` view and relies on the
    ``.loc`` assignment propagating back; an xarray copy-semantics change would silently turn
    the whole transition interpolation into a no-op.
    """
    # Four corners of a 4x4 square carry data; the cell at (1, 1) is missing.
    x = np.array([0.0, 4.0, 0.0, 4.0, 1.0])
    y = np.array([0.0, 0.0, 4.0, 4.0, 1.0])
    plane = x + 2.0 * y  # any linear interpolant reproduces an affine field exactly
    sentinel = -999.0
    values = np.stack([plane, plane])
    values[:, 4] = sentinel
    ds = xr.Dataset(
        {"kh": (("layer", "icell2d"), values)},
        coords={"layer": ["A", "B"], "icell2d": np.arange(5), "x": ("icell2d", x), "y": ("icell2d", y)},
    )
    isvalid = xr.DataArray([True] * 4 + [False], dims="icell2d", coords={"icell2d": np.arange(5)})
    ismissing = ~isvalid
    nothing_missing = xr.zeros_like(isvalid, dtype=bool)

    _interpolate_da(ds["kh"].sel(layer="A"), isvalid=isvalid, ismissing=ismissing, method="linear")
    _interpolate_da(ds["kh"].sel(layer="B"), isvalid=isvalid, ismissing=ismissing, method="nearest")

    # rtol accommodates Qhull barycentric arithmetic; the field is affine so the plane value
    # 1 + 2*1 = 3 is triangulation-independent.
    np.testing.assert_allclose(ds["kh"].sel(layer="A").values[4], 3.0, rtol=1e-12)
    # Nearest donor of (1, 1) is the corner (0, 0), at distance sqrt(2) against 3.16 for the others.
    assert ds["kh"].sel(layer="B").values[4] == 0.0
    # Valid cells are never rewritten.
    np.testing.assert_array_equal(ds["kh"].values[:, :4], np.stack([plane[:4], plane[:4]]))

    ds["kh"].loc[{"layer": "A", "icell2d": 4}] = sentinel
    _interpolate_da(ds["kh"].sel(layer="A"), isvalid=isvalid, ismissing=nothing_missing, method="linear")
    assert ds["kh"].sel(layer="A").values[4] == sentinel


def test_combine_two_layer_models_reduces_to_regis_when_pwn_is_absent():
    """An all-False mask with a 1:1 koppeltabel must hand back REGIS untouched."""
    ds = make_rect_vertex_ds(nx=2, ny=2)
    top = _top(ds)
    regis = _layer_model(ds, ["A", "B", "C"], botm=[-8.0, -16.0, -24.0], kh=[1.0, 2.0, 4.0], kv=[0.1, 0.2, 0.4])
    other = _layer_model(ds, ["p", "q", "r"], botm=[-2.0, -6.0, -12.0], kh=[16.0, 32.0, 64.0], kv=[1.6, 3.2, 6.4])

    out, cat = combine_two_layer_models(
        layer_model_regis=regis,
        layer_model_other=other,
        mask_model_other=_flags(ds, ["p", "q", "r"], cells_true=[]),
        transition_model=_flags(ds, ["p", "q", "r"], cells_true=[]),
        top=top,
        df_koppeltabel=_koppeltabel([("A", "p"), ("B", "q"), ("C", "r")]),
        split_method="nearest_ratio",
    )

    for var in ("botm", "kh", "kv"):
        np.testing.assert_array_equal(out[var].values, regis[var].values)
    for var in ("botm", "kh", "kv"):
        np.testing.assert_array_equal(cat[var].values, np.ones_like(regis[var].values, dtype=int))


@pytest.mark.parametrize(
    ("split_method", "botm_a1_regis_cells"),
    [
        # 'equal' halves the 8 m REGIS layer A; 'nearest_ratio' takes the 2 m : 6 m split of the
        # PWN pair (p, q) and applies it to A, giving 0 - (2/8)*8 = -2.
        ("equal", -4.0),
        ("nearest_ratio", -2.0),
    ],
)
def test_combine_two_layer_models_routes_and_conserves_split_thickness(split_method, botm_a1_regis_cells):
    """Split layers take their values from the right source model and preserve group bottoms.

    The koppeltabel mixes a 1:2 REGIS split (A -> p, q), a 2:1 PWN split (B, C -> r) and one
    uncoupled deep REGIS layer (D). Cells 0 and 1 are PWN, cells 2 and 3 are REGIS.
    """
    ds = make_rect_vertex_ds(nx=2, ny=2)
    top = _top(ds)
    regis = _layer_model(
        ds, ["A", "B", "C", "D"], botm=[-8.0, -16.0, -24.0, -40.0], kh=[1.0, 2.0, 4.0, 8.0], kv=[0.1, 0.2, 0.4, 0.8]
    )
    other = _layer_model(ds, ["p", "q", "r"], botm=[-2.0, -8.0, -20.0], kh=[16.0, 32.0, 64.0], kv=[1.6, 3.2, 6.4])
    pwn_cells, regis_cells = [0, 1], [2, 3]

    out, cat = combine_two_layer_models(
        layer_model_regis=regis,
        layer_model_other=other,
        mask_model_other=_flags(ds, ["p", "q", "r"], cells_true=pwn_cells),
        transition_model=_flags(ds, ["p", "q", "r"], cells_true=[]),
        top=top,
        df_koppeltabel=_koppeltabel([("A", "p"), ("A", "q"), ("B", "r"), ("C", "r"), ("D", np.nan)]),
        split_method=split_method,
    )

    assert list(out.layer.values) == ["A_1", "A_2", "B_1", "C_1", "D"]
    # Category 2 = PWN, 1 = REGIS; the uncoupled layer D is REGIS everywhere.
    np.testing.assert_array_equal(cat["kh"].values, np.array([[2, 2, 1, 1]] * 4 + [[1, 1, 1, 1]]).reshape(5, 4))

    # Routing: sublayers inherit the conductivity of the layer they were split from.
    kh_pwn = np.array([16.0, 32.0, 64.0, 64.0, 8.0])  # p, q, r, r, REGIS D
    kh_regis = np.array([1.0, 1.0, 2.0, 4.0, 8.0])  # A, A, B, C, D
    np.testing.assert_array_equal(out["kh"].sel(icell2d=pwn_cells).values, np.tile(kh_pwn[:, None], (1, 2)))
    np.testing.assert_array_equal(out["kh"].sel(icell2d=regis_cells).values, np.tile(kh_regis[:, None], (1, 2)))
    np.testing.assert_array_equal(out["kv"].sel(icell2d=pwn_cells).values, np.tile(kh_pwn[:, None] / 10.0, (1, 2)))

    # Bottoms: PWN cells get p, q and the equal halves of r between -8 and -20; REGIS cells get
    # the A split (method dependent) plus the untouched B, C. D is REGIS in every cell.
    botm_pwn = np.array([-2.0, -8.0, -14.0, -20.0, -40.0])
    botm_regis = np.array([botm_a1_regis_cells, -8.0, -16.0, -24.0, -40.0])
    np.testing.assert_array_equal(out["botm"].sel(icell2d=pwn_cells).values, np.tile(botm_pwn[:, None], (1, 2)))
    np.testing.assert_array_equal(out["botm"].sel(icell2d=regis_cells).values, np.tile(botm_regis[:, None], (1, 2)))

    # Conservation: splitting redistributes thickness inside a group but never changes the
    # group's total thickness, i.e. the group bottom equals the source layer's bottom.
    thickness = get_thickness(botm=xr.concat([top.expand_dims(layer=["mv"]), out["botm"]], dim="layer"))
    group_a = thickness.sel(layer=["A_1", "A_2"]).sum(dim="layer")
    group_r = thickness.sel(layer=["B_1", "C_1"]).sum(dim="layer")
    np.testing.assert_array_equal(group_a.sel(icell2d=regis_cells).values, [8.0, 8.0])  # REGIS A: 0 - -8
    np.testing.assert_array_equal(group_a.sel(icell2d=pwn_cells).values, [8.0, 8.0])  # PWN p+q: 0 - -8
    np.testing.assert_array_equal(group_r.sel(icell2d=pwn_cells).values, [12.0, 12.0])  # PWN r: -8 - -20
    np.testing.assert_array_equal(group_r.sel(icell2d=regis_cells).values, [16.0, 16.0])  # REGIS B+C


def test_combine_two_layer_models_interpolates_the_transition_band():
    """Transition cells are interpolated between the PWN and REGIS values, not copied from either.

    The 3x3 grid is banded north to south: PWN row, transition row, REGIS row. Because each band
    carries one constant value, the valid data is an affine function of y and the interpolated
    middle row is the exact midpoint whatever triangulation scipy picks.
    """
    ds = make_rect_vertex_ds(nx=3, ny=3)
    top = _top(ds)
    regis = _layer_model(ds, ["A"], botm=[-16.0], kh=[2.0], kv=[0.5])
    other = _layer_model(ds, ["p"], botm=[-4.0], kh=[8.0], kv=[2.0])
    pwn_cells, transition_cells, regis_cells = [0, 1, 2], [3, 4, 5], [6, 7, 8]

    out, cat = combine_two_layer_models(
        layer_model_regis=regis,
        layer_model_other=other,
        mask_model_other=_flags(ds, ["p"], cells_true=pwn_cells),
        transition_model=_flags(ds, ["p"], cells_true=transition_cells),
        top=top,
        df_koppeltabel=_koppeltabel([("A", "p")]),
        split_method="nearest_ratio",
    )

    np.testing.assert_array_equal(cat["kh"].values, [[2, 2, 2, 3, 3, 3, 1, 1, 1]])
    # rtol accommodates Qhull barycentric arithmetic on the affine field.
    for var, pwn_value, regis_value in [("kh", 8.0, 2.0), ("kv", 2.0, 0.5), ("botm", -4.0, -16.0)]:
        values = out[var].sel(layer="A_1").values
        np.testing.assert_array_equal(values[pwn_cells], [pwn_value] * 3)
        np.testing.assert_array_equal(values[regis_cells], [regis_value] * 3)
        np.testing.assert_allclose(values[transition_cells], [(pwn_value + regis_value) / 2.0] * 3, rtol=1e-12)


def test_thickness_ratios_round_trip_through_apply_ratios_to_botm():
    """Ratios reproduce the source split exactly, fall back to 1/N and spread by nearest neighbour.

    The group starts at the second layer, so ``_compute_thickness_ratios`` and
    ``_apply_ratios_to_botm`` must both take the group top from the layer above (the
    ``first_idx - 1`` lookup) rather than from the model top.
    """
    ds = make_rect_vertex_ds(nx=3, ny=1)
    top = _top(ds)
    layers = ["X", "Y", "Z"]
    botm = np.array([
        [-4.0, -4.0, -4.0],  # X
        [-8.0, -4.0, -8.0],  # Y: cell 1 is a collapsed, zero-thickness group
        [-20.0, -4.0, -20.0],  # Z
    ])
    source = xr.Dataset(
        {"botm": (("layer", "icell2d"), botm), "top": top},
        coords={"layer": layers, **_cell_coords(ds)},
    )
    # Cell 2 has no source data and must inherit from its nearest valid neighbour, cell 1
    # (100 m away) rather than cell 0 (200 m away).
    mask_valid = xr.DataArray([True, True, False], dims="icell2d", coords=_cell_coords(ds))

    ratios = _compute_thickness_ratios(source, ["Y", "Z"], mask_valid)

    # cell 0: group top is botm X = -4, so Y is 4 m of the 16 m group and Z is 12 m -> 1/4, 3/4.
    # cell 1: zero group thickness -> equal ratios 1/2.  cell 2: nearest copy of cell 1.
    np.testing.assert_array_equal(
        ratios.transpose("layer", "icell2d").values, np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    )
    np.testing.assert_array_equal(ratios.sum(dim="layer").values, np.ones(3))

    target = source.copy(deep=True)
    _apply_ratios_to_botm(target, top, ["Y", "Z"], ratios)

    # cell 0 round-trips to its own botm; cell 1 stays collapsed; cell 2 gets the halved split
    # of its 16 m group: -4 - 0.5*16 = -12.
    np.testing.assert_array_equal(target["botm"].sel(layer="Y").values, [-8.0, -4.0, -12.0])
    # The group bottom is preserved in every cell, so total group thickness is unchanged.
    np.testing.assert_array_equal(target["botm"].sel(layer="Z").values, botm[2])


def _write_resistance_polygons(data_path, values, boxes):
    """Write one C<layer>_combined.geojson per aquitard, holding constant-c polygons."""
    gdf = gpd.GeoDataFrame({"VALUE": list(values)}, geometry=[box(*b) for b in boxes], crs=_CRS)
    conductances = data_path / "conductances"
    conductances.mkdir(parents=True, exist_ok=True)
    for name in PWN_LAYER_NAMES:
        if name.startswith("S"):
            gdf.to_file(conductances / f"C{name}_combined.geojson", driver="GeoJSON")


def test_get_kv_uses_harmonic_area_weighting_and_anisotropy(tmp_path):
    """Aquitards get kv = d/c with 1/c averaged by area; aquifers get kv = kh/anisotropy."""
    ds = make_rect_vertex_ds(nx=2, ny=1)
    # Cell 0 spans x in [0, 100] and is halved by the polygon boundary at x = 50; cell 1
    # (x in [100, 200]) lies wholly inside the second polygon.
    _write_resistance_polygons(tmp_path, values=[2.0, 4.0], boxes=[(0, 0, 50, 200), (50, 0, 200, 200)])

    nlay, ncell = len(PWN_LAYER_NAMES), ds.sizes["icell2d"]
    thickness = xr.DataArray(
        np.full((nlay, ncell), 8.0),
        dims=("layer", "icell2d"),
        coords={"layer": PWN_LAYER_NAMES, "icell2d": ds.icell2d},
    )
    thickness.loc[{"layer": "S11", "icell2d": 1}] = 0.0  # collapsed cell -> fill value
    kh = xr.DataArray(
        np.full((nlay, ncell), 30.0),
        dims=("layer", "icell2d"),
        coords={"layer": PWN_LAYER_NAMES, "icell2d": ds.icell2d},
    )

    kv = get_kv(
        ds=ds,
        data_path_2024=tmp_path,
        kh=kh,
        thickness=thickness,
        anisotropy=10.0,
        fill_value_kv=1.0,
        isin_bounds=np.ones((nlay, ncell), dtype=bool),
    )

    # Aquifers: 30 / 10.
    for name in [n for n in PWN_LAYER_NAMES if n.startswith("W")]:
        np.testing.assert_array_equal(kv.sel(layer=name).values, [3.0, 3.0])
    # Aquitards: 1/c averaged over the cell, then kv = d * (1/c).
    # cell 0: 0.5*(1/2) + 0.5*(1/4) = 0.375 -> 8 * 0.375 = 3.0.  cell 1: 1/4 -> 8 * 0.25 = 2.0.
    for name in [n for n in PWN_LAYER_NAMES if n.startswith("S") and n != "S11"]:
        np.testing.assert_array_equal(kv.sel(layer=name).values, [3.0, 2.0])
    np.testing.assert_array_equal(kv.sel(layer="S11").values, [3.0, 1.0])


# ── Offline integration of get_pwn_layer_model ──────────────────────────────────────────
# A synthetic 4x4 grid of 100 m cells over [0, 400]^2, with a hand-written PWN data tree.
# The PWN boundary covers x < 200 (columns 0 and 1), so with a 150 m transition buffer the
# grid splits into PWN columns 0-1, a transition column 2 and a REGIS-only column 3.  The
# NHDZ region covers y > 200, splitting the PWN block into a KD half and a Bergen half.
_MASKED_NHDZ, _MASKED_BERGEN = [0, 1, 4, 5], [8, 9, 12, 13]
_TRANSITION_CELLS, _REGIS_CELLS = [2, 6, 10, 14], [3, 7, 11, 15]
_REGIS_LAYERS = ["mv", "A", "B", "C", "D", "E"]
_KOPPELTABEL_GROUPS = [("A", 4), ("B", 4), ("C", 4), ("D", 2)]
# Layer k of the PWN model has its bottom at -2*(k+1) minus a x/200 tilt, so every PWN layer
# is exactly 2 m thick and linear interpolation of the source points is exact.
_PWN_BASE_BOTM = -2.0 * (np.arange(len(PWN_LAYER_NAMES)) + 1.0)
_C_VALUE, _KD_VALUE, _K_VALUE, _ANISOTROPY, _REGIS_KH = 4.0, 8.0, 20.0, 10.0, 5.0


def _write_pwn_data_tree(path):
    """Write the boundary, botm, conductance and NHDZ GeoJSONs get_ds reads."""
    points = [Point(x, y) for x in (0.0, 100.0, 200.0, 300.0, 400.0) for y in (0.0, 100.0, 200.0, 300.0, 400.0)]
    xs = np.array([p.x for p in points])

    (path / "botm").mkdir(parents=True, exist_ok=True)
    botm_columns = {name: _PWN_BASE_BOTM[i] - xs / 200.0 for i, name in enumerate(PWN_LAYER_NAMES)}
    gpd.GeoDataFrame(botm_columns, geometry=points, crs=_CRS).to_file(path / "botm" / "botm.geojson", driver="GeoJSON")

    for unit in ["11", "12", "13", "21", "22", "31", "32"]:
        (path / "boundaries" / f"S{unit}").mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(geometry=[box(-1.0, -1.0, 201.0, 401.0)], crs=_CRS).to_file(
            path / "boundaries" / f"S{unit}" / f"S{unit}.geojson", driver="GeoJSON"
        )
    gpd.GeoDataFrame(geometry=[box(-1.0, 199.0, 401.0, 401.0)], crs=_CRS).to_file(
        path / "boundaries" / "triwaco_model_nhdz.geojson", driver="GeoJSON"
    )

    _write_resistance_polygons(path, values=[_C_VALUE], boxes=[(-1.0, -1.0, 401.0, 401.0)])
    conductances = path / "conductances"
    for name in PWN_LAYER_NAMES:
        if name.startswith("W"):
            gpd.GeoDataFrame({"VALUE": [_K_VALUE]}, geometry=[box(-1.0, -1.0, 401.0, 401.0)], crs=_CRS).to_file(
                conductances / f"K{name}_combined.geojson", driver="GeoJSON"
            )
    for name in ["S12", "S13", "S21", "S22", "S31"]:
        gpd.GeoDataFrame({"VALUE": [_KD_VALUE] * len(points)}, geometry=points, crs=_CRS).to_file(
            conductances / f"KD{name}_NHDZ.geojson", driver="GeoJSON"
        )


def _regis_input():
    """4x4 vertex ds standing in for REGIS, plus the model top and the koppeltabel rows."""
    ds = make_rect_vertex_ds(nx=4, ny=4, botm=(0.0, -10.0, -20.0, -30.0, -40.0, -120.0), kh=_REGIS_KH)
    ds = ds.assign_coords(layer=_REGIS_LAYERS)
    regis_per_pwn_layer = [group for group, n in _KOPPELTABEL_GROUPS for _ in range(n)]
    rows = list(zip(regis_per_pwn_layer, PWN_LAYER_NAMES, strict=True))
    return ds, _top(ds), _koppeltabel([*rows, ("E", np.nan)])


def _expected_pwn_kh(in_nhdz):
    """Kh the PWN model should produce per layer: K polygon, KD/d, or d*anisotropy/c."""
    d, inv_c = 2.0, 1.0 / _C_VALUE
    kh_from_c = d * _ANISOTROPY * inv_c
    return np.array([
        _K_VALUE
        if name.startswith("W")
        else (kh_from_c if name in {"S11", "S32"} else (_KD_VALUE / d if in_nhdz else kh_from_c))
        for name in PWN_LAYER_NAMES
    ])


def test_get_pwn_layer_model_offline_known_answers(tmp_path, monkeypatch):
    """The merged model carries the PWN values, in koppeltabel order, everywhere PWN is valid.

    ``nlmod.read.regis.get_layer_names`` is patched because it opens an OPeNDAP endpoint when
    called without a dataset; it is the only network seam in this pipeline.
    """
    monkeypatch.setattr(nlmod.read.regis, "get_layer_names", lambda: pd.Index(_REGIS_LAYERS, name="layer"))
    _write_pwn_data_tree(tmp_path)
    ds_regis, top, koppeltabel = _regis_input()
    fname_koppeltabel = tmp_path / "koppeltabel.csv"
    koppeltabel.to_csv(fname_koppeltabel)

    out = get_pwn_layer_model(
        ds_regis=ds_regis,
        data_path_2024=tmp_path,
        fname_koppeltabel=fname_koppeltabel,
        top=top,
        anisotropy=_ANISOTROPY,
        distance_transition=150.0,
        return_diagnostics=True,
    )

    # Split names follow the koppeltabel: A splits into 4, ..., D into 2, E stays uncoupled.
    expected_layers = [f"{g}_{i + 1}" for g, n in _KOPPELTABEL_GROUPS for i in range(n)] + ["E"]
    assert list(out.layer.values) == expected_layers

    coupled = out.sel(layer=expected_layers[:-1])
    np.testing.assert_array_equal(coupled["cat_kh"].values[:, _MASKED_NHDZ + _MASKED_BERGEN], 2)
    np.testing.assert_array_equal(coupled["cat_kh"].values[:, _TRANSITION_CELLS], 3)
    np.testing.assert_array_equal(coupled["cat_kh"].values[:, _REGIS_CELLS], 1)
    np.testing.assert_array_equal(out["cat_kh"].sel(layer="E").values, 1)

    # rtol accommodates Qhull barycentric arithmetic in the botm/KD interpolations, which
    # propagates into the thickness the conductivities are derived from.
    for cells, in_nhdz in [(_MASKED_NHDZ, True), (_MASKED_BERGEN, False)]:
        kh_expected = _expected_pwn_kh(in_nhdz)
        np.testing.assert_allclose(
            coupled["kh"].values[:, cells], np.tile(kh_expected[:, None], (1, len(cells))), rtol=1e-12
        )
        # Aquifers keep kv = kh/anisotropy; aquitards get kv = d/c = 2/4.
        kv_expected = np.where([n.startswith("W") for n in PWN_LAYER_NAMES], kh_expected / _ANISOTROPY, 2.0 / _C_VALUE)
        np.testing.assert_allclose(
            coupled["kv"].values[:, cells], np.tile(kv_expected[:, None], (1, len(cells))), rtol=1e-12
        )
        # Bottoms come from the PWN point cloud: -2*(k+1) tilted by -x/200.
        x = out.x.values[cells]
        np.testing.assert_allclose(
            coupled["botm"].values[:, cells], _PWN_BASE_BOTM[:, None] - x[None, :] / 200.0, rtol=1e-12
        )

    # Transition cells mix both models: strictly between REGIS kh (5) and PWN aquifer kh (20).
    aquifer_layers = [
        layer for layer, name in zip(expected_layers[:-1], PWN_LAYER_NAMES, strict=True) if name.startswith("W")
    ]
    kh_transition = out["kh"].sel(layer=aquifer_layers).values[:, _TRANSITION_CELLS]
    assert np.all(kh_transition > _REGIS_KH)
    assert np.all(kh_transition < _K_VALUE)

    # Postconditions the NPF package depends on: strictly positive, finite, monotone.
    assert np.all(out["kh"].values > 0.0)
    assert np.all(out["kv"].values > 0.0)
    assert np.all(np.isfinite(out["kh"].values))
    assert np.all(np.isfinite(out["kv"].values))
    assert (out["botm"].diff(dim="layer") <= 0.0).all()
    assert (out["botm"] <= out["top"]).all()
    np.testing.assert_array_equal(out["top"].values, top.values)
    # Values passthrough only: layers.py evaluates the get_area default eagerly (issue #61).
    np.testing.assert_array_equal(out["area"].values, ds_regis["area"].values)

    # The plot module ticks its colourbars off these attributes, and every code that actually
    # occurs must be declared.  Seeing 1/2/4/5 also proves all three kh branches ran.
    assert set(np.unique(out["kh_method"].values)) == {0, 1, 2, 4, 5}
    for name in ["botm_method", "kh_method", "kv_method"]:
        flag_values = set(out[name].attrs["flag_values"])
        assert set(np.unique(out[name].values)) <= flag_values
        assert len(out[name].attrs["flag_meanings"].split(";")) == len(flag_values)


@pytest.mark.parametrize(
    ("bad", "match"),
    [("top", "should not contain nan"), ("layers", "All REGIS layers should be present")],
)
def test_get_pwn_layer_model_rejects_nan_top_and_wrong_regis_layers(bad, match, monkeypatch):
    """The two fail-fast guards trip before any data is read."""
    monkeypatch.setattr(nlmod.read.regis, "get_layer_names", lambda: pd.Index(_REGIS_LAYERS, name="layer"))
    ds_regis, top, _ = _regis_input()
    if bad == "top":
        top = top.where(top.icell2d != 0)
    else:
        # Same number of layers, one renamed: the merge would otherwise silently couple the
        # koppeltabel to the wrong REGIS unit.
        ds_regis = ds_regis.assign_coords(layer=[*_REGIS_LAYERS[:-1], "X"])

    with pytest.raises(ValueError, match=match):
        get_pwn_layer_model(ds_regis=ds_regis, data_path_2024=None, fname_koppeltabel=None, top=top)


@pytest.mark.xfail(
    strict=True,
    reason="The guard compares layer names elementwise, so a missing layer raises pandas' "
    "'Lengths must match to compare' before the actionable message is reached. NHFLO/tools#65",
)
def test_get_pwn_layer_model_reports_a_missing_regis_layer(monkeypatch):
    """A dropped REGIS layer must be reported with the message that names the fix.

    This is what ``get_regis(.., remove_nan_layers=True)`` hands over, so it is the case
    the guard exists for -- and the one it currently fails to report usefully.
    """
    monkeypatch.setattr(nlmod.read.regis, "get_layer_names", lambda: pd.Index(_REGIS_LAYERS, name="layer"))
    ds_regis, top, _ = _regis_input()
    ds_regis = ds_regis.sel(layer=_REGIS_LAYERS[:-1])

    with pytest.raises(ValueError, match="All REGIS layers should be present"):
        get_pwn_layer_model(ds_regis=ds_regis, data_path_2024=None, fname_koppeltabel=None, top=top)
