"""Tests for the PWN layer model cross-section plotting helpers."""

import geopandas
import numpy as np
import pytest
from matplotlib.figure import Figure
from shapely.geometry import Point

from nhflotools.pwnlayers3.layers import layer_names
from nhflotools.pwnlayers3.plot import (
    _load_and_project_source_botm,  # noqa: PLC2701
    _overlay_source_botm,  # noqa: PLC2701
    _parse_flag_labels,  # noqa: PLC2701
)

# The three flag_meanings strings that pwnlayers3.layers attaches to the diagnostic
# method arrays, copied verbatim from layers.py (botm_method, kh_method, kv_method).
# They are duplicated here on purpose: the test is only meaningful if its input is an
# independent transcription of what production attaches, not an import of it.
BOTM_METHOD_MEANINGS = (
    "0: no_data (outside boundary polygon); "
    "1: linear_interpolation (griddata linear from botm.geojson point data); "
    "2: nearest_interpolation (griddata nearest fallback where linear produced NaN); "
    "3: forward_fill (missing botm filled from layer above by fix_missings_botms_and_min_layer_thickness); "
    "4: shifted_for_min_thickness (botm shifted downward to enforce monotonically decreasing sequence)"
)
BOTM_METHOD_FLAG_VALUES = [0, 1, 2, 3, 4]

KH_METHOD_MEANINGS = (
    "0: no_data (outside boundary polygon); "
    "1: W_layer_polygon_kh (direct kh from area-weighted K polygon data); "
    "2: S_layer_NHDZ_KD_linear (kh = KD/d, KD from linear interpolation of point data); "
    "3: S_layer_NHDZ_KD_nearest (kh = KD/d, KD from nearest-neighbor interpolation fallback); "
    "4: S_layer_Bergen_c_to_kh (kh = d*anisotropy/c, c from harmonic area-weighted polygon data); "
    "5: S_layer_c_to_kh (kh = d*anisotropy/c, c from harmonic area-weighted polygon data, full extent); "
    "6: fill_value (zero-thickness cell, set to fill_value_kh)"
)
KH_METHOD_FLAG_VALUES = [0, 1, 2, 3, 4, 5, 6]

KV_METHOD_MEANINGS = (
    "0: no_data (outside boundary polygon); "
    "1: W_layer_kh_anisotropy (kv = kh/anisotropy); "
    "2: S_layer_d_over_c (kv = d/c, c from harmonic area-weighted polygon data); "
    "3: fill_value (zero-thickness cell, set to fill_value_kv)"
)
KV_METHOD_FLAG_VALUES = [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("meanings", "flag_values", "expected"),
    [
        pytest.param(
            BOTM_METHOD_MEANINGS,
            BOTM_METHOD_FLAG_VALUES,
            [
                "no data",
                "linear interpolation",
                "nearest interpolation",
                "forward fill",
                "shifted for min thickness",
            ],
            id="botm_method",
        ),
        pytest.param(
            KH_METHOD_MEANINGS,
            KH_METHOD_FLAG_VALUES,
            [
                "no data",
                "W layer polygon kh",
                "S layer NHDZ KD linear",
                "S layer NHDZ KD nearest",
                "S layer Bergen c to kh",
                "S layer c to kh",
                "fill value",
            ],
            id="kh_method",
        ),
        pytest.param(
            KV_METHOD_MEANINGS,
            KV_METHOD_FLAG_VALUES,
            ["no data", "W layer kh anisotropy", "S layer d over c", "fill value"],
            id="kv_method",
        ),
    ],
)
def test_parse_flag_labels_matches_flag_values_of_production_strings(meanings, flag_values, expected):
    """Every production flag_meanings string yields one label per flag value.

    ``_plot_method_cross_section`` only applies the tick labels when
    ``len(labels) == len(flag_values)``; a parser that produces one label too many or
    too few silently leaves the colorbar showing raw integers instead of failing.
    """
    labels = _parse_flag_labels(meanings)

    assert labels == expected
    assert len(labels) == len(flag_values)


@pytest.mark.parametrize(
    ("meanings", "expected"),
    [
        # A trailing separator must not produce a phantom empty label, which would
        # break the len(labels) == n_flags gate.
        pytest.param("0: a; 1: b;", ["a", "b"], id="trailing-semicolon"),
        pytest.param("0: a;;1: b", ["a", "b"], id="empty-entry"),
        # No "N:" prefix: the whole entry is kept as the label.
        pytest.param("plain_label", ["plain label"], id="no-flag-prefix"),
        # Everything from the first "(" on is descriptive detail and is dropped.
        pytest.param("7: short (long (nested) tail)", ["short"], id="parenthetical-dropped"),
        # Multi-digit flag numbers are prefixes too, not part of the label.
        pytest.param("10: ten_th_flag", ["ten th flag"], id="multi-digit-prefix"),
    ],
)
def test_parse_flag_labels_adversarial_inputs(meanings, expected):
    """Malformed or unusual flag_meanings entries are normalised, not mis-split."""
    assert _parse_flag_labels(meanings) == expected


# Cross-section line along the x-axis; ``project`` of a point therefore returns its
# x-coordinate exactly and ``distance`` returns |y|.
LINE = [(0.0, 0.0), (100.0, 0.0)]
BUFFER = 50.0


@pytest.fixture
def botm_geojson(tmp_path):
    """Write a four-point ``botm/botm.geojson`` with a shuffled, non-Range index.

    Perpendicular distances to ``LINE`` are 30, 80, 50 and 50 m, so with
    ``BUFFER = 50`` the second point is excluded and the two points sitting exactly on
    the buffer boundary are kept (the comparison is ``<=``).
    """
    gdf = geopandas.GeoDataFrame(
        {
            # Column order deliberately differs from layer_names order (W11 precedes
            # S13 there) and includes a column that is not a layer at all.
            "S13": [100.0, 0.0, 4.0, -200.0],
            "OBJECTID": [1, 2, 3, 4],
            "W11": [10.0, 999.0, 5.0, -5.0],
        },
        geometry=[Point(20.0, 30.0), Point(40.0, 80.0), Point(60.0, -50.0), Point(80.0, 50.0)],
        index=[7, 3, 9, 1],
        crs="EPSG:28992",
    )
    fp = tmp_path / "botm" / "botm.geojson"
    fp.parent.mkdir()
    gdf.to_file(fp, driver="GeoJSON")
    return tmp_path


def test_load_and_project_source_botm_selects_by_buffer_and_orders_by_layer_names(botm_geojson):
    """Points are filtered by perpendicular distance and keyed in layer_names order."""
    result = _load_and_project_source_botm(data_path_2024=botm_geojson, line=LINE, buffer_distance=BUFFER)

    # Points 1, 3 and 4 survive; ``project`` onto the x-axis returns their x-coordinate.
    np.testing.assert_array_equal(result["d_along"], [20.0, 60.0, 80.0])

    # Keys follow layer_names order (W11 is index 0, S13 index 5), not the column order
    # in the file. The values are positionally aligned with d_along, so any reordering
    # of the keys without reordering the values would silently mis-pair z with x.
    assert list(result["layers"]) == ["W11", "S13"]
    assert layer_names.get_loc("W11") < layer_names.get_loc("S13")
    np.testing.assert_array_equal(result["layers"]["W11"], [10.0, 5.0, -5.0])
    np.testing.assert_array_equal(result["layers"]["S13"], [100.0, 4.0, -200.0])


def test_overlay_source_botm_draws_only_layers_with_in_window_points(botm_geojson):
    """Only layers with at least one finite, in-window z become a scatter collection."""
    source_points = _load_and_project_source_botm(data_path_2024=botm_geojson, line=LINE, buffer_distance=BUFFER)
    # An all-NaN layer is the case a layer model produces outside the PWN boundary.
    source_points["layers"]["S11"] = np.full(3, np.nan)

    zmin, zmax = -120.0, 25.0
    ax = Figure().subplots()
    _overlay_source_botm(ax, source_points, zmin=zmin, zmax=zmax)

    # W11 = [10, 5, -5] -> all three inside [-120, 25]; S13 = [100, 4, -200] -> only
    # the middle point survives; S11 is all NaN -> no collection at all.
    n_drawn_layers = 2
    assert len(ax.collections) == n_drawn_layers
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[20.0, 10.0], [60.0, 5.0], [80.0, -5.0]])
    np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[60.0, 4.0]])
