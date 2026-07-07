"""Tests for NHFLO well helpers."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point

from nhflotools import well


def _tata_frame(names, q_values, coordinates):
    return gpd.GeoDataFrame(
        {
            "Name": names,
            "Q_m3/d": q_values,
            "marker": [f"original-{name}" for name in names],
        },
        geometry=[Point(x, y) for x, y in coordinates],
        crs="EPSG:28992",
    )


def _tata_dataset(kh_values, chloride_values=None, x_values=None, y_values=None):
    layer = np.arange(3)
    icell2d = np.arange(2)
    if x_values is None:
        x_values = [0.0, 100.0]
    if y_values is None:
        y_values = [0.0, 0.0]
    if chloride_values is None:
        chloride_values = np.full((3, 2), 500.0)

    return xr.Dataset(
        data_vars={
            "kh": (("layer", "icell2d"), np.asarray(kh_values, dtype=float)),
            "botm": (
                ("layer", "icell2d"),
                np.array(
                    [
                        [[8.0, 18.0], [2.0, 12.0], [-6.0, 4.0]],
                    ],
                    dtype=float,
                )[0],
            ),
            "top": ("icell2d", np.array([10.0, 20.0], dtype=float)),
            "chloride": (("layer", "icell2d"), np.asarray(chloride_values, dtype=float)),
            "extent": ("bounds", np.array([0.0, 120.0, -10.0, 10.0], dtype=float)),
        },
        coords={
            "layer": layer,
            "icell2d": icell2d,
            "x": ("icell2d", np.asarray(x_values, dtype=float)),
            "y": ("icell2d", np.asarray(y_values, dtype=float)),
        },
    )


def _patch_tata_reads(monkeypatch, salt_frame, fresh_frame):
    calls = []
    frames = {
        "tata_zoutwaterbronnen.geojson": salt_frame,
        "tata_zoetwaterbronnen.geojson": fresh_frame,
    }

    def fake_read_file(path, driver=None):
        path = Path(path)
        calls.append((path, driver))
        return frames[path.name].copy(deep=True)

    monkeypatch.setattr(well.gpd, "read_file", fake_read_file)
    return calls


def _patch_thickness(monkeypatch, values=None):
    if values is None:
        values = np.full((3, 2), 10.0)

    thickness = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("layer", "icell2d"),
        coords={"layer": np.arange(3), "icell2d": np.arange(2)},
    )

    monkeypatch.setattr(well.nlmod.dims, "calculate_thickness", lambda _ds: thickness)
    return thickness


def test_get_wells_tata_dataframes_reads_files_and_sets_base_columns(monkeypatch, tmp_path):
    """Tata helper reads exact inputs and derives base WEL columns."""
    salt_frame = _tata_frame(
        ["salt-a", "salt-b"],
        [1200.0, 600.0],
        [(10.0, 1.0), (20.0, 2.0)],
    )
    fresh_frame = _tata_frame(
        ["fresh-a", "fresh-b", "fresh-c"],
        [900.0, 300.0, 150.0],
        [(1.0, 0.0), (99.0, 0.0), (101.0, 0.0)],
    )
    calls = _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    expected_thickness = np.array(
        [
            [21.0, 10.0],
            [11.0, 12.0],
            [8.0, 9.0],
        ],
        dtype=float,
    )
    thickness = _patch_thickness(
        monkeypatch,
        values=expected_thickness.tolist(),
    )
    ds = _tata_dataset(
        kh_values=[
            [5.0, 11.0],
            [10.0, 9.0],
            [15.0, 13.0],
        ],
    )

    gdf_tata_zout, gdf_tata_zoet = well.get_wells_tata_dataframes(tmp_path, ds)

    assert calls == [
        (tmp_path / "tata_zoutwaterbronnen.geojson", "GeoJSON"),
        (tmp_path / "tata_zoetwaterbronnen.geojson", "GeoJSON"),
    ]
    assert isinstance(gdf_tata_zout, gpd.GeoDataFrame)
    assert isinstance(gdf_tata_zoet, gpd.GeoDataFrame)
    assert gdf_tata_zout.crs == salt_frame.crs
    assert gdf_tata_zoet.crs == fresh_frame.crs
    assert gdf_tata_zout.geometry.to_list() == salt_frame.geometry.to_list()
    assert gdf_tata_zoet.geometry.to_list() == fresh_frame.geometry.to_list()
    assert gdf_tata_zout["x"].to_list() == [10.0, 20.0]
    assert gdf_tata_zout["y"].to_list() == [1.0, 2.0]
    assert gdf_tata_zoet["x"].to_list() == [1.0, 99.0, 101.0]
    assert gdf_tata_zoet["y"].to_list() == [0.0, 0.0, 0.0]
    assert gdf_tata_zout["CONCENTRATION"].to_list() == [0.0, 0.0]
    assert gdf_tata_zoet["CONCENTRATION"].to_list() == [0.0, 0.0, 0.0]
    assert gdf_tata_zout["Q"].to_list() == [-600.0, -300.0]
    assert gdf_tata_zoet["Q"].to_list() == [-300.0, -100.0, -50.0]
    assert gdf_tata_zout["marker"].to_list() == ["original-salt-a", "original-salt-b"]
    assert gdf_tata_zoet["marker"].to_list() == [
        "original-fresh-a",
        "original-fresh-b",
        "original-fresh-c",
    ]
    assert "thickness" in ds
    assert ds["thickness"].dims == thickness.dims
    np.testing.assert_array_equal(ds["thickness"].values, expected_thickness)


def test_tata_fresh_wells_use_nearest_cell_first_strict_kd_layer_and_offsets(monkeypatch, tmp_path, caplog):
    """Fresh Tata screens use nearest cell, strict kd threshold, and layer offsets."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(
        ["fresh-near-cell-0", "fresh-near-cell-1"],
        [900.0, 900.0],
        [(1.0, 0.0), (101.0, 0.0)],
    )
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(
        monkeypatch,
        values=[
            [1.0, 10.0],
            [20.0, 1.0],
            [1.0, 1.0],
        ],
    )
    ds = _tata_dataset(
        kh_values=[
            [100.0, 11.0],
            [6.0, 9.0],
            [150.0, 13.0],
        ],
    )

    with caplog.at_level("WARNING", logger=well.logger.name):
        _, gdf_tata_zoet = well.get_wells_tata_dataframes(tmp_path, ds)

    first_well = gdf_tata_zoet.loc[0]
    second_well = gdf_tata_zoet.loc[1]

    assert first_well["Name"] == "fresh-near-cell-0"
    assert first_well["botm"] == 2.0 + 0.001
    assert first_well["top"] == 8.0 - 0.001
    assert second_well["Name"] == "fresh-near-cell-1"
    assert second_well["botm"] == 18.0 + 0.001
    assert second_well["top"] == 20.0 - 0.001
    assert not caplog.records


def test_tata_fresh_wells_use_two_dimensional_nearest_cell(monkeypatch, tmp_path):
    """Fresh Tata screens use both x and y when selecting the nearest cell."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(["fresh-near-cell-1-by-y"], [900.0], [(0.0, 90.0)])
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(monkeypatch)
    ds = _tata_dataset(
        kh_values=[
            [5.0, 11.0],
            [10.0, 9.0],
            [15.0, 13.0],
        ],
        x_values=[0.0, 0.0],
        y_values=[0.0, 100.0],
    )

    _, gdf_tata_zoet = well.get_wells_tata_dataframes(tmp_path, ds)

    assert gdf_tata_zoet.loc[0, "botm"] == 18.0 + 0.001
    assert gdf_tata_zoet.loc[0, "top"] == 20.0 - 0.001


@pytest.mark.parametrize(
    ("chloride", "should_warn"),
    [
        (999.0, False),
        (1000.0, False),
        (1001.0, True),
    ],
)
def test_tata_chloride_warning_is_strict_and_does_not_change_placement(
    monkeypatch,
    tmp_path,
    caplog,
    chloride,
    should_warn,
):
    """Chloride warnings are strict and do not alter selected screen placement."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(["fresh-a"], [900.0], [(101.0, 0.0)])
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(monkeypatch)
    chloride_values = np.full((3, 2), 500.0)
    chloride_values[0, 1] = chloride
    chloride_values[2, 1] = 2000.0
    ds = _tata_dataset(
        kh_values=[
            [5.0, 11.0],
            [10.0, 9.0],
            [15.0, 13.0],
        ],
        chloride_values=chloride_values,
    )

    with caplog.at_level("WARNING", logger=well.logger.name):
        _, gdf_tata_zoet = well.get_wells_tata_dataframes(tmp_path, ds)

    assert gdf_tata_zoet.loc[0, "botm"] == 18.0 + 0.001
    assert gdf_tata_zoet.loc[0, "top"] == 20.0 - 0.001
    warnings = [
        record for record in caplog.records if "Unable to place zoetwaterbron in fresh aquifer" in record.getMessage()
    ]
    assert bool(warnings) is should_warn


def test_tata_fresh_wells_use_custom_kd_threshold(monkeypatch, tmp_path):
    """Fresh Tata screens honor a custom kd threshold."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(["fresh-a"], [900.0], [(1.0, 0.0)])
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(monkeypatch)
    ds = _tata_dataset(
        kh_values=[
            [11.0, 11.0],
            [13.0, 9.0],
            [15.0, 13.0],
        ],
    )

    _, gdf_tata_zoet = well.get_wells_tata_dataframes(tmp_path, ds, kd_zoet_layer=120.0)

    assert gdf_tata_zoet.loc[0, "botm"] == 2.0 + 0.001
    assert gdf_tata_zoet.loc[0, "top"] == 8.0 - 0.001


def test_tata_fresh_wells_use_custom_chloride_threshold(monkeypatch, tmp_path, caplog):
    """Fresh Tata chloride warnings honor a custom chloride threshold."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(["fresh-a"], [900.0], [(101.0, 0.0)])
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(monkeypatch)
    chloride_values = np.full((3, 2), 500.0)
    chloride_values[0, 1] = 750.0
    ds = _tata_dataset(
        kh_values=[
            [5.0, 11.0],
            [10.0, 9.0],
            [15.0, 13.0],
        ],
        chloride_values=chloride_values,
    )

    with caplog.at_level("WARNING", logger=well.logger.name):
        well.get_wells_tata_dataframes(tmp_path, ds, cl_max_zoet_layer=700.0)

    warnings = [
        record for record in caplog.records if "Unable to place zoetwaterbron in fresh aquifer" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_tata_fresh_wells_preserve_no_qualifying_kd_failure(monkeypatch, tmp_path):
    """Fresh Tata wells keep the original failure mode when no layer qualifies."""
    salt_frame = _tata_frame(["salt-a"], [1200.0], [(10.0, 1.0)])
    fresh_frame = _tata_frame(["fresh-a"], [900.0], [(1.0, 0.0)])
    _patch_tata_reads(monkeypatch, salt_frame, fresh_frame)
    _patch_thickness(monkeypatch)
    ds = _tata_dataset(
        kh_values=[
            [5.0, 5.0],
            [10.0, 10.0],
            [9.0, 9.0],
        ],
    )

    with pytest.raises(IndexError):
        well.get_wells_tata_dataframes(tmp_path, ds)
