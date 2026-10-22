"""Tests for NHFLO well helpers."""

from pathlib import Path

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
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


# --- get_wells_pwn_dataframe ------------------------------------------------
#
# One synthetic secundair bookkeeping, shared by every PWN test below.
#
# Wells (``sec_nput`` = number of wells the secundair flow is spread over):
#   W1/W2/W3  tag T1     sec_nput 3     -> extraction, split three ways
#   W4        tag T2     sec_nput 1     -> extraction, asymmetric series
#   W5        tag TX     sec_nput 2     -> tag absent from the feather  (drop)
#   W6        tag T1     sec_nput 0     -> division by zero -> +/-inf    (drop)
#   W7        tag T0     sec_nput 2     -> secundair median is zero      (drop)
#   W8        tag TINF   sec_nput 2     -> infiltration, positive median (keep)
#
# Series medians, by hand (median, not mean -- T2 and TINF are asymmetric so a
# mean would give a different answer):
#   T1   [-30, -30, -30] -> -30      (mean -30, not a discriminator)
#   T2   [-10, -20, -60] -> -20      (mean -30)
#   T0   [ -4,   0,   4] ->   0      (drops on the ``Q != 0`` mask)
#   TINF [  4,   8,  20] ->   8      (mean 32/3)
# The 'ophaal tijdstip' column is datetime: it must be excluded by
# ``numeric_only=True`` and must not become a mappable secundair tag.
_PWN_WELLS = (
    # locatie, sec_flow_tag, sec_nput, x, y
    ("W1", "T1", 3, 0.0, 1.0),
    ("W2", "T1", "3", 100.0, 2.0),  # string on purpose; GeoJSON stringifies the column anyway
    ("W3", "T1", 3, 200.0, 3.0),
    ("W4", "T2", 1, 300.0, 4.0),
    ("W5", "TX", 2, 400.0, 5.0),
    ("W6", "T1", 0, 500.0, 6.0),
    ("W7", "T0", 2, 600.0, 7.0),
    ("W8", "TINF", 2, 700.0, 8.0),
)
_PWN_FLOWS = {
    "T1": [-30.0, -30.0, -30.0],
    "T2": [-10.0, -20.0, -60.0],
    "T0": [-4.0, 0.0, 4.0],
    "TINF": [4.0, 8.0, 20.0],
}
_PWN_MEDIANS = {"T1": -30.0, "T2": -20.0, "T0": 0.0, "TINF": 8.0}


@pytest.fixture
def pwn_data_path(tmp_path):
    """Write a real ``pumping_infiltration_wells.geojson`` and ``sec_flows.feather``.

    Real files rather than patched readers: the GeoJSON round-trip is what turns
    ``sec_nput`` into strings, which is exactly the input ``well.py`` has to coerce.

    Returns
    -------
    pathlib.Path
        Directory holding both input files.
    """
    wells = gpd.GeoDataFrame(
        {
            "locatie": [row[0] for row in _PWN_WELLS],
            "sec_flow_tag": [row[1] for row in _PWN_WELLS],
            "sec_nput": [row[2] for row in _PWN_WELLS],
        },
        geometry=[Point(row[3], row[4]) for row in _PWN_WELLS],
        crs="EPSG:28992",
    )
    wells.to_file(tmp_path / "pumping_infiltration_wells.geojson", driver="GeoJSON")

    flows = pd.DataFrame({
        **_PWN_FLOWS,
        "ophaal tijdstip": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    flows.to_feather(tmp_path / "sec_flows.feather")
    return tmp_path


def test_pwn_q_splits_secundair_median_and_conserves_the_total(pwn_data_path):
    """Per-well Q is 24 * median / sec_nput, so a secundair sums back to 24 * median."""
    wdf = well.get_wells_pwn_dataframe(pwn_data_path)

    # Surviving wells, in input order; W5/W6/W7 have no usable flow.
    assert wdf.index.to_list() == ["W1", "W2", "W3", "W4", "W8"]
    assert wdf.index.name == "locatie"

    # m3/h -> m3/day is a factor 24; the secundair flow is split over sec_nput wells.
    assert wdf.loc[["W1", "W2", "W3"], "Q"].to_list() == [-240.0] * 3  # 24 * -30 / 3
    assert wdf.loc["W4", "Q"] == 24.0 * _PWN_MEDIANS["T2"] / 1  # -480.0, median not mean
    assert wdf.loc["W8", "Q"] == 24.0 * _PWN_MEDIANS["TINF"] / 2  # +96.0

    # The WEL mass-balance contract: the three T1 wells reconstruct the whole secundair.
    assert wdf.loc[["W1", "W2", "W3"], "Q"].sum() == 24.0 * _PWN_MEDIANS["T1"]

    # "3" arrived as a string from the GeoJSON and must have been coerced to a number.
    assert pd.api.types.is_numeric_dtype(wdf["sec_nput"])
    assert wdf["sec_nput"].to_list() == [3, 3, 3, 1, 2]

    # Geometry-derived and constant MAW/transport columns.
    assert wdf["x"].to_list() == [0.0, 100.0, 200.0, 300.0, 700.0]
    assert wdf["y"].to_list() == [1.0, 2.0, 3.0, 4.0, 8.0]
    assert wdf["rw"].to_list() == [0.25] * 5
    assert wdf["CONCENTRATION"].to_list() == [0.0] * 5


def test_pwn_drops_unusable_wells_warns_once_and_keeps_infiltration(pwn_data_path, caplog):
    """Unmapped, zero-nput and zero-median wells drop; a positive infiltration survives."""
    with caplog.at_level("WARNING", logger=well.logger.name):
        wdf = well.get_wells_pwn_dataframe(pwn_data_path)

    assert set(wdf.index) == {"W1", "W2", "W3", "W4", "W8"}

    # One summary warning carrying (n_dropped, n_total): 3 of the 8 input wells.
    warnings = [record for record in caplog.records if "without a nonzero secundair flow" in record.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].args == (3, len(_PWN_WELLS))

    # The drop mask is sign-symmetric: it removes non-finite and zero, never negatives,
    # and never the sole infiltration well.
    assert np.isfinite(wdf["Q"]).all()
    assert (wdf["Q"] != 0.0).all()
    assert wdf.loc["W8", "Q"] > 0.0
    assert (wdf.loc[["W1", "W2", "W3", "W4"], "Q"] < 0.0).all()


def test_pwn_unsupported_flow_product_raises(pwn_data_path):
    """Unknown products fail loudly."""
    with pytest.raises(ValueError, match="flow_product bogus not implemented"):
        well.get_wells_pwn_dataframe(pwn_data_path, flow_product="bogus")


@pytest.mark.parametrize("grouped", [False, True])
def test_pwn_timeseries_preserves_idle_wells_and_flow_totals(pwn_data_path, grouped):
    """Time-series names resolve to signed daily flows, including zero-median wells."""
    wdf, flows = well.get_wells_pwn_dataframe(pwn_data_path, flow_product="timeseries", grouped=grouped)

    assert wdf.index.to_list() == ["W1", "W2", "W3", "W4", "W7", "W8"]
    assert all(isinstance(q, str) for q in wdf["Q"])
    assert flows.index.equals(pd.date_range("2024-01-01", periods=3))
    factor = 1 if grouped else 3
    np.testing.assert_allclose(flows[wdf.loc["W1", "Q"]] * factor, [-720.0] * 3)
    np.testing.assert_allclose(flows[wdf.loc["W4", "Q"]], [-240.0, -480.0, -1440.0])
    np.testing.assert_allclose(flows[wdf.loc["W7", "Q"]], np.array([-96.0, 0.0, 96.0]) / (1 if grouped else 2))
    np.testing.assert_allclose(flows[wdf.loc["W8", "Q"]], np.array([96.0, 192.0, 480.0]) / (1 if grouped else 2))
    assert wdf.loc[["W1", "W2", "W3"], "Q"].nunique() == 1


def test_well_period_means_and_nearest_fill():
    """Average half-open periods; empty periods use each column's nearest valid source."""
    flows = pd.DataFrame(
        {"pump": [-10.0, -30.0, -90.0], "infiltration": [2.0, np.nan, 8.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"]),
    )
    ds = well.nlmod.time.set_ds_time(xr.Dataset(), start="2023-12-30", perlen=[1, 3, 1, 1, 3, 4], steady=False)
    result = well.resample_well_flows(flows, ds)
    np.testing.assert_allclose(result["pump"], [-10, -10, -30, -30, -90, -90])
    np.testing.assert_allclose(result["infiltration"], [2, 2, 2, 8, 8, 8])
    assert result.index.equals(pd.DatetimeIndex(ds.time.values))

    # A second discretization puts two observations in one period, with unequal lengths.
    ds = well.nlmod.time.set_ds_time(xr.Dataset(), start="2024-01-01", perlen=[3, 5], steady=False)
    result = well.resample_well_flows(flows, ds)
    np.testing.assert_allclose(result["pump"], [-20, -90])
    assert np.dot(result["pump"], [3, 5]) == pytest.approx(-510.0)


@pytest.mark.parametrize("grouped", [False, True])
def test_pwn_timeseries_write_real_well_packages(pwn_data_path, tmp_path, grouped):
    """Both builders write resolvable TS6 references with stepwise period-start values."""
    wdf, flows = well.get_wells_pwn_dataframe(pwn_data_path, flow_product="timeseries", grouped=grouped)
    wdf = wdf.loc[["W1", "W2", "W3", "W4"]].copy()
    wdf["top"], wdf["botm"] = 9.0, 1.0
    ds = well.nlmod.time.set_ds_time(xr.Dataset(), start="2024-01-01", perlen=[2, 1], steady=False)
    sim = flopy.mf6.MFSimulation(sim_ws=tmp_path / "model")
    flopy.mf6.ModflowTdis(sim, nper=2, perioddata=[(2.0, 2, 1.0), (1.0, 1, 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")
    flopy.mf6.ModflowGwfdis(gwf, nlay=2, nrow=1, ncol=3, delr=100.0, delc=100.0, top=10.0, botm=[5.0, 0.0])
    flopy.mf6.ModflowGwfnpf(gwf, k=[1.0, 3.0])
    wdf["geometry"] = [Point(50, 50), Point(150, 50), Point(250, 50), Point(50, 50)]
    if grouped:
        package = well.nlmod.gwf.wells.maw_from_df(wdf, gwf, group="sec_flow_tag", silent=True)
        records = package.perioddata.get_data(0)
        assert set(records["mawsetting"]) == {"RATE"}
        assert {row[2] for row in records} == set(wdf["Q"])
    else:
        package = well.nlmod.gwf.wells.wel_from_df(wdf, gwf)
        records = package.stress_period_data.get_data(0)
        assert set(records["q"]) == set(wdf["Q"])
        assert records["multiplier"].sum() == pytest.approx(4.0)
    well.nlmod.time.dataframe_to_flopy_timeseries(
        well.resample_well_flows(flows, ds),
        ds=ds,
        package=package,
        shift_label_from_right_to_left=True,
    )
    sim.write_simulation(silent=True)
    ts = package.ts[0].timeseries.get_data()
    np.testing.assert_allclose(ts["ts_time"], [0.0, 2.0, 3.0])
    names = list(package.ts[0].time_series_namerecord.get_data()[0])
    for location, expected in [
        ("W1", [-720.0 if grouped else -240.0] * 3),
        ("W4", [-360.0, -1440.0, -1440.0]),
    ]:
        column = names.index(wdf.loc[location, "Q"]) + 1
        np.testing.assert_allclose([row[column] for row in ts], expected)
    assert "STEPWISE" in (tmp_path / "model" / package.ts[0].filename).read_text().upper()


def test_pwn_grouped_medians_are_secundair_totals(pwn_data_path):
    """Grouped MAW rates are not divided over the number of physical wells."""
    wdf = well.get_wells_pwn_dataframe(pwn_data_path, grouped=True)
    np.testing.assert_allclose(wdf["Q"], [-720, -720, -720, -480, 192])


@pytest.mark.parametrize("values", [[np.nan, np.nan], [-10.0, np.inf]])
def test_well_resampling_rejects_unusable_series(values):
    """Absent or infinite observations cannot silently become a model forcing."""
    ds = well.nlmod.time.set_ds_time(xr.Dataset(), start="2024-01-01", perlen=[2], steady=False)
    flows = pd.DataFrame({"pump": values}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="must contain finite observations"):
        well.resample_well_flows(flows, ds)
