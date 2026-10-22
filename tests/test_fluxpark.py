"""FluxPark period means, spatial cropping, caching and package integration."""

import flopy
import nlmod
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from nhflotools.fluxpark import get_fluxpark_recharge
from tests.util import make_rect_vertex_ds


def _model_ds(start, perlen, steady):
    """Build a small vertex grid with the PWN model's time convention."""
    ds = make_rect_vertex_ds(delr=1.0, top=10.0, botm=(0.0,))
    return nlmod.time.set_ds_time(ds, start=start, perlen=perlen, steady=steady, steady_start=False)


def _write_fluxpark(tmp_path, dates, drainage, deficit):
    """Write daily rates with spatial variation to a synthetic source raster."""
    shape = (len(dates), 2, 2)
    source = xr.Dataset(
        {
            "drain_mm_d": (
                ("time", "y", "x"),
                np.broadcast_to(np.asarray(drainage, dtype=float)[:, None, None], shape).copy(),
            ),
            "trans_def_pot_mm_d": (
                ("time", "y", "x"),
                np.broadcast_to(np.asarray(deficit)[:, None, None], shape).copy(),
            ),
        },
        coords={"time": dates, "y": [1.5, 0.5], "x": [0.5, 1.5]},
    )
    source["drain_mm_d"] *= xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("y", "x"))
    for da in source.data_vars.values():
        da.attrs = {"units": "mm/d", "crs": "EPSG:28992"}
    path = tmp_path / "rch_fluxpark.nc"
    source.to_netcdf(path)
    return tmp_path


@pytest.mark.parametrize("steady", [True, False])
def test_period_means_before_spatial_resampling(tmp_path, steady, monkeypatch):
    """Average left-closed periods, including steady periods, before resampling."""
    dates = pd.date_range("2022-01-01", periods=6)
    path = _write_fluxpark(tmp_path, dates, [1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12])
    ds = _model_ds("2022-01-02", [2.0, 1.0], steady)
    original = ds.copy(deep=True)
    resample = nlmod.resample.structured_da_to_ds

    def resample_periods(da, grid, **kwargs):
        assert da.sizes["time"] == 1
        assert da.time.values[0] in ds.time.values
        pd.testing.assert_index_equal(grid.indexes["time"], da.indexes["time"])
        return resample(da, grid, **kwargs)

    monkeypatch.setattr(nlmod.resample, "structured_da_to_ds", resample_periods)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start)
    xr.testing.assert_identical(ds, original)
    pd.testing.assert_index_equal(result.indexes["time"], ds.indexes["time"])
    spatial = np.array([1, 2, 3, 4])
    np.testing.assert_allclose(result.recharge, np.array([4, 7])[:, None] * spatial / 1000)
    np.testing.assert_allclose(result.evaporation, np.broadcast_to(np.array([5, 8])[:, None] / 1000, (2, 4)))
    durations = xr.DataArray([2.0, 1.0], dims="time", coords={"time": ds.time})
    assert float((result.recharge * ds.area * durations).sum()) == pytest.approx((3 + 5 + 7) * spatial.sum() / 1000)


@pytest.mark.parametrize("extrapolate", [True, False])
def test_nearest_extrapolation_preserves_observed_period_means(tmp_path, extrapolate):
    """Fill only empty periods, not missing days in partly observed periods."""
    dates = pd.date_range("2022-01-02", periods=6)
    drainage = np.array([1, 3, 5, 7, 9, 11])
    path = _write_fluxpark(tmp_path, dates, drainage, 2 * drainage)
    ds = _model_ds("2022-01-01", [3.0, 5.0, 2.0], False)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start, extrapolate_in_time=extrapolate)
    pd.testing.assert_index_equal(result.indexes["time"], ds.indexes["time"])
    expected = np.array([2, 8, 11 if extrapolate else np.nan])
    np.testing.assert_allclose(result.recharge, expected[:, None] * [1, 2, 3, 4] / 1000)
    np.testing.assert_allclose(result.evaporation, np.broadcast_to(2 * expected[:, None] / 1000, (3, 4)))


@pytest.mark.parametrize(("start", "rate"), [("2021-12-01", 1), ("2022-01-08", 11)])
@pytest.mark.parametrize("extrapolate", [True, False])
def test_model_window_outside_observations(tmp_path, start, rate, extrapolate):
    """A model entirely outside source coverage still uses the nearest endpoint."""
    dates = pd.date_range("2022-01-02", periods=6)
    path = _write_fluxpark(tmp_path, dates, [1, 3, 5, 7, 9, 11], [2, 6, 10, 14, 18, 22])
    ds = _model_ds(start, [2.0], False)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start, extrapolate_in_time=extrapolate)
    rate = rate if extrapolate else np.nan
    np.testing.assert_allclose(result.recharge, rate * np.array([[1, 2, 3, 4]]) / 1000)
    np.testing.assert_allclose(result.evaporation, 2 * rate / 1000)


@pytest.mark.parametrize("extrapolate", [True, False])
def test_internal_gap_uses_nearest_observation_without_filling_spatial_nan(tmp_path, extrapolate):
    """Nearest refers to source times; observed NaNs are not empty time bins."""
    dates = pd.to_datetime(["2022-01-01", "2022-01-05", "2022-01-06"])
    path = _write_fluxpark(tmp_path, dates, [2, np.nan, 10], [4, 8, 20])
    ds = _model_ds("2022-01-01", [1.0, 2.0, 2.0, 1.0], False)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start, extrapolate_in_time=extrapolate)
    # The empty [Jan 2, Jan 4) bin is nearest to Jan 5 at its end label.
    expected = np.array([4, 8 if extrapolate else np.nan, 8, 20])
    np.testing.assert_allclose(result.evaporation, np.broadcast_to(expected[:, None] / 1000, (4, 4)))
    assert result.recharge.isel(time=[1, 2]).isnull().all()


@pytest.mark.parametrize("fillna", [True, False])
def test_spatial_fill_uses_vertex_centres_per_period(tmp_path, fillna):
    """Fill spatial holes independently per variable and period, not by cell index."""
    dates = pd.date_range("2022-01-01", periods=2)
    path = _write_fluxpark(tmp_path, dates, [1, 10], [2.0, 20.0])
    with xr.open_dataset(path / "rch_fluxpark.nc") as source:
        source.load()
    source["drain_mm_d"][:, 0, :] = np.nan
    source["trans_def_pot_mm_d"][:, 1, :] = np.nan
    source.to_netcdf(path / "rch_fluxpark.nc")
    order = [3, 0, 2, 1]
    ds = _model_ds("2022-01-01", [1.0, 1.0], False).isel(icell2d=order)
    ds = ds.assign_coords(icell2d=np.arange(4))
    cachedir = tmp_path / "cache"
    cachedir.mkdir()
    options = {
        "ds": ds,
        "data_path_fluxpark": path,
        "start": ds.time.start,
        "cachedir": str(cachedir),
        "cachename": "fluxpark",
    }
    get_fluxpark_recharge(**options, fillna=not fillna)
    result = get_fluxpark_recharge(**options, fillna=fillna)
    drainage = np.array([3, 4, 3, 4] if fillna else [np.nan, np.nan, 3, 4])
    deficit = np.array([2, 2, 2, 2] if fillna else [2, 2, np.nan, np.nan])
    for variable, spatial in [("recharge", drainage), ("evaporation", deficit)]:
        assert result[variable].dims == ("time", "icell2d")
        pd.testing.assert_index_equal(result.indexes["time"], ds.indexes["time"])
        np.testing.assert_allclose(result[variable], np.array([1, 10])[:, None] * spatial[order] / 1000)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cell_size", [1.0, 0.25])
def test_spatial_crop_precedes_time_averaging(tmp_path, monkeypatch, reverse, cell_size):
    """Clip remote pixels before grouping and preserve even a one-pixel crop."""
    ds = nlmod.time.set_ds_time(
        make_rect_vertex_ds(delr=cell_size),
        start="2022-01-01",
        perlen=[2.0],
        steady=False,
        steady_start=False,
    )
    coords = np.arange(-4.5, 7.0)
    drainage = xr.DataArray(
        np.broadcast_to(np.array([2.0, 6.0])[:, None, None], (2, len(coords), len(coords))),
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2022-01-01", periods=2), "y": coords, "x": coords},
        attrs={"units": "mm/d"},
    )
    source = xr.Dataset({"drain_mm_d": drainage, "trans_def_pot_mm_d": drainage})
    source.sortby(["x", "y"], ascending=not reverse).to_netcdf(tmp_path / "rch_fluxpark.nc")
    chunk = xr.DataArray.chunk
    cropped = []

    def chunk_cropped(data, *args, **kwargs):
        size = 2 if cell_size == 1.0 else 1
        assert data.sizes["x"] == data.sizes["y"] == size
        cropped.append(data.name)
        return chunk(data, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "chunk", chunk_cropped)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=str(tmp_path), start=ds.time.start)
    assert set(cropped) == {"drain_mm_d", "trans_def_pot_mm_d"}
    np.testing.assert_allclose(result.recharge, 0.004)
    np.testing.assert_allclose(result.evaporation, 0.004)


@pytest.mark.parametrize("long_axis", ["x", "y"])
def test_source_reads_and_resampling_are_memory_bounded(tmp_path, monkeypatch, long_axis):
    """Bound actual NetCDF reads in time and space; resample one reduced period."""
    dates = pd.date_range("2022-01-01", periods=24)
    coords = {"x": np.array([0.5, 1.5]), "y": np.array([1.5, 0.5]), "time": dates}
    coords[long_axis] = np.arange(-10.5, 530.0)
    shape = (len(dates), len(coords["y"]), len(coords["x"]))
    drainage = xr.DataArray(
        np.broadcast_to(np.arange(24, dtype=float)[:, None, None], shape),
        dims=("time", "y", "x"),
        coords=coords,
        attrs={"units": "mm/d"},
    )
    source = xr.Dataset({"drain_mm_d": drainage, "trans_def_pot_mm_d": drainage * 2})
    source["trans_def_pot_mm_d"].attrs["units"] = "mm/d"
    source.to_netcdf(tmp_path / "rch_fluxpark.nc")
    ds = nlmod.time.set_ds_time(
        make_rect_vertex_ds(nx=260 if long_axis == "x" else 1, ny=260 if long_axis == "y" else 1, delr=2.0),
        start="2022-01-01",
        perlen=[13.0, 11.0],
        steady=False,
        steady_start=False,
    )
    getitem = NetCDF4ArrayWrapper._getitem  # noqa: SLF001
    compute = xr.DataArray.compute
    resample = nlmod.resample.structured_da_to_ds
    reads = []
    periods = []
    max_chunk = (5, 256, 256)

    def read_chunk(array, key):
        if array.variable_name in {"drain_mm_d", "trans_def_pot_mm_d"}:
            counts = tuple(
                len(range(*index.indices(size))) if isinstance(index, slice) else len(index)
                for index, size in zip(key, shape, strict=True)
            )
            reads.append(counts)
            assert all(count <= bound for count, bound in zip(counts, max_chunk, strict=True))
        return getitem(array, key)

    def compute_synchronously(da, **kwargs):
        assert kwargs["scheduler"] == "synchronous"
        assert da.dims == ("y", "x")
        return compute(da, **kwargs)

    def resample_one_period(da, grid, **kwargs):
        assert da.sizes["time"] == 1
        assert da.chunks is None
        periods.append(da.time.item())
        return resample(da, grid, **kwargs)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read_chunk)
    monkeypatch.setattr(xr.DataArray, "compute", compute_synchronously)
    monkeypatch.setattr(nlmod.resample, "structured_da_to_ds", resample_one_period)
    result = get_fluxpark_recharge(ds, data_path_fluxpark=tmp_path, start=ds.time.start)
    assert reads
    assert len(periods) == len(result.data_vars) * ds.sizes["time"]
    np.testing.assert_allclose(
        result.recharge, np.broadcast_to(np.array([6, 18])[:, None] / 1000, (2, ds.sizes["icell2d"]))
    )
    np.testing.assert_allclose(result.evaporation, result.recharge * 2)


@pytest.mark.parametrize("source_dims", [("time", "y", "x"), ("x", "time", "y")])
def test_area_weighted_fluxes_on_quadtree(tmp_path, source_dims):
    """Unequal source overlaps preserve volumes on coarse and refined model cells."""
    ds = make_rect_vertex_ds(nx=4, ny=4, delr=1.0).isel(icell2d=[0, 1, 4, 5, 2, 8, 10])
    ds = ds.assign_coords(
        icell2d=np.arange(7),
        x=("icell2d", [0.5, 1.5, 0.5, 1.5, 3.0, 1.0, 3.0]),
        y=("icell2d", [3.5, 3.5, 2.5, 2.5, 3.0, 1.0, 1.0]),
    )
    ds["area"] = ("icell2d", [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    ds["icvert"].values[4:] = [[2, 4, 14, 12], [10, 12, 22, 20], [12, 14, 24, 22]]
    ds = nlmod.time.set_ds_time(ds, start="2022-01-01", perlen=[2.0], steady=False, steady_start=False)
    spatial = np.array([[1.0, 4.0, 10.0], [2.0, 8.0, 20.0], [3.0, 12.0, 30.0]])
    drainage = xr.DataArray(
        np.array([1.0, 3.0])[:, None, None] * spatial,
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2022-01-01", periods=2), "y": [3.5, 1.5, -0.5], "x": [-0.5, 1.5, 3.5]},
        attrs={"units": "mm/d"},
    )
    source = xr.Dataset({"drain_mm_d": drainage, "trans_def_pot_mm_d": 2 * drainage})
    source["trans_def_pot_mm_d"].attrs["units"] = "mm/d"
    path = tmp_path / "rch_fluxpark.nc"
    source.transpose(*source_dims).to_netcdf(path)

    result = get_fluxpark_recharge(ds, data_path_fluxpark=tmp_path, start=ds.time.start)

    # For example, the coarse NE cell overlaps four source pixels by 0.75, 2.25,
    # 0.25 and 0.75 m2: (0.75*4 + 2.25*10 + 0.25*8 + 0.75*20) / 4 = 10.625.
    expected = 2 * np.array([[2.5, 4.0, 3.75, 6.0, 10.625, 7.3125, 19.125]]) / 1000
    for variable, factor in [("recharge", 1), ("evaporation", 2)]:
        assert result[variable].dims == ("time", "icell2d")
        np.testing.assert_allclose(result[variable], factor * expected)
        assert float((result[variable] * ds.area * 2).sum()) == pytest.approx(factor * 0.658)


@pytest.mark.parametrize("changed_setting", ["end", "start", "grid"])
def test_cache_round_trip_and_time_invalidation(tmp_path, changed_setting, caplog):
    """Cached rates match a direct call and are invalidated by model-time changes."""
    dates = pd.date_range("2022-01-01", periods=6)
    path = _write_fluxpark(tmp_path, dates, [1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12])
    ds = _model_ds("2022-01-01", [2.0, 1.0, 3.0], False)
    cachedir = tmp_path / "cache"
    cachedir.mkdir()
    reference = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start)
    first = get_fluxpark_recharge(
        ds, data_path_fluxpark=path, start=ds.time.start, cachedir=str(cachedir), cachename="fluxpark"
    )
    with caplog.at_level("INFO", logger="nlmod.cache"):
        second = get_fluxpark_recharge(
            ds, data_path_fluxpark=path, start=ds.time.start, cachedir=str(cachedir), cachename="fluxpark"
        )
    assert "using cached data" in caplog.text
    xr.testing.assert_equal(first, reference)
    xr.testing.assert_equal(second, reference)
    assert second.recharge.attrs["units"] == "m/d"
    assert (cachedir / "fluxpark.nc").exists()
    assert (cachedir / "fluxpark.pklz").exists()

    if changed_setting == "end":
        changed = _model_ds("2022-01-01", [2.0, 1.0, 2.0], False)
    elif changed_setting == "start":
        changed = _model_ds("2022-01-02", [1.0, 1.0, 3.0], False)
    else:
        changed = nlmod.time.set_ds_time(
            make_rect_vertex_ds(nx=1, ny=1, delr=2.0),
            start="2022-01-01",
            perlen=[2.0, 1.0, 3.0],
            steady=False,
            steady_start=False,
        )
    updated = get_fluxpark_recharge(
        changed, data_path_fluxpark=path, start=changed.time.start, cachedir=str(cachedir), cachename="fluxpark"
    )
    xr.testing.assert_equal(updated, get_fluxpark_recharge(changed, data_path_fluxpark=path, start=changed.time.start))
    assert not np.array_equal(updated.recharge, first.recharge)
    if changed_setting == "grid":
        np.testing.assert_allclose(updated.recharge[:, 0], np.array([2, 5, 9]) * 2.5 / 1000)


def test_cache_invalidation_when_extrapolation_changes(tmp_path):
    """An extrapolated cache must not fill empty periods when filling is disabled."""
    dates = pd.date_range("2022-01-02", periods=2)
    path = _write_fluxpark(tmp_path, dates, [2, 4], [4, 8])
    ds = _model_ds("2022-01-01", [1.0, 2.0], False)
    cachedir = tmp_path / "cache"
    cachedir.mkdir()
    options = {"ds": ds, "data_path_fluxpark": path, "start": ds.time.start, "cachedir": str(cachedir)}
    filled = get_fluxpark_recharge(**options, extrapolate_in_time=True)
    unfilled = get_fluxpark_recharge(**options, extrapolate_in_time=False)
    assert filled.recharge.isel(time=0).notnull().all()
    assert unfilled.recharge.isel(time=0).isnull().all()
    xr.testing.assert_equal(filled.isel(time=1), unfilled.isel(time=1))


@pytest.mark.mf6
@pytest.mark.parametrize("perlen", [[1.0, 1.0], [2.0]])
@pytest.mark.parametrize(
    ("heads", "et_factors"),
    [
        ([11.75, 11.0, 11.0, 10.5], [1.0, 0.375, 0.0]),
        ([11.5, 10.0, 11.0, 10.0], [1.0, 0.0, 0.0]),
    ],
)
def test_fluxpark_package_rates_masks_and_evt_response(tmp_path, heads, et_factors, perlen):
    """Period means drive RCH and EVT with the correct labels and land fractions."""
    dates = pd.date_range("2023-01-01", periods=2)
    path = _write_fluxpark(tmp_path, dates, [3, 5], [4, 6])
    ds = _model_ds("2023-01-01", perlen, False)
    ds["nstp"][:] = 2 if len(perlen) == 1 else 1
    ds.attrs["model_ws"] = str(tmp_path)
    ds["ahn"] = xr.full_like(ds.top, 12.0)
    fluxpark = get_fluxpark_recharge(ds, data_path_fluxpark=path, start=ds.time.start)
    meteo_ds = ds.copy()
    meteo_ds.update(fluxpark)
    sw_frac = xr.DataArray([0.0, 0.25, 1.0, 0.0], dims="icell2d", coords={"icell2d": ds.icell2d})
    land_mask = sw_frac < 1.0
    sim = flopy.mf6.MFSimulation(sim_ws=str(tmp_path), exe_name=nlmod.util.get_exe_path())
    nlmod.sim.tdis(ds, sim)
    nlmod.sim.ims(sim)
    gwf = nlmod.gwf.gwf(ds, sim, save_flows=True)
    nlmod.gwf.disv(ds, gwf)
    nlmod.gwf.npf(ds, gwf)
    nlmod.gwf.ic(ds, gwf, starting_head=11.0)
    nlmod.gwf.oc(ds, gwf, save_budget="all")
    nlmod.gwf.rch(meteo_ds, gwf, recharge=meteo_ds["recharge"] * (1.0 - sw_frac), mask=land_mask)
    evt = nlmod.gwf.evt(
        meteo_ds,
        gwf,
        rate=meteo_ds["evaporation"] * (1.0 - sw_frac),
        surface=ds["ahn"] - 0.5,
        depth=1.0,
        mask=land_mask,
    )
    np.testing.assert_allclose(evt.stress_period_data.get_data(0)["surface"], 11.5)
    np.testing.assert_allclose(evt.stress_period_data.get_data(0)["depth"], 1.0)
    assert [record["cellid"] for record in evt.stress_period_data.get_data(0)] == [(0, 0), (0, 1), (0, 3)]
    # Strong GHBs impose heads without disabling RCH/EVT in constant-head cells.
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: [((0, i), head, 1e9) for i, head in enumerate(heads)]})
    sim.write_simulation(silent=True)
    success, output = sim.run_simulation(silent=True, report=True)
    assert success, "\n".join(output)
    with gwf.output.budget() as budget:
        expected = (
            [(1.0, 0.004, 0.005), (2.0, 0.004, 0.005)]
            if len(perlen) == 1
            else [(1.0, 0.003, 0.004), (2.0, 0.005, 0.006)]
        )
        for time, rch_rate, evt_rate in expected:
            np.testing.assert_allclose(
                budget.get_data(text="EVT", totim=time)[0]["q"], -evt_rate * np.array(et_factors), atol=1e-9
            )
            np.testing.assert_allclose(
                budget.get_data(text="RCH", totim=time)[0]["q"], rch_rate * np.array([1.0, 1.5, 4.0]), atol=1e-12
            )
