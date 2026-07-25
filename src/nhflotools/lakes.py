"""Helpers for the Bergen pond/lake (``lakes_pwn``) features.

The PWN model carves the model top down to the lake bottom in every grid cell that the
managed ponds/lakes together sufficiently cover, and then holds each carved cell at the
prescribed stage of its lake(s) with a per-lake RIV boundary: every lake keeps its own
reach, stage, conductance and boundname, also where two lakes share a cell, so per-lake
budgets stay attributable and a future MVR/weir or LAK configuration can address lakes
individually. Both the carved cell set and the RIV reach set are derived from the single
aggregator :func:`_aggregate_lake_cells`, so the two sets are equal by construction: no
carved cell is ever left without a stage, and no stage reach ever lands on an un-carved
cell. For detailed studies :func:`lak_gdf_from_lakes_pwn` prepares a LAK input frame
from the same aggregator instead, which the model script feeds directly to
:func:`nlmod.gwf.lake_from_gdf` to solve the stage from the lake water balance with
the same lakebed leakance as the RIV.

The logic lives here (rather than inline in the model script) so it can be imported and
unit-tested without running the full REGIS/AHN/MF6 build.
"""

import logging

import flopy
import nlmod
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=0.5):
    """Aggregate lake pieces to one carved/stage record per lake per grid cell.

    This is the single source of truth for the carved-cell set. It (1) keeps only lake
    pieces that carry both a stage (``strt``) and a bottom (``botm``), (2) computes each
    cell's combined lake coverage over exactly those pieces, (3) keeps cells whose combined
    coverage is strictly greater than ``min_area_fraction``, and (4) collapses the surviving
    pieces to one record per lake (``identificatie``) per cell — lakes are never merged
    across ``identificatie``, so each keeps its own stage and boundname.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset. Only ``ds['area']`` (cell area indexed by ``icell2d``) is used.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons already intersected with the model grid (see
        :func:`nlmod.dims.gdf_to_grid`). Must carry a ``cellid`` column and the
        ``strt``, ``botm``, ``clake`` and ``identificatie`` columns of ``lakes_pwn``.
    min_area_fraction : float, optional
        A cell is kept when the lakes combined cover strictly more than this fraction of
        its area. The default is 0.5, so a cell exactly half covered is not carved.

    Returns
    -------
    agg : pandas.DataFrame
        One row per (cell, lake), indexed by ``cellid`` (repeated when lakes share a
        cell), with columns ``strt`` (area-weighted within the lake), ``botm`` (minimum
        within the lake), ``cond`` (summed ``piece_area / clake``, units m2/d) and
        ``identificatie``. When no cell clears the coverage threshold (empty input, every
        piece missing ``strt``/``botm``, or all cells below ``min_area_fraction``), an
        empty ``cellid``-indexed frame with those columns is returned so the callers carve
        nothing and emit no RIV rather than raising on an empty geometry column.
    coverage : pandas.Series
        Combined stage-carrying lake coverage fraction per ``cellid``, *before* the
        ``min_area_fraction`` threshold, for every cell any surviving piece touches. Used
        for the fractional recharge mask.
    """
    lake = gdf_lake_grid.dropna(subset=["strt", "botm"]).copy()
    n_drop = len(gdf_lake_grid) - len(lake)
    if n_drop:
        logger.warning("Dropping %d lake piece(s) missing strt or botm", n_drop)

    empty_agg = pd.DataFrame(
        {"strt": [], "botm": [], "cond": [], "identificatie": []},
        index=pd.Index([], name="cellid", dtype="int64"),
    )
    empty_coverage = pd.Series([], index=pd.Index([], name="cellid", dtype="int64"), dtype="float64")
    if lake.empty:
        return empty_agg, empty_coverage

    # Use the true clipped-piece area throughout (coverage, conductance and the
    # area-weighted stage), so the three are mutually consistent regardless of any stale
    # 'area' column carried over from gdf_to_grid.
    lake["area"] = lake.geometry.area
    overlap = lake["area"].groupby(lake["cellid"]).sum()
    cell_area = ds["area"].sel(icell2d=overlap.index.to_numpy()).to_numpy()
    coverage = overlap / cell_area
    # Strict '>' so a cell exactly at min_area_fraction is excluded.
    lake = lake[lake["cellid"].map(coverage) > min_area_fraction]

    if lake.empty:
        return empty_agg, coverage

    lake["cond"] = lake["area"] / lake["clake"]
    parts = []
    for ident, group in lake.groupby("identificatie", sort=False, dropna=False):
        agg = nlmod.grid.aggregate_vector_per_cell(
            group,
            fields_methods={"strt": "area_weighted", "botm": "min", "cond": "sum"},
        )
        agg["identificatie"] = ident
        parts.append(agg)
    agg = pd.concat(parts).sort_index()
    agg.index.name = "cellid"
    return agg, coverage


def carve_lake_cells(ds, gdf_lake_grid, min_area_fraction=0.5):
    """Lower the model top to the lake bottom in sufficiently covered cells.

    The model top is set to the deepest aggregated lake bottom in every carved cell and two
    markers are added for the recharge mask, the polder-drain exclusion and the lake RIV to
    reuse: a boolean ``ds['lake_cell']`` (True exactly at the carved cells) and a float
    ``ds['lake_coverage']`` (combined stage-carrying lake coverage fraction per cell,
    including cells below the carve threshold).

    This is called from the model script directly after the layer model is finalized (the
    nlmod top from AHN/REGIS) and before ``starting_head`` and any package build. It only
    lowers the top: :func:`nlmod.layers.set_model_top` is a one-way ratchet, so when more
    carve sources join (bathymetry, panden opt-in) their bed elevations must be merged into
    one top with a single ``set_model_top`` call rather than carved incrementally — see the
    consolidated adjust-top note on NHFLO/models#126.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset with ``top``, ``botm`` and ``area``.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons intersected with the model grid (see :func:`_aggregate_lake_cells`).
    min_area_fraction : float, optional
        Minimum combined lake coverage for a cell to be carved, by default 0.5 (strict
        ``>``).

    Returns
    -------
    ds : xarray.Dataset
        Copy of the input dataset with the lowered top, ``thickness`` dropped (so it is
        recomputed from the new top/botm), and the ``lake_cell`` / ``lake_coverage``
        markers.
    lake_cellids : numpy.ndarray
        The ``icell2d`` values of the carved cells.

    Notes
    -----
    ``ds['ahn']`` is intentionally not refreshed to the carved top. The only post-carve
    reader of ``ds['ahn']`` is the polder DRN, which now excludes these cells, so a refresh
    would be inert; refreshing ``ahn`` to the lake bottom would also corrupt its meaning as
    the measured maaiveld.
    """
    agg, coverage = _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=min_area_fraction)
    # Deepest lake bed per cell: lakes sharing a cell carve it to the lower of their beds.
    cell_botm = agg["botm"].groupby(level="cellid").min()
    lake_cellids = cell_botm.index.to_numpy()

    top = ds["top"].copy()
    top.loc[{"icell2d": lake_cellids}] = cell_botm.to_numpy()
    ds = nlmod.layers.set_model_top(ds, top)
    ds = ds.drop_vars("thickness", errors="ignore")

    ds["lake_cell"] = xr.zeros_like(ds["top"], dtype=bool)
    ds["lake_cell"].loc[{"icell2d": lake_cellids}] = True
    # Coverage can nominally exceed 1 when lake polygons overlap each other; clip so the
    # fractional recharge mask never removes more than a cell's full meteoric term.
    ds["lake_coverage"] = xr.zeros_like(ds["top"], dtype=float)
    ds["lake_coverage"].loc[{"icell2d": coverage.index.to_numpy()}] = coverage.clip(upper=1.0).to_numpy()
    return ds, lake_cellids


def riv_from_lakes_pwn(ds, gwf, gdf_lake_grid, min_area_fraction=0.5):
    """Hold the carved lake cells at their prescribed stage with a per-lake RIV.

    A RIV reach is placed in every carved cell for every lake that covers it, with
    ``stage = strt``, ``rbot = botm`` (that lake's bed) and ``cond = sum(piece_area /
    clake)``, and the lake's ``identificatie`` as boundname. Because ``rbot`` equals the
    lake bed, the reach caps bed infiltration once the head drops below the lakebed
    (perched-pond behaviour) while draining freely when the head rises above the stage. In
    a cell carved by a single lake (the usual case) ``rbot`` therefore equals the carved
    top; where lakes share a cell, the cell is carved to the deepest bed and the shallower
    lake's reach keeps its own, higher ``rbot``. Keeping one reach per lake preserves
    per-lake budgets and boundnames, so MVR movers and weir outlets between individual
    lakes (and a future LAK swap) remain configurable.

    The reach set is derived from the same :func:`_aggregate_lake_cells` call as
    :func:`carve_lake_cells`, so its cell set equals the carved-cell set by construction.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset (post-carve), with ``top``, ``botm``, ``kh`` and idomain.
    gwf : flopy.mf6.ModflowGwf
        Groundwater flow model the RIV package is added to.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons intersected with the model grid (see :func:`_aggregate_lake_cells`).
    min_area_fraction : float, optional
        Minimum combined lake coverage for a cell to be carved/bounded, by default 0.5.

    Returns
    -------
    flopy.mf6.ModflowGwfriv or None
        The lake RIV package, or ``None`` when no cell clears the coverage threshold (so the
        caller adds no package). When ``ds.transport`` is set, the package name is appended
        to ``ds.attrs['ssm_sources']`` (if absent) so its CONCENTRATION aux is used by SSM.
    """
    agg, _ = _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=min_area_fraction)
    agg = agg.rename(columns={"strt": "stage", "botm": "rbot", "identificatie": "boundname"})
    if agg.empty:
        return None
    agg["aux"] = 0.0

    riv_spd = nlmod.gwf.build_spd(agg, "RIV", ds, layer_method="lay_of_rbot")

    riv = flopy.mf6.ModflowGwfriv(
        gwf,
        auxiliary="CONCENTRATION",
        boundnames=True,
        stress_period_data={0: riv_spd},
        save_flows=True,
        pname="riv_lake",
    )
    if ds.transport:
        ssm_sources = list(ds.attrs.get("ssm_sources", []))
        if riv.package_name not in ssm_sources:
            ds.attrs["ssm_sources"] = [*ssm_sources, riv.package_name]
    return riv


def lak_gdf_from_lakes_pwn(ds, gdf_lake_grid, min_area_fraction=0.5):
    """Prepare the per-cell lake frame that :func:`nlmod.gwf.lake_from_gdf` consumes.

    Alternative input to :func:`riv_from_lakes_pwn` for detailed studies: the caller
    builds the LAK package directly with :func:`nlmod.gwf.lake_from_gdf` (and typically
    :func:`nlmod.gwf.copy_meteorological_data_from_ds` for per-lake rainfall and
    evaporation), so the lake stage follows from the simulated water balance instead of
    being prescribed. The frame is derived from the same :func:`_aggregate_lake_cells`
    call as :func:`carve_lake_cells`, so the LAK connection set equals the carved-cell
    set by construction, and the lakebed leakance matches the RIV variant: each
    (cell, lake) row yields one VERTICAL connection with an effective bed resistance
    ``clake = cell_area / sum(piece_area / clake)``, so ``bedleak * cell_area`` equals
    the summed piece conductance of the corresponding RIV reach. The total exchange is
    nevertheless somewhat weaker than the RIV's, because MF6 places the connected
    cell's half-cell vertical resistance (``0.5 * thickness / k33``) in series with the
    lakebed for VERTICAL connections, which the RIV formulation applies directly to the
    cell node (12-16% lower on the PWN layer model). Like the RIV (and unlike a GHB),
    the exchange is capped at the lakebed once the aquifer head drops below it.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset (post-carve). Only ``ds['area']`` is used.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons intersected with the model grid (see
        :func:`_aggregate_lake_cells`). Per-lake outlet columns (``lakeout``,
        ``couttype``, ``outlet_invert``, ``outlet_width``, ``outlet_rough``,
        ``outlet_slope``) are carried through when present.
    min_area_fraction : float, optional
        Minimum combined lake coverage for a cell to be connected, by default 0.5
        (strict ``>``), identical to the carve threshold.

    Returns
    -------
    pandas.DataFrame or None
        One row per (cell, lake), indexed by ``icell2d`` as ``lake_from_gdf`` expects,
        with ``strt`` (one exact value per lake — the per-cell area-weighted values can
        differ at floating-point level and LAK requires a single strt), the effective
        ``clake``, ``identificatie`` and any outlet columns. ``None`` when no cell
        clears the coverage threshold, so the caller adds no package.

    Notes
    -----
    Exclude the lake cells from the RCH package (see :func:`recharge_pond_mask`) so the
    meteoric term is not counted both on the lake and on the aquifer. When two lakes
    share a carved cell, MF6 applies each lake's RAINFALL over that lake's full
    connection (cell) area while RCH excluded the cell only once, so the meteoric term
    on such a cell is over-applied; no current ``lakes_pwn`` cell is shared.
    """
    agg, _ = _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=min_area_fraction)
    if agg.empty:
        return None

    gdf = agg.drop(columns=["botm"])
    gdf["clake"] = ds["area"].sel(icell2d=gdf.index.to_numpy()).to_numpy() / gdf.pop("cond")
    gdf["strt"] = gdf.groupby("identificatie")["strt"].transform("mean")

    outlet_columns = [
        column
        for column in ("lakeout", "couttype", "outlet_invert", "outlet_width", "outlet_rough", "outlet_slope")
        if column in gdf_lake_grid.columns
    ]
    if outlet_columns:
        per_lake = gdf_lake_grid.drop_duplicates("identificatie").set_index("identificatie")[outlet_columns]
        gdf = gdf.join(per_lake, on="identificatie")
    return gdf


def recharge_pond_mask(ds, panden_riv=None, *, fractional=False):
    """Mark cells whose meteoric input is carried by a stage boundary, not by RCH.

    Areal recharge is applied to the top active cell of every non-sea cell, which
    double-counts precipitation on the managed panden (already implicit in the prescribed
    RIV stage) and applies land P-E to the carved open-water lake cells. This mask flags
    those cells so the RCH package can exclude them. Note that ``ds['recharge']`` (KNMI,
    ``method='linear'``) already nets Makkink evaporation against precipitation, and no EVT
    package is built, so excluding a cell removes its entire meteoric term exactly once.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset carrying the ``ds['lake_cell']`` and ``ds['lake_coverage']`` markers
        (see :func:`carve_lake_cells`).
    panden_riv : flopy.mf6.ModflowGwfriv or None, optional
        The infiltration-panden RIV package. When None (the default, e.g. the Bergen extent
        where the panden lie outside the grid) the mask is derived from the lake markers
        alone.
    fractional : bool, optional
        When False (default) return the boolean mask: True at carved lake cells and panden
        RIV cells. When True return a float fraction per cell instead: 1.0 at carved lake
        cells (the whole cell is modeled as lake bed held by the RIV), the panden-covered
        fraction from ``ds['panden_coverage']`` (set by
        :func:`nhflotools.panden.riv_from_oppervlakte_pwn`; 1.0 at panden RIV cells when
        that variable is absent), and the open-water coverage fraction at cells below the
        carve threshold — precipitation on such a partial lake or pand sliver feeds the
        stage boundary, not the aquifer, so the caller can scale recharge by
        ``1 - fraction`` there instead of keeping the full land P-E.

    Returns
    -------
    xarray.DataArray
        Over ``icell2d``: boolean exclusion mask (default), or the fraction of each cell's
        meteoric term carried by a stage boundary (``fractional=True``).
    """
    mask = ds["lake_cell"].copy()
    if panden_riv is not None:
        cells = np.unique([cid[-1] for cid in panden_riv.stress_period_data.data[0]["cellid"]])
        mask.loc[{"icell2d": cells}] = True
    if not fractional:
        return mask
    if "panden_coverage" in ds:
        coverage = ds["lake_coverage"] + ds["panden_coverage"]
    elif panden_riv is not None:
        coverage = xr.where(mask & ~ds["lake_cell"], 1.0, ds["lake_coverage"])
    else:
        coverage = ds["lake_coverage"]
    return xr.where(ds["lake_cell"], 1.0, coverage.clip(max=1.0), keep_attrs=True)
