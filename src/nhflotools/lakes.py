"""Helpers for the Bergen pond/lake (``lakes_pwn``) features.

The PWN model carves the model top down to the lake bottom in every grid cell that a
managed pond/lake sufficiently covers, and then holds those carved cells at their
prescribed lake stage with a per-lake RIV boundary. Both the carved cell set and the RIV
reach set are derived from the single aggregator :func:`_aggregate_lake_cells`, so the two
sets are equal by construction: no carved cell is ever left without a stage, and no stage
reach ever lands on an un-carved cell.

The logic lives here (rather than inline in the model script) so it can be imported and
unit-tested without running the full REGIS/AHN/MF6 build.
"""

import logging

import flopy
import nlmod
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=0.5):
    """Aggregate lake pieces to one carved/stage record per grid cell.

    This is the single source of truth for the carved-cell set. It (1) keeps only lake
    pieces that carry both a stage (``strt``) and a bottom (``botm``), (2) computes each
    cell's lake coverage over exactly those pieces, (3) keeps cells whose coverage is
    strictly greater than ``min_area_fraction``, and (4) collapses the surviving pieces to
    one record per cell.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset. Only ``ds['area']`` (cell area indexed by ``icell2d``) is used.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons already intersected with the model grid (see
        :func:`nlmod.dims.gdf_to_grid`). Must carry a ``cellid`` column and the
        ``strt``, ``botm``, ``clake`` and ``identificatie`` columns of ``lakes_pwn``.
    min_area_fraction : float, optional
        A cell is kept when the lake covers strictly more than this fraction of its area.
        The default is 0.5, so a cell exactly half covered is not carved.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``cellid`` with columns ``strt`` (area-weighted), ``botm`` (minimum),
        ``cond`` (summed ``piece_area / clake``, units m2/d) and ``identificatie``
        (first). Empty-input semantics follow :func:`nlmod.grid.aggregate_vector_per_cell`.
    """
    lake = gdf_lake_grid.dropna(subset=["strt", "botm"]).copy()
    n_drop = len(gdf_lake_grid) - len(lake)
    if n_drop:
        logger.warning("Dropping %d lake piece(s) missing strt or botm", n_drop)

    # Use the true clipped-piece area throughout (coverage, conductance and the
    # area-weighted stage), so the three are mutually consistent regardless of any stale
    # 'area' column carried over from gdf_to_grid.
    lake["area"] = lake.geometry.area
    overlap = lake["area"].groupby(lake["cellid"]).transform("sum").to_numpy()
    cell_area = ds["area"].sel(icell2d=lake["cellid"].to_numpy()).to_numpy()
    # Strict '>' so a cell exactly at min_area_fraction is excluded.
    lake = lake[overlap / cell_area > min_area_fraction]

    lake["cond"] = lake["area"] / lake["clake"]
    return nlmod.grid.aggregate_vector_per_cell(
        lake,
        fields_methods={
            "strt": "area_weighted",
            "botm": "min",
            "cond": "sum",
            "identificatie": "first",
        },
    )


def carve_lake_cells(ds, gdf_lake_grid, min_area_fraction=0.5):
    """Lower the model top to the lake bottom in sufficiently covered cells.

    The model top is set to the aggregated lake bottom in every carved cell and a boolean
    ``ds['lake_cell']`` marker (True exactly at the carved cells) is added for the recharge
    mask, the polder-drain exclusion and the lake RIV to reuse.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset with ``top``, ``botm`` and ``area``.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons intersected with the model grid (see :func:`_aggregate_lake_cells`).
    min_area_fraction : float, optional
        Minimum lake coverage for a cell to be carved, by default 0.5 (strict ``>``).

    Returns
    -------
    ds : xarray.Dataset
        Copy of the input dataset with the lowered top, ``thickness`` dropped (so it is
        recomputed from the new top/botm), and the boolean ``lake_cell`` marker.
    lake_cellids : numpy.ndarray
        The ``icell2d`` values of the carved cells.

    Notes
    -----
    ``ds['ahn']`` is intentionally not refreshed to the carved top. The only post-carve
    reader of ``ds['ahn']`` is the polder DRN, which now excludes these cells, so a refresh
    would be inert; refreshing ``ahn`` to the lake bottom would also corrupt its meaning as
    the measured maaiveld.
    """
    agg = _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=min_area_fraction)
    lake_cellids = agg.index.to_numpy()

    top = ds["top"].copy()
    top.loc[{"icell2d": lake_cellids}] = agg["botm"].to_numpy()
    ds = nlmod.layers.set_model_top(ds, top)
    ds = ds.drop_vars("thickness", errors="ignore")

    ds["lake_cell"] = xr.zeros_like(ds["top"], dtype=bool)
    ds["lake_cell"].loc[{"icell2d": lake_cellids}] = True
    return ds, lake_cellids


def riv_from_lakes_pwn(ds, gwf, gdf_lake_grid, min_area_fraction=0.5):
    """Hold the carved lake cells at their prescribed stage with a per-lake RIV.

    A RIV reach is placed in every carved cell with ``stage = strt``, ``rbot = botm`` (the
    carved top) and ``cond = sum(piece_area / clake)``. Because ``rbot`` equals the carved
    top, the reach caps bed infiltration once the head drops below the lakebed
    (perched-pond behaviour) while draining freely when the head rises above the stage.

    The reach set is derived from the same :func:`_aggregate_lake_cells` call as
    :func:`carve_lake_cells`, so it equals the carved-cell set by construction.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset (post-carve), with ``top``, ``botm``, ``kh`` and idomain.
    gwf : flopy.mf6.ModflowGwf
        Groundwater flow model the RIV package is added to.
    gdf_lake_grid : geopandas.GeoDataFrame
        Lake polygons intersected with the model grid (see :func:`_aggregate_lake_cells`).
    min_area_fraction : float, optional
        Minimum lake coverage for a cell to be carved/bounded, by default 0.5.

    Returns
    -------
    flopy.mf6.ModflowGwfriv
        The lake RIV package. When ``ds.transport`` is set, its package name is appended to
        ``ds.attrs['ssm_sources']`` (if absent) so its CONCENTRATION aux is used by SSM.
    """
    agg = _aggregate_lake_cells(ds, gdf_lake_grid, min_area_fraction=min_area_fraction).rename(
        columns={"strt": "stage", "botm": "rbot", "identificatie": "boundname"}
    )
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


def recharge_pond_mask(ds, panden_riv=None):
    """Mark cells whose meteoric input is carried by a stage boundary, not by RCH.

    Areal recharge is applied to the top active cell of every non-sea cell, which
    double-counts precipitation on the managed panden (already implicit in the prescribed
    RIV stage) and applies land P-E to the carved open-water lake cells. This mask flags
    those cells so the RCH package can exclude them.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset carrying the boolean ``ds['lake_cell']`` marker (see
        :func:`carve_lake_cells`).
    panden_riv : flopy.mf6.ModflowGwfriv or None, optional
        The infiltration-panden RIV package. When None (the default, e.g. the Bergen extent
        where the panden lie outside the grid) the mask is exactly ``ds['lake_cell']``.

    Returns
    -------
    xarray.DataArray
        Boolean mask over ``icell2d``: True at carved lake cells and at panden RIV cells.
    """
    mask = ds["lake_cell"].copy()
    if panden_riv is not None:
        cells = np.unique([cid[-1] for cid in panden_riv.stress_period_data.data[0]["cellid"]])
        mask.loc[{"icell2d": cells}] = True
    return mask
