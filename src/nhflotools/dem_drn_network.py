r"""Build a bidirectional fill-and-spill DRN+MVR routing network from a DEM.

Preprocesses a digital elevation model defined on a flopy DISV
:class:`flopy.discretization.VertexGrid` and produces a bidirectional
surface-runoff routing network that wires into MODFLOW 6 as a collection
of DRN cells (one per directed cell-to-neighbour link, each gated at the
local saddle elevation) and MVR records (transferring DRN discharge to
the receiver cell).

For every adjacent pair ``(i, j)`` two drains are emitted, both gated at
the saddle elevation :math:`z_{saddle} = \max(\mathrm{dem}[i],
\mathrm{dem}[j])`. The drain in cell ``i`` fires when ``h_i > z_saddle``
and the corresponding mover transfers its discharge to ``j``;
symmetrically for ``j -> i``. Net flow goes from the higher-head cell to
the lower-head cell, with direction decided at runtime by the heads --
not baked into preprocessing. Multi-cell depressions therefore fill
realistically: water redistributes through the basin's interior saddles
until some rim-saddle is overtopped, at which point overflow leaves the
basin via that direction.

Boundary cells additionally receive a one-way drain to outside the model
gated at the cell's own DEM elevation. Cells with an existing stress BC
(DRN, LAK, GHB, CHD, RIV) are excluded from the network: an edge to such
a cell becomes a one-way outlet at the *non-excluded* end so water still
leaves the routing system there. Genuine deflation hollows can be marked
as closed sinks; they neither spill nor receive routed water and
accumulate it indefinitely.

Output is intentionally a pandas DataFrame, not flopy stress data: the
MVR receiver must be a package that absorbs inflow (UZF, SFR, LAK, MAW)
and the choice depends on the surface representation the caller has set
up. Wiring the DataFrame into flopy is straightforward.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Stress packages whose presence on a cell means the cell already drains
# or receives water and should be excluded from the surface DRN network.
_EXCLUDING_PACKAGES = ("drn", "lak", "ghb", "chd", "riv")
_MIN_SHARED_VERTS = 2
_DISV_CELLID_LEN = 2


def build_bidirectional_drn_network(  # noqa: C901
    *,
    dem,
    modelgrid,
    gwf=None,
    excluded_cells=None,
    closed_sink_cells=None,
    outlet_cells=None,
    cond_per_meter=1000.0,
    layer=0,
):
    r"""Build a bidirectional saddle-gated DRN+MVR routing network on a DISV grid.

    For every rook-adjacent pair of non-excluded, non-sink cells, two drains
    and two movers are emitted at the local saddle elevation
    ``max(dem[i], dem[j])``. Direction of net flow is decided at runtime by
    the heads -- water flows from the higher-head cell to the lower-head cell
    whenever the higher head is above the saddle. Boundary cells additionally
    receive a one-way drain to outside the model, gated at the cell's own DEM.

    Parameters
    ----------
    dem : array-like, shape (ncpl,)
        Cell-centre elevation in metres, indexed by ``icell2d`` of the
        model grid.
    modelgrid : flopy.discretization.VertexGrid
        DISV model grid. Used for the rook-neighbour adjacency graph and
        shared-edge geometry.
    gwf : flopy.mf6.ModflowGwf, optional
        If supplied, every cell carrying a stress boundary condition in
        any of {DRN, LAK, GHB, CHD, RIV} is added to the excluded set.
    excluded_cells : array-like of int, optional
        Cell ids (icell2d) where no drain should be placed. A connection
        to an excluded cell becomes a one-way outlet at the non-excluded
        end (water above the saddle leaves the routing network there).
    closed_sink_cells : array-like of int, optional
        Cell ids representing genuine deflation hollows that should never
        spill. No drain is placed at the sink and no mover routes water
        *into* it; water accumulates and is expected to leave through ET
        or vertical leakage configured outside this network.
    outlet_cells : array-like of int, optional
        Cell ids where surface water leaves the model domain. Each listed
        cell receives an extra one-way drain at its own DEM elevation. If
        None, every grid-boundary cell (one with fewer rook-neighbours
        than polygon edges) is treated as an outlet.
    cond_per_meter : float, optional
        Drain conductance per metre of edge length, in m²/d per m of head
        (i.e. m/d). Discharge at each drain is
        ``Q = cond_per_meter * L_edge * max(0, h - z_saddle)``. See Notes
        for the physical interpretation. Default is 1000.0.
    layer : int, optional
        Model layer in which the drains are placed. Stored on
        ``df.attrs["layer"]`` so a caller can build flopy stress data with
        ``cellid = (layer, icell2d)``. Default is 0.

    Returns
    -------
    pandas.DataFrame
        One row per directed DRN+MVR link with columns
        ``["from_cell", "to_cell", "z_saddle", "cond", "edge_length",
        "is_outlet"]``.

        ``from_cell`` (int64) is the cell hosting the drain; ``to_cell``
        (Int64, nullable) is the receiver cell, ``pd.NA`` when the drain
        is an outlet. ``z_saddle`` is the drain elevation in metres
        (``max(dem[from], dem[to])`` for interior pairs, ``dem[from]``
        for grid-boundary outlets). ``cond`` is the drain conductance in
        m²/d. ``edge_length`` is the shared-edge length for interior
        pairs and the boundary-segment length for boundary outlets, in
        metres. ``is_outlet`` flags rows for which no mover is emitted.

    Notes
    -----
    **Conductance.** The DRN's stage-discharge relation is linear:

    .. math:: Q = c \, L_{edge} \, \max(0, h - z_{saddle})
              \quad \mathrm{[m^3/d]}

    with ``c = cond_per_meter`` (units m^2/d per m of head, i.e. m/d).
    This is the linearised analogue of a broad-crested weir
    :math:`Q = C_w L h^{1.5}`; calibrate ``cond_per_meter`` to match
    weir behaviour at a representative excess head ``dh = h - z_saddle``.

    Order-of-magnitude guidance for surface runoff in a dune setting
    (cell sizes 25-100 m, daily time step):

    * 100 m/d   -- slow spillover; basins retain water for several days.
    * 1000 m/d  -- moderate (default); suitable for daily-resolved runoff.
    * 1e4 m/d   -- fast; water spills almost as quickly as it arrives.

    A self-consistent target time-scale is the cell drainage time
    :math:`\tau \approx A_{cell} / (c \, L_{edge})`. For 100 m by 100 m
    cells (:math:`A = 10^4` m^2) and ``cond_per_meter = 1000`` with
    ``L_edge = 100`` m, the cell drains excess depth in ``tau = 0.1`` d.

    **Bidirectional saddle gates.** Every adjacent pair gets two drains
    at the same saddle elevation. The pair acts like a threshold valve
    that opens whenever either side's head exceeds the saddle, with net
    transfer toward the lower-head cell. Fill-and-spill emerges without
    DEM preprocessing: water inside a multi-cell depression
    redistributes through interior saddles, and overflow leaves through
    whichever rim-saddle is overtopped first.

    **Trapped components.** After excluded and closed-sink cells are
    removed, the remaining adjacency graph is checked for connected
    components that contain no outlet. Such components accumulate water
    with no escape path and are reported via :data:`logger.warning`.
    Inspect the DataFrame and consider extending ``outlet_cells`` or
    revising the exclusion list.

    **Solver.** All drains are linear in head and contribute only
    diagonal/off-diagonal terms. MVR moves discharge in an explicit
    outer-iteration loop, so cell-to-cell coupling does not enter the
    Jacobian. With dune-area daily time steps and short cascades the
    chain converges in a handful of outer iterations. Bump
    ``outer_maximum`` in the IMS package to comfortably exceed the
    longest expected cascade length.

    References
    ----------
    .. [1] Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: An
       optimal depression-filling and watershed-labeling algorithm for
       digital elevation models. Computers & Geosciences, 62, 117-127.
    """
    dem = np.asarray(dem, dtype=float)
    ncpl = modelgrid.ncpl
    if dem.shape != (ncpl,):
        msg = f"dem shape {dem.shape} does not match modelgrid.ncpl={ncpl}."
        raise ValueError(msg)

    excluded = _collect_excluded_cells(gwf=gwf, explicit=excluded_cells, ncpl=ncpl)
    closed_sinks = _to_cell_set(closed_sink_cells)
    overlap = excluded & closed_sinks
    if overlap:
        msg = f"Cells {sorted(overlap)} appear in both excluded and closed-sink sets."
        raise ValueError(msg)

    neighbour_dict = modelgrid.neighbors(method="rook")

    outlets = _detect_boundary_cells(modelgrid, neighbour_dict) if outlet_cells is None else _to_cell_set(outlet_cells)
    outlets -= closed_sinks
    outlets -= excluded

    rows = []
    seen_pairs = set()
    for icell, neighbours in neighbour_dict.items():
        if icell in closed_sinks:
            continue
        for jcell in neighbours:
            if jcell in closed_sinks:
                continue
            pair = (min(icell, jcell), max(icell, jcell))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rows.extend(
                _emit_pair_rows(
                    icell=icell,
                    jcell=jcell,
                    dem=dem,
                    modelgrid=modelgrid,
                    excluded=excluded,
                    cond_per_meter=cond_per_meter,
                )
            )

    for icell in sorted(outlets):
        boundary_length = _boundary_edge_length(modelgrid, icell, neighbour_dict)
        if not np.isfinite(boundary_length) or boundary_length <= 0.0:
            continue
        rows.append(
            _row(
                int(icell),
                pd.NA,
                float(dem[icell]),
                cond_per_meter * boundary_length,
                boundary_length,
                is_outlet=True,
            )
        )

    df = pd.DataFrame(
        rows,
        columns=["from_cell", "to_cell", "z_saddle", "cond", "edge_length", "is_outlet"],
    )
    if len(df):
        df["from_cell"] = df["from_cell"].astype(np.int64)
        df["to_cell"] = df["to_cell"].astype("Int64")
        df["is_outlet"] = df["is_outlet"].astype(bool)
    df.attrs["layer"] = int(layer)
    df.attrs["cond_per_meter"] = float(cond_per_meter)

    _check_trapped_components(df=df, ncpl=ncpl, excluded=excluded, closed_sinks=closed_sinks)
    return df


def plot_drn_network(
    *,
    df,
    modelgrid,
    dem,
    ax=None,
    cmap="terrain",
    arrow_color="0.2",
    arrow_scale=0.85,
):
    """Diagnostic plot of a bidirectional fill-and-spill DRN network.

    Cells are coloured by DEM elevation. Each directed DRN+MVR link is
    drawn as a small arrow from cell-centre ``i`` toward cell-centre
    ``j``; the two members of a bidirectional pair appear as
    near-parallel arrows offset perpendicular to the link so they do
    not overlap. Boundary outlets and excluded-edge outlets are marked
    with red squares at the source cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`build_bidirectional_drn_network`.
    modelgrid : flopy.discretization.VertexGrid
        DISV model grid.
    dem : array-like, shape (ncpl,)
        Cell-centre elevation, m.
    ax : matplotlib.axes.Axes, optional
        Existing axis. A new figure is created when None.
    cmap : str, optional
        Colour map for the DEM. Default is ``"terrain"``.
    arrow_color : str, optional
        Colour of the flow-direction arrows. Default is ``"0.2"``.
    arrow_scale : float, optional
        Fraction of the inter-cell distance covered by each arrow. ``<1``
        leaves a small gap so opposing arrows do not overlap. Default
        is 0.85.

    Returns
    -------
    matplotlib.axes.Axes
    """
    dem = np.asarray(dem, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    polygons = [np.asarray(modelgrid.get_cell_vertices(i)) for i in range(modelgrid.ncpl)]
    pc = PolyCollection(polygons, array=dem, cmap=cmap, edgecolors="0.7", linewidths=0.2)
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax, label="DEM (m)")

    xc = np.asarray(modelgrid.xcellcenters)
    yc = np.asarray(modelgrid.ycellcenters)

    interior = df[~df["is_outlet"] & df["to_cell"].notna()]
    if len(interior):
        i = interior["from_cell"].to_numpy(dtype=np.int64)
        j = interior["to_cell"].to_numpy(dtype=np.int64)
        dx = xc[j] - xc[i]
        dy = yc[j] - yc[i]
        d = np.hypot(dx, dy)
        d_safe = np.where(d == 0.0, 1.0, d)
        nx = -dy / d_safe
        ny = dx / d_safe
        offset = 0.05 * d
        x0 = xc[i] + nx * offset
        y0 = yc[i] + ny * offset
        u = dx * arrow_scale
        v = dy * arrow_scale
        ax.quiver(
            x0,
            y0,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=arrow_color,
            width=0.0025,
        )

    outlet = df[df["is_outlet"]]
    if len(outlet):
        i = outlet["from_cell"].to_numpy(dtype=np.int64)
        ax.plot(xc[i], yc[i], "rs", markersize=4, label="outlet")
        ax.legend(loc="best")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Bidirectional fill-and-spill DRN/MVR network")
    return ax


def _icell2d_from_cellid(cellid):
    """Return ``icell2d`` from a flopy cellid, or None if not interpretable.

    DISV cellids are tuples ``(layer, icell2d)``; DISU are scalar ints. A
    DIS cellid ``(k, i, j)`` is intentionally rejected since this module
    is DISV-only.
    """
    if isinstance(cellid, tuple):
        if len(cellid) != _DISV_CELLID_LEN:
            return None
        inner = cellid[1]
    else:
        inner = cellid
    if isinstance(inner, (int, np.integer)):
        return int(inner)
    return None


def _emit_pair_rows(*, icell, jcell, dem, modelgrid, excluded, cond_per_meter):
    """Build the DRN+MVR rows for a single rook-adjacent (icell, jcell) pair.

    Returns an empty list when both cells are excluded, when their shared
    edge has zero length, or when both ends of the edge would be skipped
    for any other reason. A single-row list is returned when exactly one
    of the cells is excluded (one-way drain at the non-excluded end).
    """
    edge_length = _shared_edge_length(modelgrid, icell, jcell)
    if not np.isfinite(edge_length) or edge_length <= 0.0:
        return []
    z_saddle = float(max(dem[icell], dem[jcell]))
    cond = cond_per_meter * edge_length

    i_excl = icell in excluded
    j_excl = jcell in excluded
    if i_excl and j_excl:
        return []
    if i_excl:
        return [_row(int(jcell), pd.NA, z_saddle, cond, edge_length, is_outlet=True)]
    if j_excl:
        return [_row(int(icell), pd.NA, z_saddle, cond, edge_length, is_outlet=True)]
    return [
        _row(int(icell), int(jcell), z_saddle, cond, edge_length, is_outlet=False),
        _row(int(jcell), int(icell), z_saddle, cond, edge_length, is_outlet=False),
    ]


def _row(from_cell, to_cell, z_saddle, cond, edge_length, *, is_outlet):
    """Single DataFrame-row dict for a directed DRN+MVR link."""
    return {
        "from_cell": from_cell,
        "to_cell": to_cell,
        "z_saddle": z_saddle,
        "cond": cond,
        "edge_length": edge_length,
        "is_outlet": is_outlet,
    }


def _collect_excluded_cells(*, gwf, explicit, ncpl):
    """Union of explicit excluded cells and cells with existing stress BCs.

    Iterates each of :data:`_EXCLUDING_PACKAGES` on the GWF model and
    extracts every ``cellid`` listed in ``stress_period_data``. DISV
    cellids are tuples ``(layer, icell2d)``; only ``icell2d`` is kept.
    """
    excluded = _to_cell_set(explicit)
    if gwf is None:
        return excluded
    for pname in _EXCLUDING_PACKAGES:
        pkg = gwf.get_package(pname)
        if pkg is None:
            continue
        spd = getattr(pkg, "stress_period_data", None)
        if spd is None:
            continue
        data = getattr(spd, "data", None)
        if data is None:
            continue
        for recarray in data.values():
            if recarray is None:
                continue
            for rec in recarray:
                icell = _icell2d_from_cellid(rec["cellid"])
                if icell is not None and 0 <= icell < ncpl:
                    excluded.add(icell)
    return excluded


def _to_cell_set(cells):
    """Turn an array-like (or None) into a set of integer cell ids."""
    if cells is None:
        return set()
    return {int(c) for c in np.asarray(cells).ravel()}


def _detect_boundary_cells(modelgrid, neighbour_dict):
    """Cells with at least one polygon edge not shared with another cell.

    A cell on the model boundary has fewer rook-neighbours than its
    polygon has edges (= number of vertices for a simple polygon).
    """
    boundary = set()
    iverts = modelgrid.iverts
    for icell in range(modelgrid.ncpl):
        n_edges = len(iverts[icell])
        n_neighbours = len(neighbour_dict.get(icell, ()))
        if n_neighbours < n_edges:
            boundary.add(icell)
    return boundary


def _shared_edge_length(modelgrid, icell, jcell):
    """Euclidean length of the edge shared between two rook-neighbour cells.

    Two cells that share a single edge have exactly two vertices in common
    and the edge length is the Euclidean distance between them. On
    quadtree-refinement boundaries a coarse cell can share a polyline of
    consecutive edges with several smaller neighbours; in that case the
    function sums the lengths of the consecutive segments in
    ``iverts[icell]`` whose endpoints both belong to the shared vertex
    set.
    """
    iverts_i = modelgrid.iverts[icell]
    iverts_j = modelgrid.iverts[jcell]
    shared = set(iverts_i) & set(iverts_j)
    if len(shared) < _MIN_SHARED_VERTS:
        return float("nan")
    verts = np.asarray(modelgrid.verts)
    if len(shared) == _MIN_SHARED_VERTS:
        a, b = tuple(shared)
        return float(np.linalg.norm(verts[a] - verts[b]))
    total = 0.0
    n = len(iverts_i)
    for k in range(n):
        a = iverts_i[k]
        b = iverts_i[(k + 1) % n]
        if a in shared and b in shared:
            total += float(np.linalg.norm(verts[a] - verts[b]))
    return total


def _boundary_edge_length(modelgrid, icell, neighbour_dict):
    """Total length of polygon edges of ``icell`` not shared with any neighbour.

    Boundary cells discharge to outside the model along these edges; the
    sum is used as the conductance length scale for the boundary-outlet
    drain.
    """
    iverts_i = modelgrid.iverts[icell]
    verts = np.asarray(modelgrid.verts)
    shared_segments = set()
    for jcell in neighbour_dict.get(icell, ()):
        iverts_j = modelgrid.iverts[jcell]
        common = set(iverts_i) & set(iverts_j)
        n = len(iverts_i)
        for k in range(n):
            a = iverts_i[k]
            b = iverts_i[(k + 1) % n]
            if a in common and b in common:
                shared_segments.add(frozenset((a, b)))
    total = 0.0
    n = len(iverts_i)
    for k in range(n):
        a = iverts_i[k]
        b = iverts_i[(k + 1) % n]
        if frozenset((a, b)) in shared_segments:
            continue
        total += float(np.linalg.norm(verts[a] - verts[b]))
    return total


def _check_trapped_components(*, df, ncpl, excluded, closed_sinks):
    """Warn about routable cells with no path to any outlet.

    Builds an undirected adjacency from interior rows of ``df``, runs
    SciPy's :func:`scipy.sparse.csgraph.connected_components`, and flags
    components that contain at least one routable cell but no outlet.
    """
    if len(df) == 0:
        return
    interior = df[~df["is_outlet"] & df["to_cell"].notna()]
    if len(interior) == 0:
        return

    rows = interior["from_cell"].to_numpy(dtype=np.int64)
    cols = interior["to_cell"].to_numpy(dtype=np.int64)
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(ncpl, ncpl))

    _, labels = connected_components(adj, directed=False)

    outlet_sources = df.loc[df["is_outlet"], "from_cell"].to_numpy(dtype=np.int64)
    outlet_components = set(labels[outlet_sources].tolist()) if outlet_sources.size else set()

    routable = np.ones(ncpl, dtype=bool)
    for c in excluded | closed_sinks:
        routable[c] = False
    in_network = np.zeros(ncpl, dtype=bool)
    in_network[df["from_cell"].to_numpy(dtype=np.int64)] = True

    component_label = labels
    is_trapped = ~np.isin(component_label, list(outlet_components)) & in_network & routable
    trapped = np.where(is_trapped)[0]
    if trapped.size:
        msg = (
            f"{trapped.size} cells have no routing path to an outlet "
            f"(first 10: {trapped[:10].tolist()}). Consider extending "
            "outlet_cells or revising excluded_cells / closed_sink_cells."
        )
        logger.warning(msg)
