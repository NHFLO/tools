"""Cross-section plotting utilities for the PWN layer model."""

import re

import geopandas
import matplotlib.pyplot as plt
import nlmod
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import LineString, Point

from nhflotools.pwnlayers3.layers import layer_names


def plot_diagnostic_cross_sections(
    ds,
    ds_regis,
    line,
    zmin=-120.0,
    zmax=25.0,
    figsize=(14, 36),
    min_label_area=1000.0,
    fontsize=None,
    data_path_2024=None,
    buffer_distance=500.0,
):
    """Plot 6 aligned cross-sections for layer model diagnostics.

    Produces a figure with six vertically stacked subplots:

    1. Horizontal hydraulic conductivity (kh) of the merged layer model.
    2. Horizontal hydraulic conductivity (kh) of the REGIS layer model.
    3. Source category of botm values (REGIS / PWN / Transition).
    4. Computation method used for PWN botm values.
    5. Computation method used for PWN kh values.
    6. Computation method used for PWN kv values.

    When ``data_path_2024`` is provided, the raw source botm point data from
    ``botm.geojson`` is projected onto the cross-section line and overlaid
    as scatter markers on subplots 1, 3, and 4. This allows direct comparison
    of raw source data with the interpolated layer boundaries.

    Parameters
    ----------
    ds : xr.Dataset
        Merged layer model from
        ``get_pwn_layer_model(return_diagnostics=True)``. Must contain
        diagnostic variables ``cat_botm``, ``botm_pwn``, ``botm_method``,
        ``kh_method``, and ``kv_method``.
    ds_regis : xr.Dataset
        REGIS layer model dataset.
    line : list of tuple
        Cross-section line as ``[(x1, y1), (x2, y2)]``.
    zmin : float, optional
        Minimum z-coordinate for the cross-section. Default is -120.0.
    zmax : float, optional
        Maximum z-coordinate for the cross-section. Default is 25.0.
    figsize : tuple of float, optional
        Figure size as ``(width, height)`` in inches. Default is (14, 36).
    min_label_area : float, optional
        Minimum polygon area (in data units) for a layer label to be
        displayed. Default is 1000.0.
    fontsize : float or None, optional
        Font size for layer labels. If None, matplotlib's default is used.
    data_path_2024 : pathlib.Path or None, optional
        Path to the 2024 PWN data directory containing ``botm/botm.geojson``.
        When provided, raw source botm points are overlaid on the
        cross-sections. Default is None.
    buffer_distance : float, optional
        Maximum perpendicular distance (m) from the cross-section line for
        source points to be included. Default is 500.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)

    # Determine shared kh color limits from the REGIS model
    kh_regis_valid = ds_regis["kh"].values[np.isfinite(ds_regis["kh"].values)]
    kh_vmin = float(np.nanmin(kh_regis_valid)) if kh_regis_valid.size else 0.1
    kh_vmax = float(np.nanmax(kh_regis_valid)) if kh_regis_valid.size else 100.0
    kh_norm = plt.Normalize(vmin=kh_vmin, vmax=kh_vmax)

    # --- Prepare source botm points for overlay ---
    source_points = None
    if data_path_2024 is not None:
        source_points = _load_and_project_source_botm(
            data_path_2024=data_path_2024,
            line=line,
            buffer_distance=buffer_distance,
        )

    # --- Subplot 1: kh of merged layer model ---
    dcs = nlmod.plot.DatasetCrossSection(ds, line=line, ax=axes[0], zmin=zmin, zmax=zmax)
    pc = dcs.plot_array(ds["kh"], cmap="viridis", norm=kh_norm)
    dcs.plot_layers(min_label_area=min_label_area, fontsize=fontsize, only_labels=True)
    dcs.plot_grid(linewidth=0.5, vertical=False)
    if source_points is not None:
        _overlay_source_botm(axes[0], source_points, zmin=zmin, zmax=zmax)
    axes[0].set_ylabel("mNAP")
    axes[0].set_title("kh merged model")
    fig.colorbar(pc, ax=axes[0], label="kh (m/d)")

    # --- Subplot 2: kh of REGIS layer model ---
    dcs = nlmod.plot.DatasetCrossSection(ds_regis, line=line, ax=axes[1], zmin=zmin, zmax=zmax)
    pc = dcs.plot_array(ds_regis["kh"], cmap="viridis", norm=kh_norm)
    dcs.plot_layers(min_label_area=min_label_area, fontsize=fontsize, only_labels=True)
    dcs.plot_grid(linewidth=0.5, vertical=False)
    axes[1].set_ylabel("mNAP")
    axes[1].set_title("kh REGIS model")
    fig.colorbar(pc, ax=axes[1], label="kh (m/d)")

    # --- Subplot 3: source category (botm) ---
    cat_cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
    cat_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ncolors=3)
    dcs = nlmod.plot.DatasetCrossSection(ds, line=line, ax=axes[2], zmin=zmin, zmax=zmax)
    pc = dcs.plot_array(ds["cat_botm"], cmap=cat_cmap, norm=cat_norm)
    dcs.plot_layers(min_label_area=min_label_area, fontsize=fontsize, only_labels=True)
    dcs.plot_grid(linewidth=0.5, vertical=False)
    if source_points is not None:
        _overlay_source_botm(axes[2], source_points, zmin=zmin, zmax=zmax)
    axes[2].set_ylabel("mNAP")
    axes[2].set_title("Source category (botm)")
    cbar = fig.colorbar(pc, ax=axes[2], ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(["REGIS", "PWN", "Transition"])

    # --- Subplots 4-6: PWN method arrays ---
    # Reconstruct a Dataset with PWN layer geometry for DatasetCrossSection
    ds_pwn = xr.Dataset(
        {
            "top": ds["top"],
            "botm": ds["botm_pwn"].rename({"layer_pwn": "layer"}),
            "xv": ds["xv"],
            "yv": ds["yv"],
            "icvert": ds["icvert"],
        },
        attrs=ds.attrs,
    )

    method_configs = [
        ("botm_method", "Source method botm (PWN)", 3),
        ("kh_method", "Source method kh (PWN)", 4),
        ("kv_method", "Source method kv (PWN)", 5),
    ]
    for var_name, title, ax_idx in method_configs:
        method_da = ds[var_name].rename({"layer_pwn": "layer"})
        _plot_method_cross_section(
            fig=fig,
            ax=axes[ax_idx],
            ds_pwn=ds_pwn,
            method_da=method_da,
            line=line,
            zmin=zmin,
            zmax=zmax,
            title=title,
            min_label_area=min_label_area,
            fontsize=fontsize,
        )

    # Overlay source botm points on the botm_method subplot
    if source_points is not None:
        _overlay_source_botm(axes[3], source_points, zmin=zmin, zmax=zmax)

    axes[-1].set_xlabel("Distance along cross-section (m)")
    fig.tight_layout()
    return fig, axes


def _parse_flag_labels(flag_meanings):
    """Extract short descriptive labels from a flag_meanings string.

    Parameters
    ----------
    flag_meanings : str
        Semicolon-separated flag descriptions in the format
        ``"0: short_name (long description); 1: short_name (long ...); ..."``.

    Returns
    -------
    list of str
        One human-readable label per flag, e.g. ``["no data", "linear interp."]``.
    """
    labels = []
    for entry in flag_meanings.split(";"):
        stripped = entry.strip()
        if not stripped:
            continue
        # Strip the leading "N: " prefix
        match = re.match(r"\d+:\s*(.*)", stripped)
        text = match.group(1) if match else stripped
        # Take the short name before any parenthesis, replace underscores
        short = text.split("(")[0].strip().replace("_", " ")
        labels.append(short)
    return labels


def _plot_method_cross_section(*, fig, ax, ds_pwn, method_da, line, zmin, zmax, title, min_label_area, fontsize):
    """Plot a single method DataArray as a discrete cross-section.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the colorbar to.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    ds_pwn : xr.Dataset
        PWN layer model dataset with geometry (top, botm, xv, yv, icvert).
    method_da : xr.DataArray
        Integer method DataArray with ``flag_values`` and ``flag_meanings``
        attributes.
    line : list of tuple
        Cross-section line.
    zmin, zmax : float
        Vertical extent.
    title : str
        Subplot title.
    min_label_area : float
        Minimum polygon area for layer labels.
    fontsize : float or None
        Font size for layer labels.
    """
    flag_values = method_da.attrs["flag_values"]
    flag_meanings = method_da.attrs["flag_meanings"]
    n_flags = len(flag_values)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_flags, 3)))[:n_flags]
    method_cmap = ListedColormap(colors)
    boundaries = [v - 0.5 for v in flag_values] + [flag_values[-1] + 0.5]
    method_norm = BoundaryNorm(boundaries, ncolors=n_flags)

    dcs = nlmod.plot.DatasetCrossSection(ds_pwn, line=line, ax=ax, zmin=zmin, zmax=zmax)
    pc = dcs.plot_array(method_da, cmap=method_cmap, norm=method_norm)
    dcs.plot_layers(min_label_area=min_label_area, fontsize=fontsize, only_labels=True)
    dcs.plot_grid(linewidth=0.5, vertical=False)
    ax.set_ylabel("mNAP")
    ax.set_title(title)

    cbar = fig.colorbar(pc, ax=ax, ticks=flag_values)
    labels = _parse_flag_labels(flag_meanings)
    if len(labels) == n_flags:
        cbar.ax.set_yticklabels(labels)


def _load_and_project_source_botm(*, data_path_2024, line, buffer_distance):
    """Load source botm point data and project onto a cross-section line.

    Parameters
    ----------
    data_path_2024 : pathlib.Path
        Path to the 2024 PWN data directory containing ``botm/botm.geojson``.
    line : list of tuple
        Cross-section line as ``[(x1, y1), (x2, y2)]``.
    buffer_distance : float
        Maximum perpendicular distance (m) from the line for a point to
        be included.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"d_along"`` : np.ndarray of distances along the line.
        - ``"layers"`` : dict mapping layer name (str) to np.ndarray of
          botm elevations for the selected points.
    """
    fp = data_path_2024 / "botm" / "botm.geojson"
    gdf_botm = geopandas.read_file(fp, driver="GeoJSON")

    cs_line = LineString(line)
    pts = gdf_botm.geometry
    distances_to_line = np.array([cs_line.distance(Point(p.x, p.y)) for p in pts])
    sel = distances_to_line <= buffer_distance

    d_along = np.array([cs_line.project(Point(p.x, p.y)) for p in pts[sel]])

    layers = {}
    for name in layer_names:
        if name in gdf_botm.columns:
            layers[name] = gdf_botm.loc[sel, name].values.astype(float)

    return {"d_along": d_along, "layers": layers}


def _overlay_source_botm(ax, source_points, *, zmin, zmax):
    """Overlay raw source botm points as scatter markers on a cross-section.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    source_points : dict
        Output from ``_load_and_project_source_botm``.
    zmin, zmax : float
        Vertical extent used to filter points.
    """
    d_along = source_points["d_along"]
    layers = source_points["layers"]
    n_layers = len(layers)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_layers, 1)))

    for idx, (name, z_values) in enumerate(layers.items()):
        valid = np.isfinite(z_values) & (z_values >= zmin) & (z_values <= zmax)
        if valid.any():
            ax.scatter(
                d_along[valid],
                z_values[valid],
                s=8,
                color=cmap[idx % len(cmap)],
                marker="o",
                linewidths=0.3,
                edgecolors="k",
                zorder=5,
                label=name,
            )
