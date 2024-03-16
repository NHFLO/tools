import logging
import os

import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from nhflotools.pwnlayers.layers import get_grid_gdf
from nhflotools.pwnlayers.plot_utils import get_figure

logger = logging.getLogger(__name__)


def plot_heatmap_overlap(df_overlap, figsize=(10, 10)):
    """Plot a heatmap with the overlap between two modellayers

    Parameters
    ----------
    df_overlap : pandas DataFrame
        matrix with the percentage of overlap for all layers in two
        layer models.
    figsize : tuple
        size of the figure. Default is (10,10)

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_overlap, cmap="viridis", ax=ax, xticklabels=True, yticklabels=True)
    ax.grid()

    return fig, ax


def plot_cross_section_layer_number(model_ds, gwf, figdir, x=None, y=None, cross_section_name="", ylim=(-400, 30)):
    """


    Parameters
    ----------
    model_ds : xr.DataSet
        dataset containing relevant model information
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    figdir : str
        directory to save figure.
    x : int, float or None, optional
        x-coördinate to make a cross section. The default is None.
    y : int, float or None, optional
        y-coördinate to make a cross section. The default is None.
    cross_section_name : name is used in the title, optional
        DESCRIPTION. The default is ''.
    ylim : tuple or list, optional
        limit of the y-axis. The default is (-400,30).

    Raises
    ------
    ValueError
        Either x or y should be defined, not both.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    assert model_ds.gridtype == "structured", f"gridtype should be structured, not {model_ds.gridtype}"

    # array met laagnummer van iedere modelcel
    lay_no = xr.zeros_like(model_ds["botm"])
    for i in range(model_ds.dims["layer"]):
        lay_no[i] = i

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)

    if (x is None) and (y is None):
        raise ValueError("please assign x or y value to plot stitches")
    if not ((x is None) or (y is None)):
        raise ValueError("please assign x or y, not both")
    if x is None:
        row = gwf.modelgrid.intersect(model_ds.extent[1] - 1, y)[0]
        np.array([(model_ds.extent[0], y), (model_ds.extent[1], y)])
        xsect = flopy.plot.PlotCrossSection(model=gwf, line={"Row": row})
    elif y is None:
        column = gwf.modelgrid.intersect(x + 1, model_ds.extent[2])[1]
        np.array([(x, model_ds.extent[2]), (x, model_ds.extent[3])])

        xsect = flopy.plot.PlotCrossSection(model=gwf, line={"Column": column})

    csa = xsect.plot_array(lay_no, cmap="tab20", alpha=0.5)

    # plot laagscheidingen
    cmap = mpl.colormaps["tab20"]

    if x is None:
        for i, lay in enumerate(model_ds.layer):
            lay_bot = model_ds["botm"].sel(layer=lay)[row, :]
            lay_bot_dis = lay_bot.x.values - lay_bot.x.values.min()
            color_num = lay_no[i][0][0]
            ax.plot(lay_bot_dis, lay_bot.values, color=cmap(color_num / 10), lw=2, ls=":")
            ax.set_title(f"y: {y} Cross-Section {cross_section_name} with layer numbers")
    if y is None:
        for i, lay in enumerate(model_ds.layer):
            lay_bot = model_ds["botm"].sel(layer=lay)[:, column]
            lay_bot_dis = (lay_bot.y.values - lay_bot.y.values.min())[::-1]
            color_num = lay_no[i][0][0]
            ax.plot(lay_bot_dis, lay_bot.values, color=cmap(color_num / 10), lw=2, ls=":")
            ax.set_title(f"x: {x} Cross-Section {cross_section_name} with layer numbers")

    ax.set_ylim(ylim)

    plt.colorbar(csa, shrink=0.75)

    fig.savefig(
        os.path.join(figdir, f"stitches_{cross_section_name}.png"),
        bbox_inches="tight",
        dpi=300,
    )

    return fig, ax


def plot_shapes(m, gdf, field, figpath, figname):
    if not os.path.isdir(figpath):
        os.makedirs(figpath)
    f, ax = get_figure(m, title=figname)
    if not gdf.empty:
        gdf.plot(field, ax=ax, legend=True)
    f.savefig(os.path.join(figpath, figname))
    plt.close(f)


def plot_data(m, data, figpath):
    # plot the data in the data-dictionary on a map
    for key in data:
        if isinstance(data[key].values, np.ndarray):
            f, ax = get_figure(m, title=key)
            # ax.imshow(data[key],extent=m.sr.get_extent())
            pcm = ax.pcolormesh(m.sr.xgrid, m.sr.ygrid, data[key])
            plt.colorbar(pcm, ax=ax)
            f.savefig(os.path.join(figpath, key))
            plt.close(f)


def plot_grid(grid, extent, extent2=None, zoom=5, loc="upper left", loc1=3, loc2=4):
    gdf = get_grid_gdf(grid, extent=extent)
    riv_gdf = get_grid_gdf(grid, kind="rivers")
    f, ax = get_figure(extent, title="Triwaco grid", figsize=(5.9, 9))
    gdf.plot(ax=ax, edgecolor="k", linewidth=0.5)
    riv_gdf.plot(ax=ax, alpha=0.5, color="r", linewidth=2)
    if extent2 is None:
        return f, ax
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

    ax2 = zoomed_inset_axes(ax, zoom=zoom, loc=loc)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    # geef met hulplijnen de locatie van ax3 weer
    mark_inset(ax, ax2, loc1=loc1, loc2=loc2, edgecolor="w", linewidth=3)
    plt.setp(ax2.spines.values(), color="w", linewidth=3)
    gdf2 = get_grid_gdf(grid, extent=extent2)
    riv_gdf2 = get_grid_gdf(grid, kind="rivers", extent=extent2)
    gdf2.plot(ax=ax2, edgecolor="k", linewidth=0.5)
    riv_gdf2.plot(ax=ax2, alpha=0.5, color="r", linewidth=4)
    ax2.axis("equal")
    ax2.axis(extent2)
    return f, [ax, ax2]
