"""Post-processing helpers for the PWN groundwater model (modelscripts/09pwnmodel2).

Covers everything after the MODFLOW run: the mass-balance discrepancy guard, loading and
deriving the head and chloride fields, the fresh/salt interface (grensvlak), and the standard
result maps. The derivations lean on nlmod where an equivalent exists (heads, concentration,
freshwater head, and the isosurface used for the interface).
"""

import logging
import os

import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import nlmod
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _check_one_budget(lst_path, budgetkey, max_pct, label):
    """Raise if the largest absolute percent discrepancy in one MODFLOW listing exceeds ``max_pct``."""
    dfs = flopy.utils.Mf6ListBudget(lst_path, budgetkey=budgetkey).get_dataframes()
    if dfs is None:
        msg = f"Could not parse a MODFLOW budget from {lst_path}"
        raise RuntimeError(msg)
    disc = float(dfs[0]["PERCENT_DISCREPANCY"].abs().max())
    logger.info("Max %s budget discrepancy: %.3f%%", label, disc)
    if disc >= max_pct:
        msg = f"{label} mass-balance discrepancy too large: {disc:.2f}%"
        raise RuntimeError(msg)


def check_budget_discrepancy(ws, model_name, *, transport=True, max_gwf_pct=1.0, max_gwt_pct=2.0):
    """Fail if the MODFLOW volumetric (and mass) budget discrepancy is too large.

    Parameters
    ----------
    ws : str
        Model workspace directory containing the ``.lst`` listing files.
    model_name : str
        Model name; listings are ``{model_name}.lst`` and ``{model_name}_gwt.lst``.
    transport : bool, optional
        Also check the transport (GWT) mass budget. Default is True.
    max_gwf_pct, max_gwt_pct : float, optional
        Maximum allowed absolute percent discrepancy for flow and transport.

    Raises
    ------
    RuntimeError
        If a budget cannot be parsed or a discrepancy meets or exceeds its threshold.
    """
    _check_one_budget(os.path.join(ws, f"{model_name}.lst"), "VOLUME BUDGET FOR ENTIRE MODEL", max_gwf_pct, "GWF")
    if transport:
        _check_one_budget(
            os.path.join(ws, f"{model_name}_gwt.lst"), "MASS BUDGET FOR ENTIRE MODEL", max_gwt_pct, "transport"
        )


def _layer_center_elevation(ds):
    """Per-layer cell-centre elevations [m NAP] as a ``(layer, icell2d)`` DataArray.

    The upper boundary of a layer is the model top for the first layer and the overlying
    layer botm otherwise; the centre is the mean of that upper boundary and the layer botm.
    """
    botm = ds["botm"]
    upper = botm.shift(layer=1)
    upper = upper.where(upper.notnull(), ds["top"])
    return (upper + botm) / 2.0


def interface_elevation(ds, concentration, threshold):
    """Fresh/salt interface elevation where chloride crosses ``threshold`` [m NAP].

    The interface is the shallowest depth at which chloride rises through ``threshold`` scanning
    down from the surface. The crossing is linearly interpolated on the layer cell-centre
    elevations (:func:`nlmod.dims.get_isosurface`). Columns that are already at or above
    ``threshold`` at the surface return the model top; columns that never reach it return the model
    bottom -- matching the previous layer-discretised ``grensvlak``.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset with ``top`` and ``botm``.
    concentration : xarray.DataArray
        Chloride concentration with a ``layer`` dimension. Dry cells (NaN) at the top or bottom of
        a column are filled from the nearest active layer before the interface is computed.
    threshold : float
        Concentration [mg Cl-/L] defining the interface.

    Returns
    -------
    xarray.DataArray
        Interface elevation [m NAP]; carries ``threshold`` in its attrs.
    """
    z = _layer_center_elevation(ds)
    # Fill dry cells at both ends so the profile is gap-free, matching the old raw-conc scan where
    # NaN cells never counted as salt; interior gaps do not occur (dry cells are contiguous).
    conc = concentration.bfill(dim="layer").ffill(dim="layer")
    # get_isosurface returns the first sign change of (conc - threshold) in either direction, but the
    # physical interface is the first up-crossing from the surface. Pin columns that are already salt
    # at the surface to the model top (this subsumes the fully-salt case) and never-salt columns to
    # the model bottom.
    gv = nlmod.dims.get_isosurface(conc, z, threshold, left=np.nan, right=np.nan)
    surface_conc = conc.isel(layer=0)
    fresh_everywhere = (conc < threshold).all(dim="layer")
    gv = xr.where(surface_conc > threshold, ds["top"], gv)
    gv = xr.where(fresh_everywhere, ds["botm"].isel(layer=-1), gv)
    gv.attrs["threshold"] = threshold
    return gv


def add_output_to_ds(ds, ws, model_name, *, transport=True, thresholds=(1000.0, 8000.0), denseref=1000.0):
    """Load the MODFLOW/MODFLOW6-GWT output and add derived fields to ``ds``.

    Adds ``head`` and ``head_filled`` and, when ``transport``, ``concentration``,
    ``conc_filled``, ``freshwater_head``, ``grensvlak_zoet``/``grensvlak_brak`` (from
    ``thresholds``), ``thickness``, ``concentration_mean`` and ``dconcentration_mean``
    (change vs. the first time step).

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset.
    ws : str
        Model workspace directory with the ``.hds``/``.ucn`` output files.
    model_name : str
        Model name used to build the output file names.
    transport : bool, optional
        Load and derive the transport (chloride) fields. Default is True.
    thresholds : tuple of float, optional
        Fresh and brackish interface thresholds [mg Cl-/L] for ``grensvlak_zoet`` and
        ``grensvlak_brak``. Default is (1000.0, 8000.0).
    denseref : float, optional
        Reference density passed to :func:`nlmod.gwt.output.freshwater_head`.

    Returns
    -------
    ds : xarray.Dataset
        The dataset with the derived fields added.
    ctop : xarray.DataArray or None
        Concentration at the groundwater surface, or None when ``transport`` is False.
    """
    ds["head"] = nlmod.gwf.output.get_heads_da(ds, fname=os.path.join(ws, f"{model_name}.hds"))
    ds["head_filled"] = ds["head"].bfill(dim="layer")

    if not transport:
        return ds, None

    conc = nlmod.gwt.output.get_concentration_da(ds, fname=os.path.join(ws, f"{model_name}_gwt.ucn"))
    ctop = nlmod.gwt.output.get_concentration_at_gw_surface(conc)
    ds["concentration"] = conc
    ds["conc_filled"] = ds["concentration"].bfill(dim="layer")

    ds["freshwater_head"] = nlmod.gwt.output.freshwater_head(
        ds, ds["head_filled"], ds["conc_filled"], denseref=denseref
    )

    threshold_fresh, threshold_brakkish = thresholds
    ds["grensvlak_zoet"] = interface_elevation(ds, ds["conc_filled"], threshold_fresh)
    ds["grensvlak_brak"] = interface_elevation(ds, ds["conc_filled"], threshold_brakkish)

    ds["thickness"] = nlmod.dims.calculate_thickness(ds)
    concentration_mean = (ds["concentration"] * ds["thickness"]).sum(dim="layer") / ds["thickness"].sum(dim="layer")
    ds["concentration_mean"] = concentration_mean
    ds["dconcentration_mean"] = concentration_mean - concentration_mean.isel(time=0)

    return ds, ctop


def _save(fig, figdir, name):
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight", dpi=150)


def plot_result_maps(ds, ctop, figdir, *, iper=-1, ilay=0):
    """Write the standard result maps to ``figdir``.

    Produces the model-grid map, an optional drain-elevation map, and (when the transport
    fields are present) freshwater-head, concentration-at-surface, and grensvlak maps for a
    single layer and stress period.

    Parameters
    ----------
    ds : xarray.Dataset
        Model dataset with the fields added by :func:`add_output_to_ds`.
    ctop : xarray.DataArray or None
        Concentration at the groundwater surface (from :func:`add_output_to_ds`).
    figdir : str
        Output directory for the PNG files.
    iper : int, optional
        Stress-period index to plot. Negative values index from the end. Default is -1 (last).
    ilay : int, optional
        Layer index for the freshwater-head map. Default is 0.
    """
    extent = ds.extent
    nper = ds.sizes["time"]
    iper = nper + iper if iper < 0 else iper
    t = pd.Timestamp(ds.time.isel(time=iper).values[()])

    fig, ax = nlmod.plot.get_map(extent, base=1e4)
    nlmod.plot.modelgrid(ds, ax=ax, color="k", lw=0.5, alpha=0.5)
    nlmod.plot.add_background_map(ax, map_provider="nlmaps.water", alpha=0.8)
    _save(fig, figdir, "doorsnedelijnen.png")

    if "drn_elev" in ds:
        f, ax = nlmod.plot.get_map(extent, base=1e4)
        ax.set_aspect("equal", adjustable="box")
        nlmod.plot.data_array(ds["drn_elev"], ds=ds)
        nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
        ax.set_title("Oppervlaktewater, infiltratie en onttrekkingen")
        _save(f, figdir, "oppervlaktewater.png")

    if "freshwater_head" not in ds:
        return

    f, ax = nlmod.plot.get_map(extent, base=1e4)
    ax.set_aspect("equal", adjustable="box")
    pc = nlmod.plot.data_array(
        ds["freshwater_head"].isel(time=iper, layer=ilay),
        ds=ds,
        norm=mpl.colors.Normalize(-5, 5),
        cmap="Spectral_r",
    )
    nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
    ax.set_title(f"$h_f$, layer={ilay}, t={t.year}-{t.month:02d}")
    f.colorbar(pc, ax=ax, shrink=0.8).set_label("freshwater head [m NAP]")
    ax.set_xlabel("X [km RD]")
    ax.set_ylabel("Y [km RD]")
    _save(f, figdir, f"map_head_L{ilay}_t{iper}.png")

    f, ax = nlmod.plot.get_map(extent, base=1e4)
    ax.set_aspect("equal", adjustable="box")
    pc = nlmod.plot.data_array(ctop.isel(time=iper), ds=ds, norm=mpl.colors.Normalize(0, 5_000.0), cmap="RdYlGn_r")
    nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
    ax.set_title(f"concentration at gw-surface, t={t.year}-{t.month:02d}")
    f.colorbar(pc, ax=ax, shrink=0.8).set_label("concentration [mg Cl-/L]")
    ax.set_xlabel("X [km RD]")
    ax.set_ylabel("Y [km RD]")
    _save(f, figdir, f"map_conc_L{ilay}_t{iper}.png")

    for gv in ["zoet", "brak"]:
        da = ds[f"grensvlak_{gv}"]
        thresh = da.attrs["threshold"]
        f, ax = nlmod.plot.get_map(extent, base=1e4)
        ax.set_aspect("equal", adjustable="box")
        pc = nlmod.plot.data_array(da.isel(time=iper), ds=ds, norm=mpl.colors.Normalize(-200, 20.0), cmap="RdYlBu_r")
        nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
        ax.set_title(f"grensvlak {gv} (cl={thresh:.0f}mg/l), t={t.year}-{t.month:02d}")
        f.colorbar(pc, ax=ax, shrink=0.8).set_label("grensvlak [m NAP]")
        ax.set_xlabel("X [km RD]")
        ax.set_ylabel("Y [km RD]")
        _save(f, figdir, f"grensvlak_{gv}_t{iper}.png")
        plt.close(f)
