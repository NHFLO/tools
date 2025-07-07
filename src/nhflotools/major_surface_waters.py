"""Functions for handling major surface waters in NHFLO."""

import nlmod
import numpy as np
import xarray as xr


def get_chd_ghb_data_from_major_surface_waters(ds, da_name="rws_oppwater", cachedir=None):
    """Get chd and ghb data from major surface waters.

    De grote oppervlaktewaterlichamen in het model zijn de Noordzee, de Waddenzee, het IJsselmeer en het Noordzeekanaal.
    Deze zijn in het model ingevoerd door shapefiles van de ligging van deze wateren te versnijden met het modelgrid.
    Deze shapefiles zijn gedownload in 2021 van Rijkswaterstaat en standaard beschikbaar gesteld binnen NLMOD. Daarbij
    is onderscheid gemaakt tussen de zee en de overige grote oppervlaktewaterlichamen. De zee randvoorwaarden zijn in
    het model opgenomen via de Constant Head package (CHD). Het peil is aan het begin van de simulatie ingesteld op
    NAP+4 cm (Klimaatdashboard KNMI 2021). De zee kan door de zeebodem dus zowel water infiltreren als draineren. De zee
    heeft een vaste chloride concentratie van 18.000 mg Cl-/l. Deze concentratie is vastgezet in het transportmodel via
    de Constant Concentration (CNC) package.

    De overige grote oppervlaktewaterlichamen, het IJsselmeer, het Markermeer en het Noordzeekanaal, zijn ingevoerd met
    de General Head Boundary package (GHB). Daarbij is een bodemweerstand van 1 dag toegepast in combinatie met het
    oppervlak van het oppervlaktewater per cel om een conductance uit te rekenen. Het peil is gebaseerd op het
    gemiddelde peil dat wordt gehanteerd volgens Rijkswaterstaat. De concentratie van dit oppervlaktewater is op
    0 mg Cl-/l ingesteld. Deze waterlichamen kunnen zowel draineren als infiltreren.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with model grid
    da_name : str, optional
        Name of the dataarray, by default "rws_oppwater"
    cachedir : str, optional
        Directory to cache the data, by default None

    Returns
    -------
    xarray.Dataset
        Dataset with chd and ghb data

    """
    gdf_surface_waters = nlmod.read.rws.get_gdf_surface_water(ds=ds)

    # add north sea to layer model
    ds.update(nlmod.read.rws.discretize_northsea(ds, gdf=gdf_surface_waters, cachedir=cachedir, cachename="northsea"))

    # extrapolate below northsea
    nlmod.dims.extrapolate_ds(ds)

    rws_ds = nlmod.read.rws.discretize_surface_water(
        ds=ds, gdf=gdf_surface_waters, da_basename=da_name, cachedir=cachedir, cachename=da_name
    )

    # update conductance in north sea  (0.1 day resistance, was 10)
    rws_ds[f"{da_name}_cond"] = xr.where(
        ds["northsea"] == 1, rws_ds[f"{da_name}_cond"] * 100, rws_ds[f"{da_name}_cond"]
    )

    # change IJsselmeer+Markermeer peil
    gdf_opp_water = nlmod.read.rws.get_gdf_surface_water(ds)
    gdf_ijsselmeer = gdf_opp_water.loc[gdf_opp_water["OWMNAAM"].isin(["IJsselmeer", "Markermeer"])]
    da_peil = nlmod.dims.gdf_to_da(gdf_ijsselmeer, ds, "peil", agg_method="mean")
    rws_ds[f"{da_name}_stage"] = xr.where(da_peil.isnull(), rws_ds[f"{da_name}_stage"], da_peil)

    return rws_ds


def chd_ghb_from_major_surface_waters(ds, gwf, sea_stage=0.0, da_name="rws_oppwater"):
    """Create chd and ghb packages from major surface waters.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with model grid
    gwf : flopy ModflowGwf
        groundwaterflow object.
    sea_stage : float, optional
        Sea stage, either a float or an argument accepted by by default 0.0
    da_name : str, optional
        Name of the dataarray, by default "rws_oppwater"
    cachedir : str, optional
        Directory to cache the data, by default None

    Returns
    -------
    flopy.modflow.ModflowGwfghb
        GHB package
    flopy.modflow.ModflowGwfchd
        CHD package
    flopy.modflow.ModflowGwfts, optional
        Time series package for sea stage
    """
    ds["sfw_stage"] = xr.where(ds["northsea"] == 0, ds[f"{da_name}_stage"], np.nan)
    ds["sfw_cond"] = xr.where(ds["northsea"] == 0, ds[f"{da_name}_cond"], 0.0)

    ghb = nlmod.gwf.ghb(
        ds,
        gwf,
        bhead="sfw_stage",
        cond="sfw_cond",
        auxiliary=0.0,
        filename=f"{ds.model_name}.ghb_sfw",
        pname="ghb1",
    )

    if isinstance(sea_stage, float):
        ts_sea_val = sea_stage
        chd = nlmod.gwf.chd(
            ds,
            gwf,
            mask="northsea",
            head=ts_sea_val,
            auxiliary=18_000.0,
            filename=f"{ds.model_name}.chd_sea",
            pname="chd",
        )
        ts_sea = None

    else:
        chd = nlmod.gwf.chd(
            ds,
            gwf,
            mask="northsea",
            head="sea_stage",
            auxiliary=18_000.0,
            filename=f"{ds.model_name}.chd_sea",
            pname="chd",
        )
        if chd is None:
            # If all values are outside active grid cells
            ts_sea = None
        else:
            ts_sea = chd.ts.initialize(
                filename="sea_lvl.ts",
                time_series_namerecord="sea_stage",
                interpolation_methodrecord="linear",
                timeseries=sea_stage,
            )
    return ghb, chd, ts_sea
