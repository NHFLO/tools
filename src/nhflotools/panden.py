import os

import flopy
import geopandas as gpd
import nlmod
import numpy as np


def get_oppervlakte_pwn_shapes(data_path_panden):
    """Get oppervlakte shapes for PWN area.

    Serves as input to

    Parameters
    ----------
    data_path_panden : str
        Path to the panden data.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with oppervlakte shapes.
    """
    panden_shp = gpd.read_file(os.path.join(data_path_panden, "Panden_ICAS_IKIEF.shp"))
    panden_shp["geometry"] = panden_shp.make_valid()  # fix geometry
    panden_shp["c"] = 1.0  # bodemweerstand
    panden_shp["stage"] = np.nan
    panden_shp.loc[panden_shp.Naam.str.contains("ICAS"), "stage"] = 2.8  # mNAP
    panden_shp.loc[panden_shp.Naam.str.contains("IKIEF"), "stage"] = 5.8  # mNAP
    panden_shp["rbot"] = panden_shp["stage"] - 2.0
    return panden_shp


def riv_from_oppervlakte_pwn(ds, gwf, data_path_panden):
    """Create RIV package from oppervlakte shapes for PWN area.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    gwf : flopy ModflowGwf
        groundwaterflow object.
    data_path_panden : str
        Path to the panden data.

    Returns
    -------
    flopy.modflow.ModflowGwfriv
        RIV package.
    """
    panden_shp = get_oppervlakte_pwn_shapes(data_path_panden=data_path_panden)
    rivdata = nlmod.grid.gdf_to_grid(panden_shp, gwf)
    rivdata["cond"] = rivdata["area"] / rivdata["c"]
    agg = nlmod.grid.aggregate_vector_per_cell(
        rivdata,
        fields_methods={
            "stage": "area_weighted",
            "c": "sum",
            "rbot": "min",
            "Naam": "first",
        },
    )
    agg["aux"] = 0.0
    agg.rename(columns={"Naam": "boundname", "c": "cond"}, inplace=True)

    riv_spd = nlmod.gwf.build_spd(agg, "RIV", ds, layer_method="lay_of_rbot")

    riv = flopy.mf6.ModflowGwfriv(
        gwf,
        auxiliary="CONCENTRATION",
        boundnames=True,
        stress_period_data={0: riv_spd},
        save_flows=True,
    )
    return riv
