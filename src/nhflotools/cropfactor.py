import geopandas as gpd
import nlmod
import numpy as np
import xarray as xr


def apply_cropfactor(ds=None, cropfactor_dir=None, gwf=None, Adjust_for_groundwaterdepth=False, Groundwaterdepth=None):
    """
    Apply a cropfactor to recharge in de ds.

    For this function to work, the precipitation and evaporation should both be seperate in the model ds.

    The function then applies a factor to only the evaporation, according to the factor that was supplied in the shapefile under the attribute "VALUE"

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to merge the PWN layer model with. This dataset should contain the variables 'precipitation' and 'evaporation'.
    cropfactor_dir : str
        The path to the shapefile with the cropfactors, stored in the attribute 'VALUE'.
    gwf : flopy ModflowGwf
        groundwaterflow object
    Adjust_for_groundwaterdepth : boolean
        If True there wil also be ajusted for locations where the groundwatertable is deep (less evaporation) according tot the old TRIWACO PWN model method.
        This method applies 100% evaporation to shallow groundwatertables (less than .5 meter deep) and 75% evaporation to deep groundwatertables (deeper than 2,5 meter)
        In between 0.5 and 2.5 the evaporation will be linearly interpolsated between 100 and 75%. Default is False
    Groundwaterdepth : str
        The path to the shapefile with the average depth of the groundwater, stored in the attribute 'diepte_gws'.

    Returns
    -------
    ds : xarray Dataset
        The dataset with added 'recharge', 'cropfactor' and 'avg_depth_groundwater' parameter.
    rch : flopy ModflowGwf
        recharge object.
    """
    # Reading the cropfactor shapefile and interpolating to the grid
    cropfactor_shp = gpd.read_file(cropfactor_dir)
    grid_cropfactor = nlmod.grid.gdf_to_grid(cropfactor_shp.loc[:, ["VALUE", "geometry"]], gwf)
    fields_methods2 = {"VALUE": "area_weighted"}
    celldata_cropfactor = nlmod.grid.aggregate_vector_per_cell(grid_cropfactor, fields_methods=fields_methods2)
    celldata_cropfactor["x"] = ds.sel(icell2d=celldata_cropfactor.index)["x"]
    celldata_cropfactor["y"] = ds.sel(icell2d=celldata_cropfactor.index)["y"]
    cropfactor = xr.full_like(ds.top, np.nan)
    cropfactor.loc[dict(icell2d=celldata_cropfactor.index)] = celldata_cropfactor["VALUE"].values
    ds["cropfactor"] = cropfactor

    if Adjust_for_groundwaterdepth:
        # Reading the average depth of the groundwater shapefile and interpolating to the grid
        avg_depth_groundwater_shp = gpd.read_file(avg_depth_groundwater_shp_dir)
        grid_avg_depth_groundwater = nlmod.grid.gdf_to_grid(
            avg_depth_groundwater_shp.loc[:, ["diepte_gws", "geometry"]], gwf
        )
        fields_methods2 = {"diepte_gws": "area_weighted"}
        celldata_avg_depth_groundwater = nlmod.grid.aggregate_vector_per_cell(
            grid_avg_depth_groundwater, fields_methods=fields_methods2
        )
        celldata_avg_depth_groundwater["x"] = ds.sel(icell2d=celldata_avg_depth_groundwater.index)["x"]
        celldata_avg_depth_groundwater["y"] = ds.sel(icell2d=celldata_avg_depth_groundwater.index)["y"]
        avg_depth_groundwater = xr.full_like(ds.top, np.nan)
        avg_depth_groundwater.loc[dict(icell2d=celldata_avg_depth_groundwater.index)] = celldata_avg_depth_groundwater[
            "diepte_gws"
        ].values
        ds["avg_depth_groundwater"] = avg_depth_groundwater

        # Start loop to prepare recharge data for every timestep
        for dag in range(len(ds["precipitation"]) - 1):
            # ds['avg_depth_groundwater'] = average groundwater depth per modelcell relative to ground level in meter.

            Recharge = ds["precipitation"][dag]
            Recharge = xr.where(
                (ds["avg_depth_groundwater"] < 0.5), Recharge - (ds["cropfactor"] * ds["evaporation"][dag]), Recharge
            )
            Recharge = xr.where(
                (ds["avg_depth_groundwater"] > 0.5) & (ds["avg_depth_groundwater"] < 2.5),
                Recharge
                - (1 - ((ds["avg_depth_groundwater"] - 0.5) / 8)) * (ds["cropfactor"] * ds["evaporation"][dag]),
                Recharge,
            )
            Recharge = xr.where(
                (ds["avg_depth_groundwater"] > 2.5),
                Recharge - (0.75 * (ds["cropfactor"] * ds["evaporation"][dag])),
                Recharge,
            )
            to_MODFLOW = Recharge
            ds["recharge"][dag] = to_MODFLOW
    else:
        for dag in range(len(ds["precipitation"]) - 1):
            Recharge = ds["precipitation"][dag]
            Recharge = Recharge - (ds["cropfactor"] * ds["evaporation"][dag])
            to_MODFLOW = Recharge
            ds["recharge"][dag] = to_MODFLOW

    # fix recharge for steady state warmup
    ds["recharge"][0] = 0.0012

    # Create RCH package
    rch = nlmod.gwf.rch(ds, gwf, mask=ds["northsea"] == 0)
    return ds, rch
