import os

import flopy
import geopandas as gpd
import matplotlib.pyplot as plt
import nlmod
import numpy as np
import pandas as pd
import pykrige
import xarray as xr
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon


def get_pwn_layer_model(modelgrid, shpdir, plot=False):
    """Reads PWN shapefiles and converts to a layer model Dataset.

    Parameters
    ----------
    modelgrid : flopy.discretization.vertexgrid.VertexGrid
        modelgrid
    shpdir : str
        directory with shapefiles.
    plot : bool, optional
        if True some plots are created during execution. Used for debugging.
        The default is False.

    Returns
    -------
    new_layer_ds : xarray Dataset
        layer model from shapefiles

    """
    # get polygon and points

    # intersect polygon(s) with grid
    ix = flopy.utils.GridIntersect(modelgrid, method="vertex")

    shp_names = ["1A", "1B", "1C", "1D", "q2"]
    default_values = [-3.0, -5.0, -15.0, -20.0, -35.0]
    shp_folders = ["Basis_aquitard", "Dikte_aquitard"]

    data = {}

    for shpfolder in shp_folders:
        arrays = []

        for j, shpnam in enumerate(shp_names):
            if shpfolder == "Dikte_aquitard" and shpnam == "q2":
                shpnam = "2"
            shpnam = shpfolder[0:2].upper() + shpnam
            try:
                poly = gpd.read_file(os.path.join(shpdir, shpfolder, f"{shpnam}_pol.shp"))
            except Exception:
                poly = gpd.read_file(os.path.join(shpdir, shpfolder, f"{shpnam}.shp"))
            try:
                pts = gpd.read_file(os.path.join(shpdir, shpfolder, f"{shpnam}_point.shp"))
            except Exception:
                pts = None

            # build array
            if shpfolder == "Basis_aquitard":
                default = default_values[j]
            elif shpfolder == "Dikte_aquitard":
                default = 0.0
            # arr = default * np.ones_like(modelgrid.botm[0])
            try:
                arr = np.full(modelgrid.shape[1:], default)
            except Exception:
                # in case modelgrid shape is (None, None)
                arr = np.full(len(modelgrid.xcellcenters), default)

            for i, row in poly.iterrows():
                geom = row.geometry.buffer(0.0) if not row.geometry.is_valid else row.geometry
                r = ix.intersect(geom)

                # set zones with value from polygon
                if row["VALUE"] != -999.0:
                    # arr[tuple(zip(*r.cellids))] = row["VALUE"]
                    arr[r.cellids.astype(int)] = row["VALUE"]

                # set interpolated zones
                elif pts is not None:
                    # calculate kriging for with points
                    mask_pts = pts.within(geom)
                    ok = pykrige.ok.OrdinaryKriging(
                        pts.geometry.x.values[mask_pts],
                        pts.geometry.y.values[mask_pts],
                        pts.loc[mask_pts, "VALUE"].values,
                        variogram_model="linear",
                        verbose=False,
                        enable_plotting=False,
                    )
                    # xpts = modelgrid.xcellcenters[tuple(zip(*r.cellids))]
                    # ypts = modelgrid.ycellcenters[tuple(zip(*r.cellids))]
                    try:
                        xpts = modelgrid.xcellcenters[r.cellids.astype(int)]
                        ypts = modelgrid.ycellcenters[r.cellids.astype(int)]
                    except:
                        xpts = np.array(modelgrid.xcellcenters)[r.cellids.astype(int)]
                        ypts = np.array(modelgrid.ycellcenters)[r.cellids.astype(int)]

                    z, _ss = ok.execute("points", xpts, ypts)

                    # arr[tuple(zip(*r.cellids))] = z
                    arr[r.cellids.astype(int)] = z
                else:
                    pass

            arrays.append(arr)

        data[shpfolder] = arrays

    # build layer model based on shapes

    # mv
    mv = gpd.read_file(os.path.join(shpdir, "mvxyz.shp"))

    x = mv.geometry.x.values
    y = mv.geometry.y.values
    z = mv["VALUE"].values

    z_mv = griddata(
        np.vstack([x, y]).T,
        z,
        np.vstack([modelgrid.xcellcenters, modelgrid.ycellcenters]).T,
        method="linear",
    )
    z_mv[np.isnan(z_mv)] = 0.0

    b_1a, b_1b, b_1c, b_1d, b_q2 = data["Basis_aquitard"]
    d_1a, d_1b, d_1c, d_1d, d_q2 = data["Dikte_aquitard"]

    top = np.vstack([z_mv, b_1a, b_1b, b_1c, b_q2])

    bot = np.vstack([b_1a + d_1a, b_1b + d_1b, b_1c + d_1c, b_1d + d_1d, b_q2 + d_q2])

    # check tops
    minthick = 0.01

    for ilay in range(1, 5):
        top[ilay] = np.where(top[ilay] < bot[ilay - 1], top[ilay], bot[ilay - 1] - minthick)

    # check bots
    for ilay in range(1, 5):
        bot[ilay] = np.where(bot[ilay] < top[ilay], bot[ilay], top[ilay] - minthick)

    # add aquitards
    top2 = np.vstack([
        top[0],
        bot[0],
        top[1],
        bot[1],
        top[2],
        bot[2],
        top[3],
        bot[3],
        top[4],
    ])
    bot2 = np.vstack([
        bot[0],
        top[1],
        bot[1],
        top[2],
        bot[2],
        top[3],
        bot[3],
        top[4],
        bot[4],
    ])

    # create data arrays
    x = xr.DataArray(modelgrid.xcellcenters, dims=("icell2d"))
    y = xr.DataArray(modelgrid.ycellcenters, dims=("icell2d"))

    t_da = xr.DataArray(
        top2,
        coords={"layer": range(top2.shape[0]), "x": x, "y": y},
        dims=("layer", "icell2d"),
    )
    b_da = xr.DataArray(
        bot2,
        coords={"layer": range(bot2.shape[0]), "x": x, "y": y},
        dims=("layer", "icell2d"),
    )

    thickness = t_da - b_da

    if plot:
        cmap = plt.get_cmap("viridis")
        vmin = b_da.min()
        vmax = t_da.max()

        fig, ax = plt.subplots(2, 5, figsize=(10, 10))

        for i in range(4):
            iax0, iax1 = ax[:, i]

            iax0.set_aspect("equal")
            iax1.set_aspect("equal")

            mv2 = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=iax0)
            qm = mv2.plot_array(t_da.sel(layer=i), cmap=cmap, vmin=vmin, vmax=vmax)
            mv2.plot_grid(color="k", lw=0.25)
            iax0.set_title("top")
            iax0.axis(modelgrid.extent)

            mv3 = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=iax1)
            qm = mv3.plot_array(b_da.sel(layer=i), cmap=cmap, vmin=vmin, vmax=vmax)
            mv3.plot_grid(color="k", lw=0.25)
            iax1.set_title("bot")
            iax1.axis(modelgrid.extent)

            # mv.plot(ax=ax, column="VALUE", cmap=cmap, vmin=vmin, vmax=vmax, markersize=5)

        fig.colorbar(qm, ax=ax, shrink=1.0)

        for ilay in range(thickness.shape[0]):
            cmap = plt.get_cmap("RdBu")
            vmax = thickness[ilay].max()
            vmin = 0.0

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect("equal")

            mv2 = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax)
            qm = mv2.plot_array(thickness.sel(layer=ilay), cmap=cmap, vmin=vmin, vmax=vmax)
            mv2.plot_grid(color="k", lw=0.25)
            ax.set_title(f"Layer {ilay}")
            ax.axis(modelgrid.extent)

            fig.colorbar(qm, ax=ax, shrink=1.0)
            fig.savefig(f"thick_{ilay}.png", bbox_inches="tight", dpi=150)

    # get
    clist = []
    cdefaults = [1.0, 100.0, 100.0, 1.0, 1.0]

    for j, letter in enumerate(["1A", "1B", "1C", "1D", "2"]):
        poly = gpd.read_file(os.path.join(shpdir, "..", "Bodemparams", f"C{letter}.shp"))

        arr = cdefaults[j] * np.ones_like(modelgrid.xcellcenters)

        for i, row in poly.iterrows():
            r = ix.intersects(row.geometry)
            # arr[tuple(zip(*r.cellids))] = row["VALUE"]
            arr[r.cellids.astype(int)] = row["VALUE"]

        clist.append(arr)

    if plot:
        parr = clist[0]

        cmap = plt.get_cmap("viridis")
        vmin = 0.0
        vmax = parr.max()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal")
        mv2 = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax)
        qm = mv2.plot_array(parr, cmap=cmap, vmin=vmin, vmax=vmax)
        mv2.plot_grid(color="k", lw=0.25)
        fig.colorbar(qm, ax=ax, shrink=1.0)
        fig.tight_layout()

    f_anisotropy = 0.25

    kh = xr.zeros_like(t_da)
    kh[0] = 7.0
    kh[1] = thickness[1] / clist[0] / f_anisotropy
    kh[2] = 7.0
    kh[3] = thickness[3] / clist[1] / f_anisotropy
    kh[4] = 12.0
    kh[5] = thickness[5] / clist[2] / f_anisotropy
    kh[6] = 15.0
    kh[7] = thickness[7] / clist[3] / f_anisotropy
    kh[8] = 20.0

    kv = xr.zeros_like(t_da)
    kv[0] = kh[0] * f_anisotropy
    kv[1] = thickness[1] / clist[0]
    kv[2] = kh[2] * f_anisotropy
    kv[3] = thickness[3] / clist[1]
    kv[4] = kh[4] * f_anisotropy
    kv[5] = thickness[5] / clist[2]
    kv[6] = kh[6] * f_anisotropy
    kv[7] = thickness[7] / clist[3]
    kv[8] = kh[8] * f_anisotropy

    # create new dataset
    new_layer_ds = xr.Dataset(data_vars={"top": t_da, "bot": b_da, "kh": kh, "kv": kv})

    return new_layer_ds.assign_coords(coords={"layer": [f"hlc_{i}" for i in range(new_layer_ds.dims["layer"])]})



def update_layermodel(layermodel_orig, layermodel_update):
    """Updates the REGIS Holocene layer with information from a PWN layer
    dataset.


    Parameters
    ----------
    layermodel_orig : xarray Dataset
        original layer model
    layermodel_update : xarray Dataset
        layer model used to update original layer model

    Returns
    -------
    regis_pwn_ds : xarray Dataset
        updated layer model

    """
    float_correction = 1e-5  # for dealing w floating point differences

    # create new empty ds
    regis_pwn_ds = xr.Dataset()

    # find holoceen (remove all layers above Holoceen)
    layer_no = np.where((layermodel_orig.layer == "HLc").values)[0][0]
    new_layers = np.append(
        layermodel_update.layer.data,
        layermodel_orig.layer.data[layer_no + 1 :].astype("<U8"),
    ).astype("O")

    top_new = xr.DataArray(
        np.ones((len(new_layers), len(layermodel_update.x))) * np.nan,
        dims=("layer", "icell2d"),
        coords={
            "x": layermodel_update.x,
            "y": layermodel_update.y,
            "layer": new_layers,
        },
    )
    bot_new = xr.DataArray(
        np.ones((len(new_layers), len(layermodel_update.x))) * np.nan,
        dims=("layer", "icell2d"),
        coords={
            "x": layermodel_update.x,
            "y": layermodel_update.y,
            "layer": new_layers,
        },
    )
    kh_new = xr.DataArray(
        np.ones((len(new_layers), len(layermodel_update.x))) * np.nan,
        dims=("layer", "icell2d"),
        coords={
            "x": layermodel_update.x,
            "y": layermodel_update.y,
            "layer": new_layers,
        },
    )
    kv_new = xr.DataArray(
        np.ones((len(new_layers), len(layermodel_update.x))) * np.nan,
        dims=("layer", "icell2d"),
        coords={
            "x": layermodel_update.x,
            "y": layermodel_update.y,
            "layer": new_layers,
        },
    )

    # haal overlap tussen geotop en regis weg

    for lay in range(layermodel_update.dims["layer"]):
        # Alle nieuwe cellen die onder de onderkant van het holoceen liggen worden inactief
        mask1 = layermodel_update["top"][lay] <= (layermodel_orig["bot"][layer_no] - float_correction)
        layermodel_update["top"][lay] = xr.where(mask1, np.nan, layermodel_update["top"][lay])
        layermodel_update["bot"][lay] = xr.where(mask1, np.nan, layermodel_update["bot"][lay])
        layermodel_update["kh"][lay] = xr.where(mask1, np.nan, layermodel_update["kh"][lay])
        layermodel_update["kv"][lay] = xr.where(mask1, np.nan, layermodel_update["kv"][lay])

        # Alle geotop cellen waarvan de bodem onder de onderkant van het holoceen ligt, krijgen als bodem de onderkant van het holoceen
        mask2 = layermodel_update["bot"][lay] < layermodel_orig["bot"][layer_no]
        layermodel_update["bot"][lay] = xr.where(
            mask2 * (~mask1),
            layermodel_orig["bot"][layer_no],
            layermodel_update["bot"][lay],
        )

        # # Alle geotop cellen die boven de bovenkant van het holoceen liggen worden inactief
        # mask3 = layermodel_update['bot'][lay] >= (
        #     layermodel_orig['top'][layer_no] - float_correction)
        # layermodel_update['top'][lay] = xr.where(
        #     mask3, np.nan, layermodel_update['top'][lay])
        # layermodel_update['bot'][lay] = xr.where(
        #     mask3, np.nan, layermodel_update['bot'][lay])
        # layermodel_update['kh'][lay] = xr.where(mask3, np.nan, layermodel_update['kh'][lay])
        # layermodel_update['kv'][lay] = xr.where(mask3, np.nan, layermodel_update['kv'][lay])

        # # Alle geotop cellen waarvan de top boven de top van het holoceen ligt, krijgen als top het holoceen van regis
        # mask4 = layermodel_update['top'][lay] >= layermodel_orig['top'][layer_no]
        # layermodel_update['top'][lay] = xr.where(
        #     mask4 * (~mask3), layermodel_orig['top'][layer_no], layermodel_update['top'][lay])

        # overal waar holoceen inactief is, wordt geotop ook inactief
        mask5 = layermodel_orig["bot"][layer_no].isnull()
        layermodel_update["top"][lay] = xr.where(mask5, np.nan, layermodel_update["top"][lay])
        layermodel_update["bot"][lay] = xr.where(mask5, np.nan, layermodel_update["bot"][lay])
        layermodel_update["kh"][lay] = xr.where(mask5, np.nan, layermodel_update["kh"][lay])
        layermodel_update["kv"][lay] = xr.where(mask5, np.nan, layermodel_update["kv"][lay])

        if (mask2 * (~mask1)).sum() > 0:
            pass

    top_new[: len(layermodel_update.layer), :] = layermodel_update["top"].data
    top_new[len(layermodel_update.layer) :, :] = layermodel_orig["top"].data[layer_no + 1 :]

    bot_new[: len(layermodel_update.layer), :] = layermodel_update["bot"].data
    bot_new[len(layermodel_update.layer) :, :] = layermodel_orig["bot"].data[layer_no + 1 :]

    kh_new[: len(layermodel_update.layer), :] = layermodel_update["kh"].data
    kh_new[len(layermodel_update.layer) :, :] = layermodel_orig["kh"].data[layer_no + 1 :]

    kv_new[: len(layermodel_update.layer), :] = layermodel_update["kv"].data
    kv_new[len(layermodel_update.layer) :, :] = layermodel_orig["kv"].data[layer_no + 1 :]

    regis_pwn_ds["top"] = top_new
    regis_pwn_ds["bot"] = bot_new
    regis_pwn_ds["kh"] = kh_new
    regis_pwn_ds["kv"] = kv_new

    for dsvar in ["icvert", "xv", "yv", "x", "y"]:
        regis_pwn_ds[dsvar] = layermodel_orig[dsvar]

    for key, item in layermodel_orig.attrs.items():
        regis_pwn_ds.attrs.update({key: item})

    return regis_pwn_ds


def get_surface_water_bgt(oppwaterg, fname_bgt, gwf, cachedir="."):
    """Read surface water data from a bgt geojson file.

    Parameters
    ----------
    oppwaterg : geopandas.GeoDataFrame
        GeoDataFrame with surface water geometries for each cell.
    fname_bgt : str
        file path of geojson with bgt data.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        Groundwater flo model.
    cachedir : str, optional
        This directory is used to cache the file and to read the
        file from cache. The default is ".".

    Raises
    ------
    FileNotFoundError
        Error raised if there is no bgt file available.

    Returns
    -------
    bgtg : geopandas.GeoDataFrame
        GeoDataFrame with surface water geometries for each cell. combination
        of bgtg and other surface water

    """
    # put drains in most bgt-shapes at minimum surface level at watercourses
    fname_bgtg = os.path.join(cachedir, "bgt_waterdeel_grid.geojson")
    if os.path.isfile(fname_bgtg):
        # read bgtg from cache
        bgtg = gpd.read_file(fname_bgtg).set_index("index")
        bgtg = bgtg.set_crs(epsg=28992, allow_override=True)
    else:
        if os.path.isfile(fname_bgt):
            # read bgt from cache
            bgt = gpd.read_file(fname_bgt).set_index("index")
            bgt = bgt.set_crs(epsg=28992, allow_override=True)
        else:
            msg = "No stored surface water data!" f" {fname_bgt}"
            raise FileNotFoundError(msg)

        # set the index to the unique column of lokaalID
        bgt = bgt.set_index("lokaalID")

        # remove the North Sea
        for index in [
            "L0002.7131a9bd31ee1bbfe05332a1e90a6c7f",
            "L0002.7131a9bd2eaa1bbfe05332a1e90a6c7f",
            "L0002.7131a9bd30111bbfe05332a1e90a6c7f",
        ]:
            if index in bgt.index:
                bgt = bgt.drop(index)

        # cut by the grid
        bgtg = nlmod.mdims.gdf2grid(bgt, gwf, method="vertex", desc="Intersecting bgt with grid")
        bgtg.reset_index().to_file(fname_bgtg, driver="GeoJSON")

    # replace the shapes in bgtg by the ones from pwn
    opw2bgt = {}
    opw2bgt["vijver 1, 2 en 3"] = [
        "W0651.865c8d7a2be84961826f052b681a7f63",
        "W0651.be0fc7d4ecbd4459932a17fc38797f07",
        "W0651.4422eb7cde924cfc9392959423afbcb5",
    ]
    opw2bgt["vijver 4"] = "W0651.729e81ab73374e9fa70abf139251f923"
    opw2bgt["libellenpoel"] = "G0373.9f930048a0ee488fabf6e822fdac4994"
    opw2bgt["boringkanaal B"] = [
        "G0373.0ca7da473d004ccc84317497348b62f7",
        "G0373.44d141555adc43498f67443c267694c6",
        "G0373.dee82e56db11492eb475a822514bcec5",
        "G0373.ee58e91c92b44316b7ab734d485a864d",
        "G0373.3905b70fd5d24d888932edf1489f33d2",
        "W0651.91036ab15e7d4dceb81dca0baf6760a3",
    ]
    opw2bgt["Guurtjeslaan"] = [
        "G0373.3b8d111efde54846bf2f01152e176e74",
        "G0373.bc1e0539b4ad4669912a452acae8a003",
    ]
    opw2bgt["boringkanaal C"] = [
        "G0373.e498c5cf32684b95a260151eb58e1fd1",
        "G0373.dcecefc3ff6d4ddcb6951bf7c300f556",
        "G0373.29b5a08aae3e48d6a2a39ae5a3a1faa3",
    ]

    for name in opw2bgt:
        bgtg = pd.concat((bgtg, oppwaterg.loc[[name]]))
        bgtg = bgtg.drop(opw2bgt[name])

    # remove other surface water in dunes nearby pumping station
    bgtg = bgtg.drop([
        "G0373.e679cb9e868c4fdab8b4e7ef9013834f",
        "W0651.f741c8d00fc64c7c98c8e3f11cbff325",
        "W0651.476d8e16fd314ca6ba91f6904c07698f",
        "W0651.c8ab0748c2fc436fa596352e2c3515a2",
    ])

    # and south of boringkanaal C
    bgtg = bgtg.drop([
        "W0651.5e21acfa055d4915af629bf77c165c35",
        "W0651.8c29fa2fbbe8437494bd827e7302655c",
        "W0651.923a762194ca47c6b686d2443231d407",
        "W0651.6df2a79d1e184cdd88bf76ce64b76422",
        "W0651.f2d8aa8fa45949f2ad6ba14c70b30a54",
        "W0651.a99152df739947a2a1a5411157328287",
        "W0651.45b659b8d5254f4eb2a4dc440cd464b2",
    ])
    # some ponds near the sea
    return bgtg.drop([
        "G0373.88944c1bb9e14bdfb35dc46d2c6eff70",
        "W0651.15d7bba7d7b9462d9585095d7407cd80",
        "W0651.e74ba578b2c14f349e72f8d301500dde",
        "G0373.2045f4999fe6446eb42c4705698e74ab",
        "W0651.e16e974e96484d07b55ec2e0ebf5459f",
        "W0651.2d134bd13ad94d8bb6977140fe8bf4ab",
        "G0373.ab88b1eb82da4831be36b6a785ff7ec4",
    ])



def line2hfb(gdf, gwf, prevent_rings=True, plot=False):
    """Obtain the cells with a horizontal flow barrier between them from a
    geodataframe with line elements.


    Parameters
    ----------
    gdf : gpd.GeoDataframe
        geodataframe with line elements.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        grondwater flow model.
    prevent_rings : bool, optional
        DESCRIPTION. The default is True.
    plot : bool, optional
        If True create a simple plot of the grid cells and shapefile. For a
        more complex plot you can use plot_hfb. The default is False.

    Returns
    -------
    cellids : 2d list of ints
        a list with pairs of cells that have a hfb between them.

    """
    # for the idea, sea:
    # https://gis.stackexchange.com/questions/188755/how-to-snap-a-road-network-to-a-hexagonal-grid-in-qgis

    gdfg = nlmod.mdims.gdf2grid(gdf, gwf)

    cell2d = pd.DataFrame(gwf.disv.cell2d.array).set_index("icell2d")
    vertices = pd.DataFrame(gwf.disv.vertices.array).set_index("iv")

    # for every cell determine which cell-edge could form the line
    # by testing for an intersection with a triangle to the cell-center
    icvert = cell2d.loc[:, cell2d.columns.str.startswith("icvert")].values

    hfb_seg = []
    for index in gdfg.index.unique():
        # Get the nearest hexagon sides where routes cross
        for icell2d in gdfg.loc[index, "cellid"]:
            for i in range(cell2d.at[icell2d, "ncvert"] - 1):
                iv1 = icvert[icell2d, i]
                iv2 = icvert[icell2d, i + 1]
                # make sure vert1 is lower than vert2
                if iv1 > iv2:
                    iv1, iv2 = iv2, iv1
                coords = [
                    (cell2d.at[icell2d, "xc"], cell2d.at[icell2d, "yc"]),
                    (vertices.at[iv1, "xv"], vertices.at[iv1, "yv"]),
                    (vertices.at[iv2, "xv"], vertices.at[iv2, "yv"]),
                ]
                triangle = Polygon(coords)
                if triangle.intersects(gdf.loc[index, "geometry"]):
                    hfb_seg.append((icell2d, iv1, iv2))
    hfb_seg = np.array(hfb_seg)

    if prevent_rings:
        # find out if there are cells with segments on each side
        # remove the segments whose centroid is farthest from the line
        for icell2d in np.unique(hfb_seg[:, 0]):
            mask = hfb_seg[:, 0] == icell2d
            if mask.sum() >= cell2d.at[icell2d, "ncvert"] - 1:
                segs = hfb_seg[mask]
                dist = []
                for seg in segs:
                    p = Point(
                        vertices.loc[seg[1:3], "xv"].mean(),
                        vertices.loc[seg[1:3], "yv"].mean(),
                    )
                    dist.append(gdf.distance(p).min())
                iv1, iv2 = segs[np.argmax(dist), [1, 2]]
                mask = (hfb_seg[:, 1] == iv1) & (hfb_seg[:, 2] == iv2)
                hfb_seg = hfb_seg[~mask]

    # get unique segments
    hfb_seg = np.unique(hfb_seg[:, 1:], axis=0)

    # Get rid of disconnected (or 'open') segments
    # Let's remove disconnected/open segments
    iv = np.unique(hfb_seg)
    segments_per_iv = pd.Series([np.sum(hfb_seg == x) for x in iv], index=iv)
    mask = np.full(hfb_seg.shape[0], True)
    for i, segment in enumerate(hfb_seg):
        # one vertex is not connected and the other one at least to two other segments
        if (segments_per_iv[segment[0]] == 1 and segments_per_iv[segment[1]] >= 3) or (
            segments_per_iv[segment[1]] == 1 and segments_per_iv[segment[0]] >= 3
        ):
            mask[i] = False
    hfb_seg = hfb_seg[mask]

    if plot:
        # test by plotting
        ax = gdfg.plot()
        for i, seg in enumerate(hfb_seg):
            x = [vertices.at[seg[0], "xv"], vertices.at[seg[1], "xv"]]
            y = [vertices.at[seg[0], "yv"], vertices.at[seg[1], "yv"]]
            ax.plot(x, y)

    # find out between which cellid's these segments are
    segments = []
    for icell2d in cell2d.index:
        for i in range(cell2d.at[icell2d, "ncvert"] - 1):
            iv1 = icvert[icell2d, i]
            iv2 = icvert[icell2d, i + 1]
            # make sure vert1 is lower than vert2
            if iv1 > iv2:
                iv1, iv2 = iv2, iv1
            segments.append((icell2d, (iv1, iv2)))
    segments = pd.DataFrame(segments, columns=["icell2d", "verts"])
    segments = segments.set_index(["verts"])

    cellids = []
    for seg in hfb_seg:
        cellids.append(list(segments.loc[[tuple(seg)]].values[:, 0]))
    return cellids


def plot_hfb(cellids, gwf, ax=None):
    """Plots a horizontal flow barrier.

    Parameters
    ----------
    cellids : list of lists of integers or flopy.mf6.ModflowGwfhfb
        list with the ids of adjacent cells that should get a horizontal
        flow barrier, hfb is the output of line2hfb.
    gwf : flopy groundwater flow model
        DESCRIPTION.
    ax : matplotlib axes


    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    if ax is None:
        _fig, ax = plt.subplots()

    if isinstance(cellids, flopy.mf6.ModflowGwfhfb):
        spd = cellids.stress_period_data.data[0]
        cellids = [[line[0][1], line[1][1]] for line in spd]

    for line in cellids:
        pc1 = Polygon(gwf.modelgrid.get_cell_vertices(line[0]))
        pc2 = Polygon(gwf.modelgrid.get_cell_vertices(line[1]))
        x, y = pc1.intersection(pc2).xy
        ax.plot(x, y, color="red")

    return ax
