import logging
import os

import flopy
import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import triwaco
import xarray as xr
from shapely.geometry import LineString
from shapely.geometry import Polygon

from nhflotools.pwnlayers.utils import _read_bodemparams
from nhflotools.pwnlayers.utils import _read_fluzo
from nhflotools.pwnlayers.utils import _read_kd_of_aquifers
from nhflotools.pwnlayers.utils import _read_kv_area
from nhflotools.pwnlayers.utils import _read_layer_kh
from nhflotools.pwnlayers.utils import _read_mask_of_aquifers
from nhflotools.pwnlayers.utils import _read_thickness_of_aquitards
from nhflotools.pwnlayers.utils import _read_top_of_aquitards
from nhflotools.pwnlayers.utils import _read_topsysteem
from nhflotools.pwnlayers.utils import _read_zout
from nhflotools.pwnlayers.utils import add_point_data_to_riv_gdf
from nhflotools.pwnlayers.utils import add_regis_to_bottom_of_pwn
from nhflotools.pwnlayers.utils import add_top_system_cell
from nhflotools.pwnlayers.utils import combine_layer_models_regis_pwn
from nhflotools.pwnlayers.utils import compare_layer_models_top_view
from nhflotools.pwnlayers.utils import fill_gaps
from nhflotools.pwnlayers.utils import geodataframe_to_grid
from nhflotools.pwnlayers.utils import inpolygon
from nhflotools.pwnlayers.utils import read_polygon_shape
from nhflotools.pwnlayers.utils import shp2grid2

logger = logging.getLogger(__name__)


def triwaco_to_flopy(
    ds=None,
    nlay=8,
    topsystem=4,
    model_ws=".",
    modelname="modflowtest",
    dx=500.0,
    extent=[96000.0, 109000.0, 497000.0, 515000.0],
    dx2=100.0,
    extent2=[101000.0, 103500.0, 502000.0, 508000.0],
    drainage_systems=[2, 3],
    figpath=None,
    quasi3d=False,
    density=True,
    datadir=None,
):
    """Create a flopy model from a Triwaco dataset.


    Parameters
    ----------
    ds : TYPE, optional
        DESCRIPTION. The default is None.
    nlay : TYPE, optional
        DESCRIPTION. The default is 8.
    topsystem : TYPE, optional
        DESCRIPTION. The default is 4.
    model_ws : TYPE, optional
        DESCRIPTION. The default is '.'.
    modelname : TYPE, optional
        DESCRIPTION. The default is 'modflowtest'.
    dx : TYPE, optional
        DESCRIPTION. The default is 500..
    extent : TYPE, optional
        DESCRIPTION. The default is [96000., 109000., 497000., 515000.].
    dx2 : TYPE, optional
        DESCRIPTION. The default is 100..
    extent2 : TYPE, optional
        DESCRIPTION. The default is [101000., 103500., 502000., 508000.].
    drainage_systems : TYPE, optional
        DESCRIPTION. The default is [2, 3].
    figpath : TYPE, optional
        DESCRIPTION. The default is None.
    quasi3d : TYPE, optional
        DESCRIPTION. The default is False.
    density : TYPE, optional
        DESCRIPTION. The default is True.
    datadir : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    m : TYPE
        DESCRIPTION.

    """
    # make a flopy model
    m = flopy.modflow.Modflow(model_ws=model_ws, modelname=modelname)

    # dis package
    if dx2:
        delr = np.diff(
            np.hstack(
                (
                    np.arange(extent[0], extent2[0], dx),
                    np.arange(extent2[0], extent2[1], dx2),
                    np.arange(extent2[1], extent[1] + dx, dx),
                )
            )
        )
        delc = np.abs(
            np.diff(
                np.hstack(
                    (
                        np.arange(extent[3], extent2[3], -dx),
                        np.arange(extent2[3], extent2[2], -dx2),
                        np.arange(extent2[2], extent[2] - dx, -dx),
                    )
                )
            )
        )
        nrow = len(delc)
        ncol = len(delr)
    else:
        nrow = int((extent[3] - extent[2]) / dx)
        ncol = int((extent[1] - extent[0]) / dx)
        delr = np.ones(ncol) * dx
        delc = np.ones(nrow) * dx
    if quasi3d:
        laycbd = [1] * nlay
        laycbd[-1] = 0
    else:
        nlay = nlay + (nlay - 1)
        laycbd = [0] * nlay
    nbotm = nlay + np.array(laycbd).sum()
    botm = np.full((nbotm, nrow, ncol), np.NaN)
    if ds is None:
        # with fictional top and botm
        top = 0
        for lay in range(nbotm):
            botm[lay] = -lay
    else:
        # calculate top and botm
        # top = data['RL1']
        # use the peil of the topsystem
        top = ds["TOP"].values

        if quasi3d:
            # RLi Elevation of top of aquifer i
            for lay in range(1, nlay):
                botm[lay * 2 - 1] = ds["RL{}".format(lay + 1)]
            # THi Elevation of base of aquifer i
            for lay in range(nlay):
                botm[lay * 2] = ds["TH{}".format(lay + 1)]
        else:
            for lay in range(nlay):
                if np.mod(lay, 2) == 1:
                    # RLi Elevation of top of aquifer i
                    iaq = int((lay + 1) / 2) + 1
                    botm[lay] = ds["RL{}".format(iaq)]
                else:
                    # THi Elevation of base of aquifer i
                    iaq = int(lay / 2) + 1
                    botm[lay] = ds["TH{}".format(iaq)]

        # make sure the first layer is at least 1 cm thick
        # mask = (top-botm[0]) < 0.01
        # top[mask] = botm[0][mask] + 0.01

        for lay in range(nlay):
            # first fill any gaps
            xcentergrid, ycentergrid = np.meshgrid(
                delr.cumsum() - delr / 2, delc.cumsum() - delc / 2
            )
            botm[lay] = fill_gaps(botm[lay], xcentergrid, ycentergrid)
            # then make sure each cell is at least 1 cm thick
            if lay == 0:
                top_layer = top
            else:
                top_layer = botm[lay - 1]
            mask = (top_layer - botm[lay]) < 0.01
            botm[lay][mask] = top_layer[mask] - 0.01

    today = pd.to_datetime("today")
    flopy.modflow.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        laycbd=laycbd,
        top=top,
        botm=botm,
        xul=extent[0],
        yul=extent[3],
        proj4_str="EPSG:28992",
        start_datetime=today.strftime("%m/%d/%Y"),
    )
    if ds is None:
        flopy.modflow.ModflowBas(m)
        return m

    # read information abound the triwaco grid
    if datadir is None:
        fname = "data/extracted/BasisbestandenNHDZmodelPWN_CvG_20181022/" "grid.teo"
    else:
        fname = os.path.join(
            datadir, "Grid", "grid.teo"
        )
    grid = triwaco.read_teo(fname)

    # lpf package
    laycbd = m.dis.laycbd.array

    thickness = np.ones_like(m.dis.botm.array)
    for lay, botlay in enumerate(m.dis.botm.array):
        if lay == 0:
            thickness[lay] = m.dis.top.array - botlay
        else:
            thickness[lay] = m.dis.botm.array[lay - 1] - botlay

    # thickness[thickness<0.01]=0.01
    laytyp = [0] * nlay
    # laytyp[0]=1 # first layer is convertable
    hk = np.full((nlay, nrow, ncol), np.NaN)
    for lay in range(m.nlay):
        if quasi3d or (np.mod(lay, 2) == 0):
            # layer is an aquifer
            if quasi3d:
                iaq = lay + 1
            else:
                iaq = int(lay / 2) + 1
            px = "PX{}".format(iaq)
            if px in ds:
                hk[lay] = ds[px]
            else:
                ind = lay + laycbd[:lay].sum()

                hk[lay] = getattr(ds, "TX{}".format(iaq)) / thickness[ind]
    if quasi3d:
        vkcb = np.full((nlay, nrow, ncol), np.NaN)
        for lay in range(m.nlay - 1):
            if laycbd[lay]:
                ind = lay + laycbd[: lay + 1].sum()
                vkcb[lay] = thickness[ind] / ds["CL{}".format(lay + 1)]
        vkcb[np.isnan(vkcb)] = 100
    else:
        # set vkcb to the default value of flopy
        vkcb = 0.0
        # also aquitards have a hydraulic conductivity
        for lay in range(1, m.nlay, 2):
            iaq = int((lay - 1) / 2) + 1
            hk[lay] = thickness[lay] / ds["CL{}".format(iaq)]
    hk[np.isnan(hk)] = 100

    vka = hk
    flopy.modflow.ModflowLpf(m, laytyp=laytyp, hk=hk, vka=vka, vkcb=vkcb)

    grid_ix = flopy.utils.GridIntersect(m.modelgrid, method="vertex")

    # bas package
    if False:
        # make cells inactive where there is no botm in one of the layers
        ibound = ~np.any(np.isnan(botm), axis=0)
        ibound = np.reshape(ibound, [1, nrow, ncol])
        ibound = np.repeat(ibound, nlay, axis=0)
        # or where there is no hk
        ibound = ibound & ~np.isnan(m.lpf.hk.array)
        # or where there is no vkcb where it should have been
        lays = np.where(m.dis.laycbd.array)[0]
        ibound[lays] = ibound[lays] & ~np.isnan(m.lpf.vkcb.array[lays])
        ibound = ibound.astype(int)

        flopy.modflow.ModflowBas(m, ibound=ibound)
    else:
        boundary = get_grid_gdf(grid, kind="boundary")
        ibound2d = inpolygon(
            m.modelgrid.xcellcenters, m.modelgrid.ycellcenters, boundary.iloc[0, 0]
        )
        ibound2d = ibound2d.astype(int)
        ibound2d_c = ibound2d.copy()
        boundary_linestring = boundary.copy()
        boundary_linestring.geometry = boundary.boundary
        bound_cells = geodataframe_to_grid(
            boundary_linestring, mgrid=m.modelgrid, grid_ix=grid_ix, progressbar=False
        )
        bound_cells["row"] = bound_cells.cellids.apply(lambda s: s[0])
        bound_cells["col"] = bound_cells.cellids.apply(lambda s: s[1])

        ibound2d[bound_cells["row"], bound_cells["col"]] = -1
        ibound = np.ones((m.nlay, m.nrow, m.ncol))
        for lay in range(m.nlay):
            if quasi3d or (np.mod(lay, 2) == 0):
                ibound[lay] = ibound2d
            else:
                ibound[lay] = ibound2d_c
        strt = np.full((m.nlay, m.nrow, m.ncol), np.NaN)
        for lay in range(1, 9):
            if datadir is None:
                fname = (
                    "data/extracted/BasisbestandenNHDZmodelPWN_CvG_20180611/"
                    f"boundary/BH{lay}_2007.shp"
                )
            else:
                fname = os.path.join(
                    datadir,
                    "boundary",
                    f"BH{lay}_2007.shp",
                )

            strt2d = shp2grid2(
                fname,
                mgrid=m.modelgrid,
                grid_ix=grid_ix,
                method="linear",
                fields=["VALUE"],
                progressbar=False,
            )
            if quasi3d:
                strt[lay - 1] = strt2d
            else:
                strt[int((lay - 1) * 2)] = strt2d
        if not quasi3d:
            for lay in range(1, m.nlay, 2):
                strt[lay] = (strt[lay - 1] + strt[lay + 1]) / 2

        # make sure there are no NaN's in strt
        for lay in range(m.nlay):
            strt[lay] = fill_gaps(
                strt[lay], m.modelgrid.xcellcenters, m.modelgrid.ycellcenters
            )

        flopy.modflow.ModflowBas(m, ibound=ibound, strt=strt)

    # top-systheem 4:
    assert topsystem == 4, ValueError()
    # rch package
    rech = ds["GWA20022015"].values
    rech[np.isnan(rech)] = 0.0
    flopy.modflow.ModflowRch(m, rech=rech)

    # drn and riv package
    riv_spd = np.full((0, 6), np.NaN)
    ghb_spd = np.full((0, 5), np.NaN)
    drn_spd = np.full((0, 5), np.NaN)

    # determine the botm of the aquifers
    ind = np.arange(m.dis.nlay) + np.cumsum(np.hstack((0, m.dis.laycbd[:-1])))
    botm_aq = m.dis.botm.array[ind]

    # add the first drainage level by hand from the polygons
    if 1 not in drainage_systems:
        if datadir is None:
            pathname = os.path.join(
                datadir, "extracted", "BasisbestandenNHDZmodelPWN_CvG_20180611"
            )
            pathname2 = os.path.join(
                datadir, "extracted", "BasisbestandenNHDZmodelPWN_CvG_20180910"
            )
        else:
            pathname = os.path.join(
                datadir
            )
            pathname2 = os.path.join(
                datadir
            )
        fname = os.path.join(pathname2, "Topsyst", "Peilgebieden.shp")
        pg = gpd.read_file(fname)
        # pg['IWS_GPGVAS'][pg['IWS_GPGVAS']==0.0]=-999.
        # pg['GPGZMRPL'][pg['GPGZMRPL']==0.0]=-999.
        # pg['GPGWNTPL'][pg['GPGWNTPL']==0.0]=-999.
        # pg['IWS_GPGOND'][(pg['IWS_GPGOND']==0.0) | (pg['IWS_GPGOND']>=90)]=-999.
        # pg['IWS_GPGBOV'][(pg['IWS_GPGBOV']==0.0) | (pg['IWS_GPGBOV']>=90)]=-999.
        # pg['IWS_ONDERG'][(pg['IWS_ONDERG']==0.0) | (pg['IWS_ONDERG']>=90)]=-999.
        # pg['IWS_BOVENG'][(pg['IWS_BOVENG']==0.0) | (pg['IWS_BOVENG']>=90)]=-999.
        for i, pol in pg.iterrows():
            if pol.IWS_GPGVAS == 0.0:
                HHNK_GPGVAS = -999
            else:
                HHNK_GPGVAS = pol.IWS_GPGVAS

            if pol.GPGZMRPL == 0.0:
                HHNK_GPGZMRPL = -999
            else:
                HHNK_GPGZMRPL = pol.GPGZMRPL

            if pol.GPGWNTPL == 0.0:
                HHNK_GPGWNTPL = -999
            else:
                HHNK_GPGWNTPL = pol.GPGWNTPL

            if pol.IWS_GPGOND == 0.0 or pol.IWS_GPGOND >= 90:
                HHNK_GPGOND = -999
            else:
                HHNK_GPGOND = pol.IWS_GPGOND

            if pol.IWS_GPGBOV == 0.0 or pol.IWS_GPGBOV >= 90:
                HHNK_GPGBOV = -999
            else:
                HHNK_GPGBOV = pol.IWS_GPGBOV

            if pol.IWS_ONDERG == 0.0 or pol.IWS_ONDERG >= 90:
                HHNK_ONDERG = -999
            else:
                HHNK_ONDERG = pol.IWS_ONDERG

            if pol.IWS_BOVENG == 0.0 or pol.IWS_BOVENG >= 90:
                HHNK_BOVENG = -999
            else:
                HHNK_BOVENG = pol.IWS_BOVENG

            if HHNK_GPGZMRPL > -999:
                HHNK_RP3zomer = HHNK_GPGZMRPL
            elif HHNK_GPGVAS > -999:
                HHNK_RP3zomer = HHNK_GPGVAS
            else:
                HHNK_RP3zomer = max(HHNK_GPGOND, HHNK_GPGBOV, HHNK_ONDERG, HHNK_BOVENG)

            if HHNK_GPGWNTPL > -999:
                HHNK_RP3winter = HHNK_GPGWNTPL
            elif HHNK_GPGVAS > -999:
                HHNK_RP3winter = HHNK_GPGVAS
            else:
                HHNK_RP3winter = (
                    max(HHNK_GPGOND, HHNK_GPGBOV, HHNK_ONDERG, HHNK_BOVENG) - 0.2
                )

            pg.at[i, "HHNK_RP3gem"] = np.mean([HHNK_RP3zomer, HHNK_RP3winter])

        # read the polderpeil
        fname = os.path.join(pathname, "Topsyst", "gem_polderpeil2007.shp")
        gempeil = read_polygon_shape(fname, "Hp")

        # read the panden
        fname = os.path.join(pathname, "Topsyst", "Panden2008.shp")
        panden = read_polygon_shape(fname, "panden")
        panden = panden.loc[panden.is_valid]
        panden = panden.loc[
            (panden.panden < 100000) & (panden.panden > 0) & ~panden.is_empty
        ]

        ts1 = gpd.overlay(gempeil, panden, "union")
        ts1 = ts1[ts1.area > 1]

        # read the drainage-resistance
        fname = os.path.join(pathname, "Topsyst", "drainageweerstand.shp")
        wdrainage = read_polygon_shape(fname, "Wd")
        ts1 = gpd.overlay(ts1, wdrainage, "intersection")
        ts1 = ts1[ts1.area > 1]

        # read the infiltration-resistance
        fname = os.path.join(pathname, "Topsyst", "infiltratieweerstand.shp")
        winfiltratie = read_polygon_shape(fname, "Wi")
        ts1 = gpd.overlay(ts1, winfiltratie, "intersection")
        ts1 = ts1[ts1.area > 1]
        ts1.loc[~np.isnan(ts1.panden), "Wi"] = 0.1

        fname = os.path.join(pathname, "Topsyst", "codes_voor_typedrainage.shp")
        codesoort = read_polygon_shape(fname, "codesoort")
        ts1 = gpd.overlay(ts1, codesoort, "intersection")
        ts1 = ts1[ts1.area > 1]

        if False:
            fname = os.path.join(pathname, "Topsyst", "MVdtm2007.shp")
            MVdtm = gpd.read_file(fname)
            MVdtm = MVdtm[MVdtm["VALUE"] < -10]

            fname = os.path.join(pathname, "Topsyst", "mvpolder2007.shp")
            gpd.read_file(fname)

            panden = gpd.sjoin(panden, gempeil, rsuffix="peil")
            overig = gpd.overlay(gempeil, panden, how="difference")
            overig.crs = panden.crs
            # add infiltration resistance
            panden["Wi"] = 0.1
            # overig = gpd.sjoin(overig,winfiltratie,rsuffix='Wi')
            overig = gpd.overlay(overig, winfiltratie, how="intersection")
            # combine panden and overig again
            columns = ["geometry", "Hp", "Wi"]
            ts1 = gpd.GeoDataFrame(
                pd.concat([panden[columns], overig[columns]], ignore_index=True)
            )
            # add drainage resistance
            wdrainage = gpd.overlay(wdrainage, codesoort, "intersection")
            ts1 = gpd.overlay(ts1, wdrainage, how="intersection")

        # calculate BD (where isnan(BD) BD=mv-0.08)
        ts1["BD"] = np.NaN
        mask = ts1["codesoort"] != 4
        codesoort = ts1.loc[mask, "codesoort"]
        gempeil = ts1.loc[mask, "Hp"]
        panden = ts1.loc[mask, "panden"]
        ts1.loc[mask, "BD"] = (
            110 * (codesoort == 1)
            + (gempeil - 0.10) * (codesoort == 2)
            + 110 * (codesoort == 3)
            + (gempeil - 2) * (codesoort == 11)
            + (gempeil - 10) * (codesoort == 5)
            + (gempeil - 10) * (codesoort == 6)
            + 110 * (codesoort == 7)
            + (gempeil - 0.5) * (codesoort == 8)
            + 110 * (codesoort == 9)
            + 110 * (codesoort == 10)
            - 108 * ((panden < 100000) & (panden > 0))
        )

        # codesoort
        # 1: gebied rond infiltratiepanden (doet niets, bahelave waar panden zijn)
        # 2: polder
        # 3: duingebied (doet niets)
        # 4: ander polder-gebied (drainage?)
        # 5: noordzee (lage bodem, dus blijft infiltreren)
        # 6: Noordzeekanaal (lage bodem, dus blijft infiltreren)
        # 7: hoogovens (doet niets)
        # 9: onttrekkingsputten (doet niets)
        # 10: duinmeertjes (doen niets)

        # cut by the grid
        ts1 = geodataframe_to_grid(
            ts1,
            m.modelgrid,
            grid_ix=grid_ix,
            keepcols=ts1.columns.difference({"geometry"}),
            progressbar=False,
        )

        # overig = gpd.sjoin(overig,gempeil,rsuffix='peil')
        # ts1 = gpd.overlay(ts1,codesoort,how='intersection')
        for i, pol in ts1.iterrows():
            row, col = pol.cellids
            # get the bottom
            if np.isnan(pol.BD):
                # codesoort=4
                b = ds["mv"][row, col] - 0.08
            else:
                b = pol.BD

            # detemine layer
            li = 0
            # l = np.where(pol.Hp>botm_aq[:,pol.row,pol.col])[0][0]
            # Calculate the conductances
            ci = pol.geometry.area / pol.Wi
            cd = pol.geometry.area / pol.Wd

            # and add the cell to the riv, ghb or drn packages
            riv_spd, ghb_spd, drn_spd = add_top_system_cell(
                li, row, col, pol.Hp, ci, cd, b, riv_spd, ghb_spd, drn_spd
            )

    # get stage
    Hp = ds["RP3"].values

    # determine in which layer this level is
    level = np.full(Hp.shape, 0)
    if False:
        # use the layer that the polder level Hp falls in
        for lay in range(nlay):
            mask = Hp < botm_aq[lay]
            level[mask] = level[mask] + 1
    # get surface area of each cell
    delr, delc = np.meshgrid(m.dis.delr, m.dis.delc)
    area = delc * delc
    for iS in drainage_systems:
        print("Drainage/infiltration system {}".format(iS))
        # use the river package and the drain package to incorporate different
        # drainage and infiltration resistances
        Wd = getattr(ds, "RP{}".format(3 + iS))
        Wi = getattr(ds, "RP{}".format(6 + iS))
        BD = getattr(ds, "RP{}".format(9 + iS))
        if isinstance(BD, xr.DataArray):
            BD = BD.values

        # first the river cells
        # Calculate the conductance
        Ci = area / Wi
        # infiltration will only take place if Hp>BD
        mask = BD >= Hp
        Ci[mask] = 0.0
        if np.any(Wi < 10000):
            e = Hp.copy()
            r, c = np.where(~np.isnan(e) & (Ci > 0) & (BD < 100))
            lrcecb = np.vstack((level[r, c], r, c, Hp[r, c], Ci[r, c], BD[r, c])).T
            riv_spd = np.vstack((riv_spd, lrcecb))

        # the river pacckage assumes an equal conductance for infiltration and drainage
        # therefore add drains to incorporate different conductances
        # Calculate the conductance
        Cd = area / Wd
        if np.any(Wd < 10000):
            # substract the infiltration conductance
            mask = ~np.isnan(Ci)
            Cd[mask] = Cd[mask] - Ci[mask]
            if False:
                # assume for now that the drainage conductance should allways
                # be higher than the infiltration conductance
                Cd[Cd < 0] = 0
            # the elevation of the drains is the maximium of Hp and BD
            e = Hp.copy()
            # use the bottom of the drain if the bottom is higher than the polder water level
            mask = BD > Hp
            e[mask] = BD[mask]
            r, c = np.where(~np.isnan(e) & ~np.isnan(Cd) & (Cd != 0) & (BD < 100))
            lrcec = np.vstack((level[r, c], r, c, e[r, c], Cd[r, c])).T
            drn_spd = np.vstack((drn_spd, lrcec))

    if True:
        # add data from Triwaco-rivers
        riv_gdf = get_grid_gdf(grid, kind="rivers")

        # en lees ook wat eigenschappen de rivers in
        if datadir is None:
            pathname = os.path.join(
                "data", "extracted", "BasisbestandenNHDZmodelPWN_CvG_20180611"
            )
        else:
            pathname = os.path.join(
                datadir, 
            )

        # codesoortRA
        # 1: overgang duin
        # 2: polder
        # 3: duingebied rond infiltratiekanalen
        # 4: andere delen polder
        # 5: zee, doe niets
        # 6: noordzeekanaal, in laag 1 én 2
        # 7: hoogovens
        # 8: bestaat niet
        # 9 onttrekkingsputten: gemodelleerd als HOBO's, alleen in laag 2
        # 10: meertjes (?) in duinen
        # 11: gebied direct rond infiltratiekanalen, doe niets
        # 12: vier kleine meertjes bij Bakkum, doe niets
        # 13:
        # 14:
        # 15: twee kleine polygoontjes in duin, doe niets
        # determine river-activity RA1
        # cut with codesoortRA
        fname = os.path.join(
            pathname, "Rivers", "Codes_voor_type_river_activity2008.shp"
        )
        codesoortRA = read_polygon_shape(fname, "codesoortRA")
        riv_gdf = gpd.sjoin(riv_gdf, codesoortRA).drop("index_right", axis=1)
        codesoortRA = riv_gdf["codesoortRA"]
        riv_gdf["RA1"] = (
            1 * (codesoortRA == 1)
            + 1 * (codesoortRA == 2)
            + 3 * (codesoortRA == 3)
            + 3 * (codesoortRA == 4)
            + 0 * (codesoortRA == 5)
            + 1 * (codesoortRA == 6)
            + 1 * (codesoortRA == 7)
            + 1 * (codesoortRA == 8)
            + 0 * (codesoortRA == 9)
            + 2 * (codesoortRA == 10)
            + 0 * (codesoortRA == 11)
        )

        # RA2 is vervangen door mnw-putten
        if False:
            riv_gdf["RA2"] = (
                0 * (codesoortRA == 1)
                + 0 * (codesoortRA == 2)
                + 0 * (codesoortRA == 3)
                + 0 * (codesoortRA == 4)
                + 0 * (codesoortRA == 5)
                + 1 * (codesoortRA == 6)
                + 0 * (codesoortRA == 7)
                + 0 * (codesoortRA == 8)
                + 2 * (codesoortRA == 9)
                + 0 * (codesoortRA == 10)
                + 0 * (codesoortRA == 11)
            )
            if figpath is not None:
                from nhflotools.pwnlayers.plot import plot_shapes

                plot_shapes(m, riv_gdf, "RA2", figpath, "RA2")

        # determine river width RW1
        fname = os.path.join(pathname, "Rivers", "RW1_2007.shp")
        add_point_data_to_riv_gdf(fname, riv_gdf, "RW1", -999.0)

        # determine drainage resistance CD1
        fname = os.path.join(pathname, "Rivers", "CD1_2007.shp")
        add_point_data_to_riv_gdf(fname, riv_gdf, "CD1", -999.0)

        # determine infiltration resistance CI1
        fname = os.path.join(pathname, "Rivers", "CI1_2007.shp")
        add_point_data_to_riv_gdf(fname, riv_gdf, "CI1", 1.0)

        # detemine bottom of rivers BR1
        fname = os.path.join(pathname, "Rivers", "BR1_2007.shp")
        add_point_data_to_riv_gdf(fname, riv_gdf, "BR1", 1.0)

        # detemine water level HR1
        fname = os.path.join(pathname, "Topsyst", "gem_polderpeil2007.shp")
        HR1 = read_polygon_shape(fname, "HR1")
        riv_gdf = gpd.sjoin(riv_gdf, HR1).drop("index_right", axis=1)

        # cut by the grid
        riv_gdf = geodataframe_to_grid(
            riv_gdf,
            m.modelgrid,
            grid_ix=grid_ix,
            keepcols=riv_gdf.columns.difference({"geometry"}),
            progressbar=False,
        )

        for i, pol in riv_gdf.iterrows():
            if pol.RA1 > 0 and pol.RA1 != 2:  # for now ignore HOBOs
                # get the bottom
                if pol.RA1 == 1:
                    b = np.NaN
                elif pol.RA1 == 2:
                    raise (ValueError("HOBOs are not supported!"))
                elif pol.RA1 == 3:
                    b = pol.BR1
                else:
                    raise (ValueError("Unknown RA: {}".format(pol.RA1)))

                # detemine layer
                layer = 0
                # l = np.where(pol.HR1>botm_aq[:,pol.row,pol.col])[0][0]
                # Calculate the conductances
                area = pol.geometry.length * pol.RW1
                ci = area / pol.CI1
                cd = area / pol.CD1

                row, col = pol.cellids

                # and add the cell to the riv, ghb or drn packages
                riv_spd, ghb_spd, drn_spd = add_top_system_cell(
                    layer, row, col, pol.HR1, ci, cd, b, riv_spd, ghb_spd, drn_spd
                )

    flopy.modflow.ModflowRiv(m, stress_period_data=riv_spd)
    flopy.modflow.ModflowGhb(m, stress_period_data=ghb_spd)
    flopy.modflow.ModflowDrn(m, stress_period_data=drn_spd)

    # oc package
    flopy.modflow.ModflowOc(m)

    # pcg package
    flopy.modflow.ModflowPcg(m, rclose=0.001, hclose=0.001)

    # vdf-package
    if density:
        if quasi3d:
            raise (Exception("Cannot use quasi-3d layers when using SEAWAT"))
        dense = np.full((nlay, nrow, ncol), np.NaN)
        for lay in range(nlay):
            if np.mod(lay, 2) == 0:
                # layer is an quifer
                iaq = int(lay / 2) + 1
                key = "DA{}".format(iaq)
            else:
                # layer is an aquitard
                iaq = int((lay - 1) / 2) + 1
                key = "DC{}".format(iaq)
            dense[lay] = ds[key]
            dense[lay] = fill_gaps(
                dense[lay], m.modelgrid.xcellcenters, m.modelgrid.ycellcenters
            )

        flopy.seawat.SeawatVdf(m, mtdnconc=0, denseref=1000.0, dense=dense)

        import sys

        if sys.platform.startswith("win"):
            m.exe_name = nlmod.util.get_exe_path("swtv4")
        else:
            m.exe_name = nlmod.util.get_exe_path("swtv4")
    else:
        import sys

        if sys.platform.startswith("win"):
            m.exe_name = nlmod.util.get_exe_path("mf2005")
        else:
            m.exe_name = nlmod.util.get_exe_path("mf2005")

    return m


def read_pwn_data(m, ds, datadir=None, cachedir=None):
    """reads model data from a directory


    Parameters
    ----------
    m : flopy.modflow.mf.Modflow
        modflow model, the grid of this model is used to project the pwn data
        on.
    ds : xarray Dataset
        model dataset.
    datadir : str, optional
        directory with modeldata. The default is None.
    cachedir : str, optional
        cachedir used to cache files using the decorator
        nlmod.cache.cache_netcdf. The default is None.

    Returns
    -------
    ds : xarray Dataset
        model dataset.

    """
    # if datadir is None:
    #     pathname = os.path.join(
    #         "data", "extracted", "BasisbestandenNHDZmodelPWN_CvG_20180611"
    #     )
    #     pathname2 = os.path.join(
    #         "data", "extracted", "BasisbestandenNHDZmodelPWN_CvG_20180910"
    #     )
    # else:
    #     pathname = os.path.join(datadir, "BasisbestandenNHDZmodelPWN_CvG_20180611")
    #     pathname2 = os.path.join(datadir, "BasisbestandenNHDZmodelPWN_CvG_20180910")
    pathname = pathname2 = datadir

    ds.update(
        _read_top_of_aquitards(ds, pathname, m, cachedir=cachedir, cachename="triw_top")
    )

    ds.update(
        _read_thickness_of_aquitards(
            ds, pathname, m, cachedir=cachedir, cachename="triw_thick"
        )
    )
    ds.update(
        _read_kd_of_aquifers(ds, pathname, m, cachedir=cachedir, cachename="triw_kd")
    )
    ds.update(
        _read_mask_of_aquifers(
            ds, pathname, m, cachedir=cachedir, cachename="triw_aqui_mask"
        )
    )

    ds.update(
        _read_layer_kh(ds, pathname, m, cachedir=cachedir, cachename="triw_lay_kh")
    )
    ds.update(
        _read_kv_area(ds, pathname, m, cachedir=cachedir, cachename="triw_kv_area")
    )

    ds.update(
        _read_topsysteem(ds, pathname, m, cachedir=cachedir, cachename="triw_topsys")
    )

    ds.update(_read_zout(ds, pathname2, m, cachedir=cachedir, cachename="triw_zout"))

    ds.update(
        _read_bodemparams(
            ds, pathname2, m, cachedir=cachedir, cachename="triw_bodempar"
        )
    )

    # some more data that is taken from Modelopzet_model.ini
    ds.attrs["RL1"] = 50  # m NAP
    ds.attrs["TX8"] = 5000  # m2/d
    ds.attrs["neerslag"] = 0.00225  # m/d
    ds.attrs["verdamping"] = 0.00161  # m/d
    ds.attrs["alpha"] = 1  # -
    ds.attrs["gwstdiep"] = 0
    ds.attrs["RP5"] = 20  # d
    ds.attrs["RP6"] = 10000  # d
    ds.attrs["RP8"] = 100  # d
    ds.attrs["RP9"] = 10000  # d
    ds.attrs["RP12"] = 110  # m NAP
    ds.attrs["RP13"] = 110  # m NAP

    ds.update(_read_fluzo(ds, datadir, m, cachedir=cachedir, cachename="triw_fluzo"))

    return ds


def combine_pwn_regis_ds(pwn_ds, regis_ds, datadir=None, df_koppeltabel=None):
    """Create a new layer model based on regis and pwn models.


    Parameters
    ----------
    pwn_ds : xr.DataSet
        lagenmodel van pwn.
    regis_ds : xr.DataSet
        lagenmodel regis.
    datadir : str, optional
        datadirectory met koppeltabel. The default is None.
    df_koppeltabel : pandas DataFrame, optional
        dataframe van koppeltabel. The default is None.

    Raises
    ------
    NotImplementedError
        some combinations of regis and pwn are not implemented yet.

    Returns
    -------
    pwn_regis_ds : xr.DataSet
        combined model

    """

    if nlmod.util.compare_model_extents(regis_ds.extent, pwn_ds.extent) == 1:
        pwn_regis_ds = add_regis_to_bottom_of_pwn(pwn_ds, regis_ds)
    else:
        pwn_regis_ds = combine_layer_models_regis_pwn(
            pwn_ds, regis_ds, datadir, df_koppeltabel
        )

    return pwn_regis_ds


def get_overlap_model_layers(
    ml_layer_ds1,
    ml_layer_ds2,
    name_bot_ds1="botm",
    name_bot_ds2="botm",
    name_ds1=None,
    name_ds2=None,
    only_borders=False,
):
    """Get the overlap for each layer in layer model 1 with each layer in
    layer model 2.


    Parameters
    ----------
    ml_layer_ds1 : xr.Dataset
        layer model 1.
    ml_layer_ds2 :  xr.Dataset
        layer model 2.
    name_bot_ds1 : str, optional
        name of the data variable in model_ds1 with the bottom data array.
        The default is 'botm'.
    name_bot_ds2 : TYPE, optional
        name of the data variable in model_ds2 with the bottom data array.
        The default is 'botm'.
    name_ds1 : str or None, optional
        name of the layer model 1. If None the modelname attribute of the
        model_ds1 is used. The default is 'layer model 1'.
    name_ds2 : str or None, optional
        name of the layer model 2.  If None the modelname attribute of the
        model_ds2 is used. The default is 'layer model 2'.
    only_borders : bool, optional
        if True only the borders of the two layer models are compared. If
        False the comparison is made for all cells. The default is False.

    Returns
    -------
    df_nan : pandas DataFrame
        The percentage of nan values in one or both of the two compared layers.
    df_overlap : pandas DataFrame
        The percentage of overlap between two layers, that is the number of
        cells in one layer model that have some overlap with the same cell in
        another layer model divided by the total number of cells. A cell is
        considered to have some overlap if the cell in one layer model is not
        completely above or below the cell in another layer model. Inactive
        cells with nan values are considered to have no overlap.
    df_overlap_nonan : pandas DataFrame
        The same as df_overlap only now inactive cells with nan values are
        excluded from the calculation.

    """
    if name_ds1 is None:
        name_ds1 = ml_layer_ds1.model_name

    if name_ds2 is None:
        name_ds2 = ml_layer_ds2.model_name

    df_base = pd.DataFrame(index=ml_layer_ds1.layer + 1, columns=ml_layer_ds2.layer)
    df_base.index.name = name_ds1
    df_base.columns.name = name_ds2
    df_nan = df_base.copy()
    df_above = df_base.copy()
    df_below = df_base.copy()
    df_overlap = df_base.copy()
    df_overlap_nonan = df_base.copy()

    if only_borders:
        border_mask = xr.zeros_like(ml_layer_ds1[name_bot_ds1][0])
        border_mask[:, -1] = 1
        border_mask[0, :] = 1
        border_mask[-1, :] = 1

    y_overlap = np.array(list(set(ml_layer_ds1.y.values) & set(ml_layer_ds2.y.values)))
    x_overlap = np.array(list(set(ml_layer_ds1.x.values) & set(ml_layer_ds2.x.values)))

    y_overlap = np.sort(y_overlap)[::-1]
    x_overlap.sort()

    for model_lay_1 in range(len(ml_layer_ds1.layer)):
        for model_lay_2 in range(len(ml_layer_ds2.layer)):
            compare_lay = compare_layer_models_top_view(
                ml_layer_ds2,
                ml_layer_ds1,
                model_lay_2,
                model_lay_1,
                name_bot_ds2,
                name_bot_ds1,
                x_overlap,
                y_overlap,
            )
            if only_borders:
                compare_lay = xr.where(border_mask, compare_lay, np.nan)
                per_nan = 100 * (compare_lay == 12.0).sum() / border_mask.values.sum()
                per_above = 100 * (compare_lay == 11.0).sum() / border_mask.values.sum()
                per_below = 100 * (compare_lay == 8.0).sum() / border_mask.values.sum()
            else:
                per_nan = (
                    100
                    * (compare_lay == 12.0).sum()
                    / (compare_lay.shape[0] * compare_lay.shape[1])
                )
                per_above = (
                    100
                    * (compare_lay == 11.0).sum()
                    / (compare_lay.shape[0] * compare_lay.shape[1])
                )
                per_below = (
                    100
                    * (compare_lay == 8.0).sum()
                    / (compare_lay.shape[0] * compare_lay.shape[1])
                )
            per_overlap = 100 - per_nan - per_above - per_below
            per_overlap_no_nan = (
                100 * per_overlap / (per_above + per_below + per_overlap)
            )

            df_nan.loc[
                model_lay_1 + 1, ml_layer_ds2.layer.data[model_lay_2]
            ] = per_nan.data
            df_above.loc[
                model_lay_1 + 1, ml_layer_ds2.layer.data[model_lay_2]
            ] = per_above.data
            df_below.loc[
                model_lay_1 + 1, ml_layer_ds2.layer.data[model_lay_2]
            ] = per_below.data
            df_overlap.loc[
                model_lay_1 + 1, ml_layer_ds2.layer.data[model_lay_2]
            ] = per_overlap.data
            df_overlap_nonan.loc[
                model_lay_1 + 1, ml_layer_ds2.layer.data[model_lay_2]
            ] = per_overlap_no_nan.data

        print(
            f"compared model {name_ds1} layer {model_lay_1} with all layers in {name_ds2}"
        )

    df_overlap = df_overlap.astype(float)
    df_overlap[df_overlap == 0] = np.nan
    nan_cols = df_overlap.columns[df_overlap.isna().all(axis=0)]
    df_overlap.drop(columns=nan_cols, inplace=True)

    df_overlap_nonan = df_overlap_nonan.astype(float)
    df_overlap_nonan[df_overlap_nonan == 0] = np.nan
    nan_cols = df_overlap_nonan.columns[df_overlap_nonan.isna().all(axis=0)]
    df_overlap_nonan.drop(columns=nan_cols, inplace=True)

    return df_nan, df_overlap, df_overlap_nonan


def get_pwn_extent(regis_ds, pwn_ds):
    """get the extent of the part of the pwn model that is inside the
    regis model.


    Parameters
    ----------
    regis_ds : xarray dataset
        dataset of regis model.
    pwn_ds : xarray dataset
        dataset of pwn model

    Returns
    -------
    extent_pwn : list, tuple or numpy array
        extent of the part of the pwn model that is inside the regis model

    """

    model_layer_combi_type = nlmod.util.compare_model_extents(
        regis_ds.extent, pwn_ds.extent
    )

    delr = regis_ds.delr
    delc = regis_ds.delc

    x = regis_ds.x.values
    y = regis_ds.y.values

    if model_layer_combi_type == 1:
        new_pwn_extent = regis_ds.extent.copy()
    else:
        xmin = x[x >= (pwn_ds.extent[0] + 0.5 * delr)].min() - 0.5 * delr
        xmax = x[x <= (pwn_ds.extent[1] - 0.5 * delr)].max() + 0.5 * delr
        ymin = y[y >= (pwn_ds.extent[2] + 0.5 * delc)].min() - 0.5 * delc
        ymax = y[y <= (pwn_ds.extent[3] - 0.5 * delc)].max() + 0.5 * delc
        new_pwn_extent = [xmin, xmax, ymin, ymax]

    return new_pwn_extent


def get_grid_gdf(grid, kind="elements", extent=None):
    x = grid["X-COORDINATES NODES"]
    y = grid["Y-COORDINATES NODES"]

    geometry = []
    if kind == "elements":
        e1 = grid["ELEMENT NODES 1"] - 1
        e2 = grid["ELEMENT NODES 2"] - 1
        e3 = grid["ELEMENT NODES 3"] - 1
        for i in range(grid["NUMBER ELEMENTS"]):
            nodes = [e1[i], e2[i], e3[i]]
            geometry.append(Polygon(zip(x[nodes], y[nodes])))
        index = range(1, grid["NUMBER ELEMENTS"] + 1)
    elif kind == "rivers":
        lrn = grid["LIST RIVER NODES"] - 1
        nnr = grid["NUMBER NODES/RIVER"]
        for i in range(grid["NUMBER RIVERS"]):
            nodes = lrn[nnr[:i].sum() : nnr[: i + 1].sum()]
            geometry.append(LineString(zip(x[nodes], y[nodes])))
        index = grid["RIVERNUMBER"]
    elif kind == "boundary":
        lbn = grid["LIST BOUNDARY NODES"] - 1
        geometry = [Polygon(zip(x[lbn], y[lbn]))]
        index = ["BOUNDARY"]
    else:
        raise (ValueError())
    gdf = gpd.GeoDataFrame(geometry=geometry, crs={}, index=index)
    if extent is not None:
        gdf = gdf.loc[gdf.intersects(nlmod.util.polygon_from_extent(extent))]
    return gdf
