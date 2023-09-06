import logging

import flopy
import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def add_extraction_wells(fname, sheet, m, quasi3d, package="mnw1"):
    if sheet == "Putclusters_model2016":
        # read extraction wells
        pp = pd.read_excel(
            fname,
            sheet,
            skiprows=2,
            usecols=[2, 3, 4, 5],
            index_col="putcode",
            engine="openpyxl",
        )
        pp = gpd.GeoDataFrame(
            pp, geometry=[Point((s["X"], s["Y"])) for i, s in pp.iterrows()]
        )
        # reac well-clusters
        # pc = pd.read_excel(fname,sheet,skiprows=2,usecols=np.hstack([8,np.arange(25,300)])) # monthly values
        pc = pd.read_excel(
            fname, sheet, skiprows=2, usecols=np.arange(8, 25), engine="openpyxl"
        )  # yearly values
        if "putcluster" in pc:
            pc = pc.set_index("putcluster")
        else:
            pc = pc.set_index("putcluster.1")
        # remove empty rows
        pc = pc.loc[~np.all(pc.isna(), axis=1)]
        # make sure the names of the clusters are strings
        pc.index = pc.index.astype(int).astype(str)
        pp["putcluster"] = pp["putcluster"].astype(str)

        # rename 21xxx to Pxxx and 22xxx to Qxxx
        index = []
        for putcluster in pc.index:
            if len(putcluster) == 5:
                if putcluster.startswith("21"):
                    putcluster = putcluster.replace("21", "P")
                elif putcluster.startswith("22"):
                    putcluster = putcluster.replace("22", "Q")
            index.append(putcluster)
        pc.index = index

        # test if every putcluster has a discharge
        assert np.all([x in pc.index for x in pp.putcluster])

        # convert the date that are in the columns
        date = []
        for col in pc.columns:
            if len(col) == 5:
                # it is only a year
                # + pd.to_timedelta(1,'y')
                d = pd.to_datetime(col, format="Q%Y")
            elif len(col) == 7:
                # + pd.to_timedelta(1,'M')
                d = pd.to_datetime(col, format="Q%Y%m")
            else:
                raise (ValueError)
            date.append(d)
        pc.columns = date
        # switch rows and columns (so the index consisct of datetime-objects)
        pc = pc.T
        if False:
            # plot the discharge of each of the putstrengen
            pc.plot()
        # determine the extraction in each well
        for putcluster in pc.columns:
            mask = pp.putcluster == putcluster
            pp.loc[mask, "Q"] = pc.loc["2012", putcluster].mean() / mask.sum()
        pp["Aquifer"] = 2
    elif sheet == "Putten_model2016":
        pp = pd.read_excel(
            fname, sheet, skiprows=2, index_col="PUTCODE", engine="openpyxl"
        )
        pp = gpd.GeoDataFrame(
            pp, geometry=[Point((s["XCOOR"], s["YCOOR"])) for i, s in pp.iterrows()]
        )
        pp["Q"] = pp["Q2012"]
    else:
        raise (ValueError(sheet))

    # only keep wells that are inside the model area
    pp = pp.loc[pp.within(get_bounds_polygon(m))]

    if package == "mnw2":
        # use mnw2, which is not supported by SEAWAT (and does not work yet)
        mnw = []
        for putcluster in pc.columns:
            mask = pp.putcluster == putcluster
            rcs = [
                m.modelgrid.intersect(x, y)
                for x, y in zip(pp.loc[mask, "geometry"].x, pp.loc[mask, "geometry"].y)
            ]
            lay = pp.loc[mask, "Aquifer"] - 1  # so Aquifer is one-based
            if not quasi3d:
                lay = lay * 2
            row = [rc[0] for rc in rcs]
            col = [rc[0] for rc in rcs]
            spd = flopy.modflow.mfmnw2.Mnw.get_empty_stress_period_data(nper=1)
            # determine the discharge
            spd["qdes"][0] = pp.loc[mask, "Q"].sum()
            mnw.append(
                flopy.modflow.mfmnw2.Mnw(
                    putcluster,
                    nnodes=mask.sum(),
                    k=lay,
                    i=row,
                    j=col,
                    stress_period_data=spd,
                )
            )
        flopy.modflow.ModflowMnw2(model=m, mnwmax=len(mnw), mnw=mnw, itmp=[len(mnw)])
    elif package == "mnw1":
        # use mnw 1
        spd = []
        for ipc, putcluster in enumerate(pc.columns):
            mask = pp.putcluster == putcluster
            for ipp, putcode in enumerate(pp.index[mask]):
                lay = pp.at[putcode, "Aquifer"] - 1
                if not quasi3d:
                    lay = lay * 2
                row, col = m.modelgrid.intersect(
                    pp.at[putcode, "geometry"].x, pp.at[putcode, "geometry"].y
                )
                lrcqm = [ipc, lay, row, col, pp.at[putcode, "Q"]]
                if ipp > 0:
                    lrcqm.append("MN")
                else:
                    lrcqm.append("")
                spd.append(lrcqm)
        dtype = np.dtype(
            [
                ("mnw_no", np.int64),
                ("k", np.int64),
                ("i", np.int64),
                ("j", np.int64),
                ("qdes", np.float32),
                ("mntxt", object),
            ]
        )
        flopy.modflow.ModflowMnw1(
            m, stress_period_data=spd, dtype=dtype, mxmnw=len(spd)
        )
    else:
        # use the normal wel-package
        spd = []
        for putcode in pp.index:
            lay = pp.at[putcode, "Aquifer"] - 1
            if not quasi3d:
                lay = lay * 2
            row, col = m.modelgrid.intersect(
                x=pp.at[putcode, "geometry"].x, y=pp.at[putcode, "geometry"].y
            )
            lrcqm = [lay, row, col, pp.at[putcode, "Q"]]
            spd.append(lrcqm)
        flopy.modflow.ModflowWel(m, stress_period_data=spd)


def get_bounds_polygon(m):
    # make a polygon of the model boundary
    extent = m.modelgrid.extent
    return nlmod.util.polygon_from_extent(extent)
