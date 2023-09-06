# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:15:42 2018

@author: Artesia
"""

import os
import time
import warnings
import zipfile

import flopy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.qhull as qhull
from flopy.utils import Util2d
from flopy.utils import Util3d
from flopy.utils import reference
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.lib.recfunctions import append_fields
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm


def colorbar_inside(mappable=None, ax=None, width=0.2, height="90%", loc=5, **kw):
    if ax is None:
        ax = plt.gca()
    cax = inset_axes(ax, width=width, height=height, loc=loc)
    cb = plt.colorbar(mappable, cax=cax, ax=ax, **kw)
    if loc == 1 or loc == 4 or loc == 5:
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position("left")
    return cb


def title_inside(title, ax=None, x=0.5, y=0.98, **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.text(
        x,
        y,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        **kwargs,
    )


def geodataframe2grid(mf, shp_in):
    """Cut a geodataframe shp_in by the grid of a modflow model ml"""
    geom_col_name = shp_in._geometry_column_name

    # make a polygon for each of the grid-cells
    # this may result in lines on the edge of the polygon counting double
    grid_polygons = []
    for row in range(mf.sr.nrow):
        for col in range(mf.sr.ncol):
            vert = mf.sr.get_vertices(row, col)
            pol = Polygon(vert)
            pol.row = row
            pol.col = col
            grid_polygons.append(pol)

    s = STRtree(grid_polygons)

    shp_list = []
    # cut the lines with the grid
    for index, row in shp_in.iterrows():
        g = row[geom_col_name]
        result = s.query(g)
        for r in result:
            i = g.intersection(r)
            _add_shapes_to_list(i, g.geometryType(), geom_col_name, row, shp_list, r)
    if len(shp_list) == 0:
        warnings.warn("No overlap between model and shape")
        shp_out = shp_in.loc[[]]
        shp_out["row"] = 0
        shp_out["col"] = 0
    else:
        shp_out = gpd.GeoDataFrame(pd.DataFrame(shp_list), geometry=geom_col_name)
    return shp_out


def geodataframe2grid2(gdf, mgrid=None, grid_ix=None, keepcols=None, progressbar=False):
    if grid_ix is None and mgrid is not None:
        grid_ix = flopy.utils.GridIntersect(mgrid, method="vertex")
    elif grid_ix is None and mgrid is None:
        raise ValueError("Provide either 'mgrid' or 'grid_ix'!")

    reclist = []

    for _, row in (
        tqdm(gdf.iterrows(), total=gdf.index.size) if progressbar else gdf.iterrows()
    ):
        ishp = row.geometry

        if not ishp.is_valid:
            ishp = ishp.buffer(0)

        r = grid_ix.intersect(ishp)

        if keepcols is not None:
            dtypes = gdf.dtypes.loc[keepcols].to_list()
            val_arrs = [
                ival * np.ones(r.shape[0], dtype=idtype)
                for ival, idtype in zip(row.loc[keepcols], dtypes)
            ]
            r = append_fields(
                r, keepcols, val_arrs, dtypes, usemask=False, asrecarray=True
            )
        if r.shape[0] > 0:
            reclist.append(r)

    rec = np.concatenate(reclist)
    gdf = gpd.GeoDataFrame(rec, geometry="ixshapes")
    gdf.rename(columns={"ixshapes": "geometry"}, inplace=True)
    return gdf


def _add_shapes_to_list(i, geometryType, geom_col_name, row, shp_list, r):
    """subfunction of geodataframe2grid"""
    if geometryType == "LineString":
        if not i.is_empty:
            it = i.geometryType()
            if it == "GeometryCollection":
                for im in i.geoms:
                    _add_shapes_to_list(
                        im, geometryType, geom_col_name, row, shp_list, r
                    )
            elif it == "MultiLineString":
                for im in i.geoms:
                    _add_shapes_to_list(
                        im, geometryType, geom_col_name, row, shp_list, r
                    )
            elif it == "LineString":
                # TODO: to make sure lines are not counted double
                # do not add the line if the line is on the north or west
                # border of the cell-edge
                rown = row.copy()
                rown[geom_col_name] = i
                rown["row"] = r.row
                rown["col"] = r.col
                shp_list.append(rown)
            elif it == "Point":
                # endpoint of the linestring is on the cell-edge
                pass
            elif it == "MultiPoint":
                # mutiple endpoints of the linestring are on the cell-edge
                pass
            else:
                raise NotImplementedError(
                    "geometryType " + it + " not yet supprted in geodataframe2grid"
                )
    elif geometryType == "Polygon" or geometryType == "MultiPolygon":
        it = i.geometryType()
        if it == "GeometryCollection":
            for im in i.geoms:
                _add_shapes_to_list(im, geometryType, geom_col_name, row, shp_list, r)
        elif it == "MultiPolygon":
            for im in i.geoms:
                _add_shapes_to_list(im, geometryType, geom_col_name, row, shp_list, r)
        elif it == "Polygon":
            rown = row.copy()
            rown[geom_col_name] = i
            rown["row"] = r.row
            rown["col"] = r.col
            shp_list.append(rown)
        elif it == "Point":
            # endpoint of the polygon is on the cell-edge
            pass
        elif it == "LineString" or it == "MultiLineString":
            # one of the edges of the polygon is on a cell-egde
            pass
        else:
            raise NotImplementedError(
                "geometryType " + it + " not yet supprted in geodataframe2grid"
            )
    else:
        raise NotImplementedError(
            "geometryType " + geometryType + " not yet supprted in geodataframe2grid"
        )


def refine_grid(ml, xe_new, ye_new):
    # refine a modflow-grid, using nearest-interpolation
    # for the model-properties

    sr_old = ml.sr
    # xc_old=ml.sr.xcentergrid[0, :]
    # yc_old=ml.sr.ycentergrid[:, 0]

    xc_new = xe_new[0:-1] + np.diff(xe_new) / 2
    yc_new = ye_new[0:-1] + np.diff(ye_new) / 2
    xgrid, ygrid = np.meshgrid(xc_new, yc_new)

    packages = ml.get_package_list()

    def set_util2d(pack, parnam, parval, dtype=np.float32):
        value = Util2d(
            pack.parent,
            parval.shape,
            dtype,
            parval,
            name=parnam,
            locat=pack.unit_number[0],
        )
        setattr(pack, parnam, value)

    def set_util3d(pack, parnam, parval, dtype=np.float32):
        value = Util3d(
            pack.parent, parval.shape, dtype, parval, parnam, locat=pack.unit_number[0]
        )
        setattr(pack, parnam, value)

    def change_util2d(pack, parnam, sr_old, grid, ygrid):
        val_old = getattr(pack, parnam)
        val_new = sr_old.interpolate(val_old.array, (xgrid, ygrid), method="nearest")
        set_util2d(pack, parnam, val_new, val_old.dtype)

    def change_util3d(pack, parnam, sr_old, xgrid, ygrid):
        val_old = getattr(pack, parnam)
        val_new = np.empty((val_old.shape[0], xgrid.shape[0], xgrid.shape[1]))
        for iL, val in enumerate(val_old.array):
            val_new[iL] = sr_old.interpolate(val, (xgrid, ygrid), method="nearest")
        set_util3d(pack, parnam, val_new, val_old.dtype)

    for pack in packages:
        if pack in ["PCG", "OC"]:
            # packages have no spatial component
            pass

        elif pack == "DIS":
            ml.dis.ncol = len(xc_new)
            ml.dis.nrow = len(yc_new)
            set_util2d(ml.dis, "delr", np.diff(xe_new))
            set_util2d(ml.dis, "delc", -np.diff(ye_new))
            change_util2d(ml.dis, "top", sr_old, xgrid, ygrid)
            change_util3d(ml.dis, "botm", sr_old, xgrid, ygrid)
            ml.dis.sr = reference.SpatialReference(
                ml.dis.delr.array,
                ml.dis.delc.array,
                ml.dis.lenuni,
                xul=sr_old.xul,
                yul=sr_old.yul,
                rotation=sr_old.rotation,
                proj4_str=sr_old.proj4_str,
            )

        elif pack == "BAS6":
            change_util3d(ml.bas6, "ibound", sr_old, xgrid, ygrid)
            change_util3d(ml.bas6, "strt", sr_old, xgrid, ygrid)

        elif pack == "LPF":
            change_util3d(ml.lpf, "hani", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "hk", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "ss", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "sy", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "vka", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "vkcb", sr_old, xgrid, ygrid)
            change_util3d(ml.lpf, "wetdry", sr_old, xgrid, ygrid)

        elif pack == "RCH":
            change_util3d(ml.rch, "rech", sr_old, xgrid, ygrid)

        else:
            raise NotImplementedError(
                pack + "-package not implemeted yet in refine_grid"
            )


def unzip_file(src, dst, force=False, preserve_datetime=False):
    """Unzip file

    Parameters
    ----------
    src : str
        source zip file
    dst : str
        destination directory
    force : boolean, optional
        force unpack if dst already exists
    preserve_datetime : boolean, optional
        use date of the zipfile for the destination file

    Returns
    -------
    int
        1 of True

    """
    if os.path.exists(dst):
        if not force:
            print(
                "File not unzipped. Destination already exists. "
                "Use 'force=True' to unzip."
            )
            return
    if preserve_datetime:
        zipf = zipfile.ZipFile(src, "r")
        for f in zipf.infolist():
            zipf.extract(f, path=dst)
            date_time = time.mktime(f.date_time + (0, 0, -1))
            os.utime(os.path.join(dst, f.filename), (date_time, date_time))
        zipf.close()
    else:
        zipf = zipfile.ZipFile(src, "r")
        zipf.extractall(dst)
        zipf.close()
    return 1


def df2gdf(df, xcol="x", ycol="y"):
    """Convert DataFrame to a point GeoDataFrame

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    xcol : str
        column name with x values
    ycol : str
        column name with y values

    Returns
    -------
    gdf : geopandas.GeoDataFrame
    """
    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=[Point((s[xcol], s[ycol])) for i, s in df.iterrows()]
    )
    return gdf


def get_mt3d_results(f, kstpkper=(0, 0), mflay=0, inact=1e30):
    """ """
    ucnobj = flopy.utils.UcnFile(f)
    c = ucnobj.get_data(kstpkper=kstpkper, mflay=mflay)
    # set inactive to NaN
    c[c == inact] = np.nan

    return c


def script_newer_than_output(fscript, foutput):
    if not os.path.exists(foutput):
        return True

    tm_script = os.path.getmtime(fscript)

    if isinstance(foutput, str):
        tm_output = os.path.getmtime(foutput)
    else:
        tm_output = np.zeros(len(foutput))
        for i in range(len(foutput)):
            tm_output[i] = os.path.getmtime(foutput[i])

    return np.all(tm_script > tm_output)


def unzip_changed_files(
    zipname, pathname, check_time=True, check_size=False, debug=False
):
    # Extract each file in a zip-file only when the properties are different
    # With the default arguments this method only checks the modification time
    with zipfile.ZipFile(zipname) as zf:
        infolist = zf.infolist()
        for info in infolist:
            fname = os.path.join(pathname, info.filename)
            extract = False
            if os.path.exists(fname):
                if check_time:
                    tz = time.mktime(info.date_time + (0, 0, -1))
                    tf = os.path.getmtime(fname)
                    if tz != tf:
                        extract = True
                if check_size:
                    sz = info.file_size
                    sf = os.path.getsize(fname)
                    if sz != sf:
                        extract = True
            else:
                extract = True
            if extract:
                if debug:
                    print("extracting {}".format(info.filename))
                zf.extract(info.filename, pathname)
                # set the correct modification time
                # (which is the time of extraction by default)
                tz = time.mktime(info.date_time + (0, 0, -1))
                os.utime(os.path.join(pathname, info.filename), (tz, tz))


def interp_weights(xy, uv, d=2):
    """Calculate interpolation weights

    Parameters
    ----------
    xy : np.array
        array containing x-coordinates in first column and y-coordinates
        in second column
    uv : np.array
        array containing coordinates at which interpolation weights should
        be calculated, x-data in first column and y-data in second column
    d : int, optional
        dimension of data? (the default is 2, which works for 2D data)

    Returns
    -------
    vertices: np.array
        array containing interpolation vertices

    weights: np.array
        array containing interpolation weights per point

    Reference
    ---------
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    """

    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts):
    """interpolate values at locations defined by vertices and points,
       as calculated by interp_weights function.

    Parameters
    ----------
    values : np.array
        array containing values to interpolate
    vtx : np.array
        array containing interpolation vertices, see interp_weights()
    wts : np.array
        array containing interpolation weights, see interp_weights()

    Returns
    -------
    arr: np.array
        array containing interpolated values at locations as given by
        vtx and wts

    Reference
    ---------
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    """

    return np.einsum("nj,nj->n", np.take(values, vtx), wts)


def inpolygon(x, y, polygon, engine="matplotlib"):
    """find out which points defined by x and y are within polygon

    Parameters
    ----------
    x : np.array
        x-coordinates of grid (same shape as y)
    y : np.array
        y-coordinates of grid (same shape as x)
    poolygon : shapely Polygon or MuliPolygon
        the polygon for which you want mask to be True
    engine : str
        Use 'matplotlib' for speed, for all other values it uses shapely

    Returns
    -------
    mask: np.array
        an array of the same shape as x and y: True for points within polygon

    """
    shape = x.shape
    points = list(zip(x.flatten(), y.flatten()))
    if engine == "matplotlib":
        if isinstance(polygon, MultiPolygon):
            mask = np.full((len(points)), False)
            for pol2 in polygon:
                if not isinstance(pol2, Polygon):
                    raise (Exception("{} not supported".format(type(pol2))))
                if isinstance(pol2.boundary, MultiLineString):
                    xb, yb = pol2.boundary[0].xy
                else:
                    xb, yb = pol2.boundary.xy
                path = Path(list(zip(xb, yb)))
                mask = mask | path.contains_points(points)
        elif isinstance(polygon, Polygon):
            xb, yb = polygon.boundary.xy
            path = Path(list(zip(xb, yb)))
            mask = path.contains_points(points)
        else:
            raise (Exception("{} not supported".format(type(polygon))))
    else:
        mask = [polygon.contains(Point(x, y)) for x, y in points]
        mask = np.array(mask)
    return mask.reshape(shape)


def extent2polygon(extent):
    """Make a Polygon of the extent of a matplotlib axes"""
    nw = (extent[0], extent[2])
    no = (extent[1], extent[2])
    zo = (extent[1], extent[3])
    zw = (extent[0], extent[3])
    polygon = Polygon([nw, no, zo, zw])
    return polygon


def rotate_yticklabels(ax):
    yticklabels = ax.yaxis.get_ticklabels()
    plt.setp(yticklabels, rotation=90, verticalalignment="center")


def rd_ticks(ax, base=1000.0, fmt_base=1000, fmt="{:.0f}"):
    """Add ticks every 1000 (base) m, and divide ticklabels by 1000 (fmt_base)"""

    def fmt_rd_ticks(x, y):
        return fmt.format(x / fmt_base)

    if base is not None:
        ax.xaxis.set_major_locator(MultipleLocator(base))
        ax.yaxis.set_major_locator(MultipleLocator(base))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_rd_ticks))


def get_line_length(verts):
    lengths = []
    for i in range(len(verts) - 1):
        dx = verts[i + 1][0] - verts[i][0]
        dy = verts[i + 1][1] - verts[i][1]
        length = np.sqrt(dx**2 + dy**2)
        lengths.append(length)
    return np.sum(lengths)
