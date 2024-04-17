import configparser
import logging
import os
import time
import zipfile

import flopy
import geopandas as gpd
import nlmod
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from nhflotools.pwnlayers import triwaco
from nlmod import cache
from numpy.lib.recfunctions import append_fields
from shapely.geometry import Point
from tqdm import tqdm

logger = logging.getLogger(__name__)


def geodataframe_to_grid(gdf, mgrid=None, grid_ix=None, keepcols=None, progressbar=False):
    if grid_ix is None and mgrid is not None:
        grid_ix = flopy.utils.GridIntersect(mgrid, method="vertex")
    elif grid_ix is None and mgrid is None:
        raise ValueError("Provide either 'mgrid' or 'grid_ix'!")

    reclist = []

    for _, row in tqdm(gdf.iterrows(), total=gdf.index.size) if progressbar else gdf.iterrows():
        ishp = row.geometry

        if not ishp.is_valid:
            ishp = ishp.buffer(0)

        r = grid_ix.intersect(ishp)

        if keepcols is not None:
            dtypes = gdf.dtypes.loc[keepcols].to_list()
            val_arrs = [ival * np.ones(r.shape[0], dtype=idtype) for ival, idtype in zip(row.loc[keepcols], dtypes)]
            r = append_fields(r, keepcols, val_arrs, dtypes, usemask=False, asrecarray=True)
        if r.shape[0] > 0:
            reclist.append(r)

    rec = np.concatenate(reclist)
    gdf = gpd.GeoDataFrame(rec, geometry="ixshapes")
    gdf.rename(columns={"ixshapes": "geometry"}, inplace=True)
    return gdf


# def gdf_inpolygon(ds, gdf, buffer=0.):
#     """Counts in how many polygons a cell appears.

#     Parameters
#     ----------
#     ds : xr.DataSet
#         xarray with model data
#     gdf : geopandas.GeoDataFrame
#         geodataframe with geometry
#     buffer : float, optional
#         buffer around geometry, by default 0.

#     Returns
#     -------
#     da : xr.DataArray
#         boolean data array with True for all cells within gdf

#     """
#     isstructured = "icell2d" not in ds.x.dims
#     # check if grid is structured
#     if isstructured:
#         mask = np.zeros((len(ds.x), len(ds.y)), dtype=int)
#     else:
#         mask = np.zeros((len(ds.x),), dtype=int)

#     for _, row in gdf.iterrows():
#         # minx, miny, maxx, maxy
#         in_extentx = (ds.x >= row.geometry.bounds[0] - buffer) & (ds.x <= row.geometry.bounds[2] + buffer)
#         in_extenty = (ds.y >= row.geometry.bounds[1] - buffer) & (ds.y <= row.geometry.bounds[3] + buffer)

#         if not in_extentx.any() or not in_extenty.any():
#             continue

#         if buffer > 0.:
#             if isstructured:
#                 xx, yy = np.meshgrid(ds.x[in_extentx], ds.y[in_extenty], indexing="ij")
#                 counts = inpolygon(xx.flatten(), yy.flatten(), row.geometry.buffer(buffer))
#                 mask[in_extentx * in_extenty] += counts

#             else:
#                 in_extent = in_extentx.values & in_extenty.values
#                 mask[in_extent] += inpolygon(ds.x[in_extent], ds.y[in_extent], row.geometry.buffer(buffer))

#         else:
#             mask[in_extent] += inpolygon(ds.x[in_extent], ds.y[in_extent], row.geometry)

#     if "icell2d" in ds.x.dims:
#         da = xr.DataArray(mask, coords={"icell2d": ds.icell2d})
#     else:
#         da = xr.DataArray(mask, coords={"x": ds.x, "y": ds.y})

#     return da


# def inpolygon(x, y, polygon, engine="matplotlib"):
#     """Counts in how many polygons a coordinate appears.

#     Parameters
#     ----------
#     x : np.array
#         x-coordinates of grid (same shape as y)
#     y : np.array
#         y-coordinates of grid (same shape as x)
#     polygon : shapely Polygon or MuliPolygon
#         the polygon for which you want mask to be True
#     engine : str
#         Use 'matplotlib' for speed, for all other values it uses shapely

#     Returns
#     -------
#     mask: np.array of integers
#         an array of the same shape as x and y: 1 for points within polygon and
#         0 for points outside polygon. In case of multipolygon the mask is 1
#         for points within 1 polygon and 2 in case of two overlapping polygons, etc.

#     """
#     if len(x) == 0:
#         return np.array([], dtype=int)

#     shape = x.shape
#     points = list(zip(np.asarray(x).flatten(), np.asarray(y).flatten()))
#     if engine == "matplotlib":
#         if isinstance(polygon, MultiPolygon):
#             mask = np.zeros((len(points)), dtype=int)

#             for polygon2 in polygon.geoms:
#                 path = Path(polygon2.exterior.coords)
#                 mask += path.contains_points(points)

#         elif isinstance(polygon, Polygon):
#             path = Path(polygon.exterior.coords)
#             mask = path.contains_points(points).astype(int)

#         else:
#             raise (Exception("{} not supported".format(type(polygon))))

#     else:
#         mask = [polygon.contains(Point(x, y)) for x, y in points]
#         mask = np.asarray(mask, dtype=int)
#     return mask.reshape(shape)


def compare_layer_models_top_view(
    ml_layer_ds1,
    ml_layer_ds2,
    layer_mod1=0,
    layer_mod2=0,
    name_bot_ds1="botm",
    name_bot_ds2="botm",
    xsel=None,
    ysel=None,
):
    """

    Parameters
    ----------
    ml_layer_ds1 : xr.Dataset
        layer model 1.
    ml_layer_ds2 :  xr.Dataset
        layer model 2.
    layer_mod1 : int, optional
        The index of the layer in lay_mod1. The default is 0.
    layer_mod2 : int, optional
        The index of the layer in lay_mod2. The default is 0.
    name_bot_ds1 : str, optional
        name of the data variable in model_ds1 with the bottom data array.
        The default is 'botm'.
    name_bot_ds2 : TYPE, optional
        name of the data variable in model_ds2 with the bottom data array.
        The default is 'botm'.
    xsel : list or np.array, optional
        x-coördinates that are used for the comparison. x-coördinates should
        be available within the extent of both layer models. If None the
        x-coördinates in ml_layer_ds1 are sued. The default is None.
    ysel : TYPE, optional
        y-coördinates that are used for the comparison. y-coördinates should
        be available within the extent of both layer models. If None the
        y-coördinates in ml_layer_ds1 are sued. The default is None.

    Returns
    -------
    compare : xarray DataArray of integers
        number for each cell indicating the overlap between two layer models.

    """
    if ysel is None:
        ysel = ml_layer_ds1.y.data
    if xsel is None:
        xsel = ml_layer_ds1.x.data

    lay_mod1 = ml_layer_ds1.sel(x=xsel, y=ysel)
    lay_mod2 = ml_layer_ds2.sel(x=xsel, y=ysel)

    bot1 = lay_mod1[name_bot_ds1][layer_mod1]
    if len(lay_mod1["top"].dims) == 2:
        if layer_mod1 == 0:
            top1 = lay_mod1["top"]
        else:
            top1 = lay_mod1[name_bot_ds1][layer_mod1 - 1]
    elif len(lay_mod1["top"].dims) == 3:
        top1 = lay_mod1["top"][layer_mod1]

    bot2 = lay_mod2[name_bot_ds2][layer_mod2]
    if len(lay_mod2["top"].dims) == 2:
        if layer_mod2 == 0:
            top2 = lay_mod2["top"]
        else:
            top2 = lay_mod2[name_bot_ds2][layer_mod2 - 1]
        top2 = lay_mod2["top"]
    elif len(lay_mod2["top"].dims) == 3:
        top2 = lay_mod2["top"][layer_mod2]

    compare = compare_top_bots(bot1, top1, bot2, top2)

    return compare


def compare_top_bots(bot1, top1, bot2, top2):
    """Compare two layer models. For a visual explanation of the comparison
    see https://github.com/ArtesiaWater/nlmod/blob/dev/examples/06_compare_layermodels.ipynb


    Parameters
    ----------
    bot1 : numpy array or xarray DataArray
        bottom layer 1.
    top1 : numpy array or xarray DataArray
        top layer 1.
    bot2 : numpy array or xarray DataArray
        bottom layer 2.
    top2 : numpy array or xarray DataArray
        top layer 2.

    Returns
    -------
    compare : numpy array or xarray DataArray
        For each cell a number indicating the overlap.

        numbers indicate the following:
            1. equal
            2. top within: top2 lower & bot2 equal
            3. bottom within: bot2 higher & top2 equal
            4. within: top2 lower & bot2 higher
            5. outside: top2 higher & bot2 lower
            6. top outside: top2 higher & bot2 equal
            7. bot outside: bot2 lower & top2 equal
            8. under: bot1 >= top2
            9. shifted down: (top1 > top2 > bot1) & (bot1 > bot2)
            10. shifted up: (top1 < top2) & (bot1 < bot2 < top1)
            11. above: top1 <= bot2
            12. nan (np.isnan(bot1, bot2, top1, top2).any())

    """
    compare = xr.zeros_like(top1)

    # 1: equal
    mask_eq = (top1 == top2) & (bot1 == bot2)

    # 2: top within: top2 lower & bot2 equal
    mask_top_within = (top1 > top2) & (bot1 == bot2)

    # 3: bottom within: bot2 higher & top2 equal
    mask_bot_within = (top1 == top2) & (bot1 < bot2)

    # 4: within: top2 lower & bot2 higher
    mask_within = (top1 > top2) & (bot1 < bot2)

    # 5: outside: top2 higher & bot2 lower
    mask_outside = (top1 < top2) & (bot1 > bot2)

    # 6: top outside: top2 higher & bot2 equal
    mask_top_oustide = (top1 < top2) & (bot1 == bot2)

    # 7: bot outside: bot2 lower & top2 equal
    mask_bot_outside = (top1 == top2) & (bot1 > bot2)

    # 8: under: bot1 >= top2
    mask_under = bot1 >= top2

    # 9: shifted down: (top1 > top2 > bot1) & (bot1 > bot2)
    mask_shift_down = ((top1 > top2) & (top2 > bot1)) & (bot1 > bot2)

    # 10: shifted up: (top1 < top2) & (bot1 < bot2 < top1)
    mask_shift_up = (top1 < top2) & ((bot1 < bot2) & (bot2 < top1))

    # 11: above: top1 <= bot2
    mask_above = top1 <= bot2

    # 12: bot1 is nan
    mask_botnan = np.logical_or(np.isnan(bot1), np.isnan(bot2))
    mask_topnan = np.logical_or(np.isnan(top1), np.isnan(top2))
    mask_nan = np.logical_or(mask_botnan, mask_topnan)

    compare = xr.where(mask_eq, 1, compare)
    compare = xr.where(mask_top_within, 2, compare)
    compare = xr.where(mask_bot_within, 3, compare)
    compare = xr.where(mask_within, 4, compare)
    compare = xr.where(mask_outside, 5, compare)
    compare = xr.where(mask_top_oustide, 6, compare)
    compare = xr.where(mask_bot_outside, 7, compare)
    compare = xr.where(mask_under, 8, compare)
    compare = xr.where(mask_shift_down, 9, compare)
    compare = xr.where(mask_shift_up, 10, compare)
    compare = xr.where(mask_above, 11, compare)
    compare = xr.where(mask_nan, 12, compare)

    return compare


def add_regis_to_bottom_of_pwn(pwn_ds, regis_ds):
    """Extend the pwn model by using the regis model for the layers below
    the pwn model.


    Parameters
    ----------
    pwn_ds : xr.DataSet
        lagenmodel van pwn.
    regis_ds : xr.DataSet
        lagenmodel regis.

    Returns
    -------
    pwn_regis_ds : xr.DataSet
        combined model

    """
    lay_count = len(pwn_ds.layer)
    new_bot = pwn_ds["botm"].data.copy()
    new_top = pwn_ds["top"].data.copy()
    new_kh = pwn_ds["kh"].data.copy()
    new_kv = pwn_ds["kv"].data.copy()
    new_layer = pwn_ds["botm"].layer.data.copy()
    lname_regis = []
    for lay in regis_ds.layer:
        mask_lay = pwn_ds["botm"][-1].data > regis_ds["botm"].sel(layer=lay).data
        if mask_lay.any():
            bot_lay = np.where(mask_lay, regis_ds["botm"].sel(layer=lay).data, np.nan)
            kh_lay = np.where(mask_lay, regis_ds["kh"].sel(layer=lay).data, np.nan)
            kv_lay = np.where(mask_lay, regis_ds["kv"].sel(layer=lay).data, np.nan)
            top_lay = new_bot[lay_count - 1]
            new_bot = np.concatenate((new_bot, np.array([bot_lay])))
            new_kh = np.concatenate((new_kh, np.array([kh_lay])))
            new_kv = np.concatenate((new_kv, np.array([kv_lay])))
            new_top = np.concatenate((new_top, np.array([top_lay])))
            lname_regis.append(str(lay.values))
            new_layer = np.append(new_layer, lay_count)
            lay_count += 1
            logger.info(f"adding regis layer {lay.values!s}  to pwn_model layers")

    pwn_regis_ds = xr.Dataset(coords={"x": pwn_ds.x, "y": pwn_ds.y, "layer": new_layer})

    pwn_regis_ds["botm"] = xr.DataArray(
        new_bot,
        dims=("layer", "y", "x"),
        coords={"layer": new_layer, "x": pwn_ds.x, "y": pwn_ds.y},
    )
    pwn_regis_ds["top"] = xr.DataArray(
        new_top,
        dims=("layer", "y", "x"),
        coords={"layer": new_layer, "x": pwn_ds.x, "y": pwn_ds.y},
    )
    pwn_regis_ds["kh"] = xr.DataArray(
        new_kh,
        dims=("layer", "y", "x"),
        coords={"layer": new_layer, "x": pwn_ds.x, "y": pwn_ds.y},
    )
    pwn_regis_ds["kv"] = xr.DataArray(
        new_kv,
        dims=("layer", "y", "x"),
        coords={"layer": new_layer, "x": pwn_ds.x, "y": pwn_ds.y},
    )

    lname_pwn = [f"pwn_lay_{i + 1}" for i in range(len(pwn_ds.layer))]
    lnames_pwn_regis = lname_pwn + lname_regis

    pwn_regis_ds["lnames"] = xr.DataArray(lnames_pwn_regis, dims=("layer"), coords={"layer": new_layer})

    _ = [pwn_regis_ds.attrs.update({key: item}) for key, item in regis_ds.attrs.items()]

    return pwn_regis_ds


def combine_layer_models_regis_pwn(pwn_ds, regis_ds, datadir=None, df_koppeltabel=None):
    """Combine model layers from regis and pwn using a 'koppeltabel'

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
    ValueError
        invalid values in koppeltabel.

    Returns
    -------
    pwn_regis_ds : xr.DataSet
        combined model
    """
    if df_koppeltabel is None:
        fname_koppeltabel = os.path.join(datadir, "pwn_modellagen", "combine_regis_pwn.csv")
        df_koppeltabel = pd.read_csv(fname_koppeltabel, skiprows=1, index_col=4)

    pwn_regis_ds = xr.Dataset(
        coords={
            "x": regis_ds.x.data,
            "y": regis_ds.y.data,
            "layer": df_koppeltabel.index.values,
        }
    )

    _ = [pwn_regis_ds.attrs.update({key: item}) for key, item in regis_ds.attrs.items()]

    empty_da = xr.DataArray(
        dims=("layer", "y", "x"),
        coords={
            "x": regis_ds.x.data,
            "y": regis_ds.y.data,
            "layer": df_koppeltabel.index.values,
        },
    )

    bot = empty_da.copy()
    top = empty_da.copy()
    kh = empty_da.copy()
    kv = empty_da.copy()

    y_mask = [True if y in pwn_ds.y else False for y in regis_ds.y.values]
    x_mask = [True if x in pwn_ds.x else False for x in regis_ds.x.values]

    column_reg_mod = df_koppeltabel.columns[1]
    for i, lay in enumerate(df_koppeltabel.index):
        regis_lay = df_koppeltabel.loc[lay, column_reg_mod]
        pwn_lay = df_koppeltabel.loc[lay, "pwn_lay"]

        logger.info(f"combine regis layer {regis_lay} with pwn layer {pwn_lay}")

        if regis_lay == pwn_lay:
            raise ValueError(f"invalid values encountered, regis layer is {regis_lay} and pwn layer is {pwn_lay}")

        if isinstance(regis_lay, str):
            bot[i] = regis_ds.botm.sel(layer=regis_lay)
            top[i] = regis_ds.top.sel(layer=regis_lay)
            kh[i] = regis_ds.kh.sel(layer=regis_lay)
            kv[i] = regis_ds.kv.sel(layer=regis_lay)
        elif np.isnan(regis_lay):
            bot[i] = np.nan
            top[i] = np.nan
            kh[i] = np.nan
            kv[i] = np.nan
        else:
            raise ValueError("invalid value encountered in regis_lay_nam")

        if isinstance(pwn_lay, str):
            if pwn_lay.isdigit():
                pwn_lay = int(pwn_lay)

        if isinstance(pwn_lay, int):
            # brand pwn modellaag in regis laag
            bot[i, y_mask, x_mask] = pwn_ds.botm.sel(layer=pwn_lay - 1)
            top[i, y_mask, x_mask] = pwn_ds.top.sel(layer=pwn_lay - 1)
            kh[i, y_mask, x_mask] = pwn_ds.kh.sel(layer=pwn_lay - 1)
            kv[i, y_mask, x_mask] = pwn_ds.kv.sel(layer=pwn_lay - 1)
            pwn_final_lay = pwn_lay
        elif pwn_lay == "REGIS":
            # plak REGIS model onder pwn model
            regis_bot = bot[i, y_mask, x_mask]
            regis_top = top[i, y_mask, x_mask]
            regis_kh = kh[i, y_mask, x_mask]
            regis_kv = kv[i, y_mask, x_mask]
            pwn_bot = pwn_ds.botm.sel(layer=pwn_final_lay - 1)

            bot[i, y_mask, x_mask] = xr.where(regis_bot < pwn_bot, regis_bot, np.nan)
            top[i, y_mask, x_mask] = xr.where(regis_top < pwn_bot, regis_top, pwn_bot)
            kh[i, y_mask, x_mask] = xr.where(regis_bot < pwn_bot, regis_kh, np.nan)
            kv[i, y_mask, x_mask] = xr.where(regis_bot < pwn_bot, regis_kv, np.nan)
        elif np.isnan(pwn_lay):
            # maak laag met idomain -1 waar wel regis maar geen pwn laag zit
            bot[i, y_mask, x_mask] = np.nan
            top[i, y_mask, x_mask] = np.nan
            kh[i, y_mask, x_mask] = np.nan
            kv[i, y_mask, x_mask] = np.nan
        else:
            raise ValueError("invalid value encountered in pwn_lay")

    pwn_regis_ds["botm"] = bot
    pwn_regis_ds["top"] = top
    pwn_regis_ds["kh"] = kh
    pwn_regis_ds["kv"] = kv

    return pwn_regis_ds


def unzip_changed_files(zipname, dirname, check_time=True, check_size=False):
    """Extract each file in a zip-file only when the properties are different.
    With the default arguments this method only checks the modification time.

    Parameters
    ----------
    zipname : str
        file name of the zip file
    dirname : str
        extract to this directory
    check_time : bool, optional
        check if file has been updated, by default True
    check_size : bool, optional
        check if file size has changed, by default False
    """
    with zipfile.ZipFile(zipname) as zf:
        infolist = zf.infolist()
        for info in infolist:
            fname = os.path.join(dirname, info.filename)
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
                logger.info(f"extracting {info.filename}")
                zf.extract(info.filename, dirname)
                # set the correct modification time
                # (which is the time of extraction by default)
                tz = time.mktime(info.date_time + (0, 0, -1))
                os.utime(os.path.join(dirname, info.filename), (tz, tz))


def set_ds_grid(ds, extent, delr, delc, refined_extent=None):
    """Set the grid parameters of a model dataset.

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    extent : list, tuple or np.array
        desired model extent (xmin, xmax, ymin, ymax)
    delr : int or float,
        cell size along rows, equal to dx
    delc : int or float,
        cell size along columns, equal to dy
    refined_extent : tuple, list or None, optional
        properties of a local grid refinement, should be (extent, delr, delc).
        The default is None.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    if refined_extent is None:
        xmid, ymid = nlmod.dims.get_xy_mid_structured(extent, delr, delc, descending_y=True)
        ds.attrs["delr"] = delr
        ds.attrs["delc"] = delc
        ds.assign_coords(coords={"x": xmid, "y": ymid})
        ds.attrs["gridtype"] = "structured"
    else:
        extent2, delr2, delc2 = refined_extent
        xmid1, ymid1 = nlmod.dims.get_xy_mid_structured(extent, delr, delc, descending_y=True)
        xmid2, ymid2 = nlmod.dims.get_xy_mid_structured(*refined_extent, descending_y=True)
        xmid = np.concatenate([
            xmid1[xmid1 < extent2[0]],
            xmid2,
            xmid1[xmid1 > extent2[1]],
        ])
        ymid = np.concatenate([
            ymid1[ymid1 > extent2[3]],
            ymid2,
            ymid1[ymid1 < extent2[2]],
        ])

        delr = np.diff(
            np.hstack((
                np.arange(extent[0], extent2[0], delr),
                np.arange(extent2[0], extent2[1], delr2),
                np.arange(extent2[1], extent[1] + delr, delr),
            ))
        )
        delc = np.abs(
            np.diff(
                np.hstack((
                    np.arange(extent[3], extent2[3], -delc),
                    np.arange(extent2[3], extent2[2], -delc2),
                    np.arange(extent2[2], extent[2] - delc, -delc),
                ))
            )
        )

        ds.assign_coords(coords={"x": xmid, "y": ymid})
        ds["delr"] = xr.DataArray(delr, dims=("x"), coords={"x": xmid})
        ds["delc"] = xr.DataArray(delc, dims=("y"), coords={"y": ymid})
        ds.attrs["gridtype"] = "vertex"

    ds.attrs["extent"] = extent

    return ds


@cache.cache_netcdf(coords_2d=True)
def _read_top_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read top of aquitards

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'TS11', 'TS12', 'TS13', 'TS21', 'TS22', 'TS31',
        'TS32' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read top of aquitards")

    ds_out = xr.Dataset()

    for name in ["TS11", "TS12", "TS13", "TS21", "TS22", "TS31", "TS32"]:
        fname = os.path.join(pathname, "laagopbouw", "Top_aquitard", f"{name}.shp")
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix)
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_thickness_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read thickness of aquitards

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'DS11', 'DS12', 'DS13', 'DS21', 'DS22', 'DS31'
        variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read thickness of aquitards")

    ds_out = xr.Dataset()

    # read thickness of aquitards
    for name in ["DS11", "DS12", "DS13", "DS21", "DS22", "DS31"]:
        fname = os.path.join(pathname, "laagopbouw", "Dikte_aquitard", f"{name}.shp")

        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix)
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_kd_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read kd of aquitards

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 's11kd', 's12kd', 's13kd', 's21kd', 's22kd', 's31kd',
        's32kd' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read kd of aquifers")

    ds_out = xr.Dataset()

    # read kD-waarden of aquifers
    for name in ["s11kd", "s12kd", "s13kd", "s21kd", "s22kd", "s31kd", "s32kd"]:
        fname = os.path.join(pathname, "Bodemparams", "KDwaarden_aquitards", f"{name}.shp")
        gdf = gpd.read_file(fname)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_mask_of_aquifers(ds, pathname, length_transition=100.0, ix=None):
    """Read mask of aquifers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with '12', '13', '21', '22' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.
    """
    logging.info("read mask of aquifers")

    ds_out = xr.Dataset()

    # read masks of auifers
    for name in ["12", "13", "21", "22"]:
        key = f"ms{name}kd"
        fname = os.path.join(
            pathname,
            "Bodemparams",
            "Maskers_kdwaarden_aquitards",
            f"masker_aquitard{name}_kd.shp",
        )
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)
        ds_out[key] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest", ix=ix)
        ds_out[f"{key}_mask"] = ~np.isnan(ds_out[key])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{key}_transition"] = in_transition & ~ds_out[f"{key}_mask"]

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_layer_kh(ds, pathname, length_transition=100.0, ix=None):
    """Read hydraulic conductivity of layers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'KW11', 'KW12', 'KW13', 'KW21', 'KW22', 'KW31',
        'KW32' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.

    """
    logging.info("read hydraulic conductivity of layers")

    ds_out = xr.Dataset()

    # read hydraulic conductivity of layers
    for name in ["KW11", "KW12", "KW13", "KW21", "KW22", "KW31", "KW32"]:
        fname = os.path.join(pathname, "Bodemparams", "Kwaarden_aquifers", f"{name}.shp")
        gdf = gpd.read_file(fname)
        ds_out[name] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted")
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_kv_area(ds, pathname, length_transition=100.0, ix=None):
    """Read vertical resistance of layers

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100.
    ix : GridIntersect, optional
        If not provided it is computed from ds.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'C11AREA', 'C12AREA', 'C13AREA', 'C21AREA',
        'C22AREA', 'C31AREA', 'C32AREA' variables.
    ds_out_mask : xr.DataSet
        xarray dataset with True for all cells for which ds_out has valid data.
    ds_out_mask_transition : xr.DataSet
        xarray dataset with True for all cells in the transition zone.
    """
    logging.info("read vertical resistance of layers")

    ds_out = xr.Dataset()

    # read vertical resistance per area
    for name in [
        "C11AREA",
        "C12AREA",
        "C13AREA",
        "C21AREA",
        "C22AREA",
        "C31AREA",
        "C32AREA",
    ]:
        fname = os.path.join(pathname, "Bodemparams", "Cwaarden_aquitards", f"{name}.shp")
        gdf = gpd.read_file(fname)
        gdf.geometry = gdf.buffer(0)

        # some overlying shapes give different results when aggregated with
        # nearest. Remove underlying shape to get same results as Triwaco
        if name == "C13AREA":
            gdf2 = gdf.copy()
            for i in [7, 8, 12, 13]:
                gdf2.geometry = [geom.difference(gdf.loc[i, "geometry"]) for geom in gdf.geometry.values]
                gdf2.loc[i, "geometry"] = gdf.loc[i, "geometry"]
            gdf = gdf2
        elif name == "C21AREA":
            geom_1 = gdf.loc[1].geometry.difference(gdf.loc[4].geometry)
            gdf.loc[1, "geometry"] = geom_1
        elif name == "C22AREA":
            gdf2 = gdf.copy()
            for i in [6, 8, 9]:
                gdf2.geometry = [geom.difference(gdf.loc[i, "geometry"]) for geom in gdf.geometry.values]
                gdf2.loc[i, "geometry"] = gdf.loc[i, "geometry"]
            gdf = gdf2

        ds_out[name] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest", ix=ix)

        nanmask = np.isnan(ds_out[name])
        if name == "C11AREA":
            ds_out[name].values[nanmask] = 1.0
        else:
            ds_out[name].values[nanmask] = 10.0

        ds_out[f"{name}_mask"] = xr.ones_like(ds_out[name], dtype=bool)
        ds_out[f"{name}_transition"] = xr.zeros_like(ds_out[name], dtype=bool)

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_topsysteem(ds, pathname):
    """Read topsysteem

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'mvpolder', 'MVdtm', 'mvDTM', 'gempeil',
        'TOP', 'codesoort', 'draindiepte' variables.

    """
    logging.info("read topsysteem")

    ds_out = xr.Dataset()

    # read surface level
    fname = os.path.join(pathname, "Topsyst", "mvpolder2007.shp")
    gdf = gpd.read_file(fname)
    ds_out["mvpolder"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    fname = os.path.join(pathname, "Topsyst", "MVdtm2007.shp")
    gdf = gpd.read_file(fname)
    ds_out["MVdtm"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
    ds_out["mvDTM"] = ds_out["MVdtm"]  # both ways are used in expressions

    fname = os.path.join(pathname, "Topsyst", "gem_polderpeil2007.shp")
    gdf = gpd.read_file(fname)
    gdf.geometry = gdf.buffer(0)
    ds_out["gempeil"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted")

    # determine the top of the groundwater system
    top = ds_out["gempeil"].copy()
    # use nearest interpolation to fill gaps
    top = nlmod.dims.fillnan_da(top, ds=ds, method="nearest")
    ds_out["TOP"] = top

    fname = os.path.join(pathname, "Topsyst", "codes_voor_typedrainage.shp")
    gdf = gpd.read_file(fname)
    gdf.geometry = gdf.buffer(0)
    ds_out["codesoort"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    fname = os.path.join(pathname, "Topsyst", "diepte_landbouw_drains.shp")
    gdf = gpd.read_file(fname)
    ds_out["draindiepte"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_zout(ds, pathname2, m):
    """Read zout

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    m : flopy.modflow
        modflow model.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'zout', 'ZOUTDEF', variables.

    """
    logging.info("read zout")

    ds_out = xr.Dataset()

    # read data from BasisbestandenNHDZmodelPWN_CvG_20180910
    fname = os.path.join(pathname2, "Zout", "diepte_grensvlak_zoet_brakofzout.shp")
    gdf = gpd.read_file(fname)
    gdf.geometry = gdf.buffer(0)
    ds_out["zout"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    fname = os.path.join(pathname2, "Zout", "Zout_definitie_dichtheidondergrensvlak.shp")
    gdf = gpd.read_file(fname)
    zoutdef_arr = nlmod.dims.interpolate_gdf_to_array(gdf, m, field="VALUE", method="linear")
    ds_out["ZOUTDEF"] = ("y", "x"), zoutdef_arr

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_bodemparams(ds, pathname2):
    """Read bodemparams

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'DWATBorGebruiken', 'Bor_s3p1_C', 'Bor_w3p1_C'
        variables.

    """
    logging.info("read bodem parameters")

    ds_out = xr.Dataset()

    fname = os.path.join(pathname2, "Bodemparams", "Cwaarden_aquitards", "vlakborgebruiken.shp")
    gdf = gpd.read_file(fname)
    ds_out["DWATBorGebruiken"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
    fname = os.path.join(pathname2, "Bodemparams", "Cwaarden_aquitards", "DWAT_Boringen_Selectie.shp")

    gdf = gpd.read_file(fname)
    mask = gdf["PAKKET"] == "s3.1"
    ds_out["Bor_s3p1_C"] = nlmod.dims.gdf_to_da(gdf.loc[mask], ds, column="CLAAG", agg_method="nearest")
    mask = gdf["PAKKET"] == "w3.1"
    ds_out["Bor_w3p1_C"] = nlmod.dims.gdf_to_da(gdf.loc[mask], ds, column="KDLAAG", agg_method="nearest")

    return ds_out


@cache.cache_netcdf(coords_2d=True)
def _read_fluzo(ds, datadir):
    """Read fluzo

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    m : flopy.modflow
        modflow model.

    Returns
    -------
    ds_out : xr.DataSet
        xarray dataset with 'GWA20022015' variables.

    """
    logging.info("read fluzo")

    ds_out = xr.Dataset()

    # read FLUZO-result
    if datadir is None:
        fname = "data/extracted/SimBasis/_GrondwateraanvullingFLUZO/GWA20022015.ado"
    else:
        fname = os.path.join(datadir, "SimBasis/_GrondwateraanvullingFLUZO/GWA20022015.ado")

    ado = triwaco.read_ado(fname)
    # read grid
    if datadir is None:
        fname = "data/extracted/BasisbestandenNHDZmodelPWN_CvG_20181022/grid.teo"
    else:
        fname = os.path.join(datadir, "Grid", "grid.teo")
    grid = triwaco.read_teo(fname)
    gdf = triwaco.get_node_gdf(grid, extent=ds.extent)
    gdf["GWA20022015"] = ado["GWA20022015"][gdf.index]
    gdf.geometry = gdf.buffer(0)
    ds_out["GWA20022015"] = nlmod.dims.gdf_to_da(gdf, ds, column="GWA20022015", agg_method="area_weighted")

    return ds_out


def evaluate_expressions_from_inifile(ini_fname, ds):
    """Evaluatie expressions in an inifile using the variables in an xarray
    Dataset.


    Parameters
    ----------
    ini_fname : str
        path of the inifile.
    ds : xarray.Dataset
        Model dataset.

    Returns
    -------
    None.

    Notes
    -----
    Sometimes you have to run this function twice to obtain alle the variables
    in the ini file because some variables are used before they are created.

    """
    assert os.path.exists(ini_fname), "inifile does not exist"

    if ini_fname.endswith(".xlsx"):
        df = pd.read_excel(ini_fname, "expressions", engine="openpyxl")
        for expression in df.iloc[:, 1]:
            (expression, ds)
    else:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(ini_fname)
        if True:
            for parameter in config["Expressions"]:
                expression = "{}={}".format(parameter, config["Expressions"][parameter])
                evaluate_expression(expression, ds)
        else:
            for parameter in config["Parameters"]:
                if config["Parameters"][parameter].split(",")[5] == "Expression":
                    expression = "{}={}".format(parameter, config["Expressions"][parameter])
                    evaluate_expression(expression, ds)


def evaluate_expression(expression, ds):
    """
    Evaluates an expression on the data

    Parameters
    ----------
    expression : string
        Expression of TriWaCo
    ds: xarray dataset
        model data

    # for example:
    expression = 'RL2=IF(TH1-DS11<TH1-0.1,TH1-DS11,TH1-0.1)'
    expression = 'RL3=TH2-DS12'

    """

    def add_var(var):
        var = var.strip()
        if len(var) == 0:
            return ""
        if var.lower() == "if":
            return "xr.where"
        if is_number(var):
            return var
        # the data is stored in a model dataset
        return f'ds["{var.strip()}"]'

    cmd = ""
    var = ""
    for x in expression:
        if x not in ["(", ")", "+", "-", "*", "/", ",", "&", "=", "!", "<", ">"]:
            var = var + x
        else:
            cmd = cmd + add_var(var)
            cmd = cmd + x
            var = ""
    cmd = cmd + add_var(var)
    # replace && by ()&()
    if "&&" in cmd:
        ind = cmd.index("&&")
        # add a parethesis before
        for i in range(ind - 1, -1, -1):
            if cmd[i] in ["(", ")", "+", "-", "*", "/", ",", "&"]:
                break
        cmd = f"{cmd[: i + 1]}({cmd[i + 1 :]}"
        # add a parethesis after
        for i in range(ind + 3, len(cmd)):
            if cmd[i] in ["(", ")", "+", "-", "*", "/", ",", "&"]:
                break
        cmd = f"{cmd[:i]}){cmd[i:]}"
        cmd = cmd.replace("&&", ")&(")
    try:
        exec(cmd)
        print(f"Expression succeeded: {cmd}")
    except KeyError as e:
        print(f"Expression failed: {cmd}")
        print(type(e), e)


def is_number(s):
    # test if a string s represents a number
    try:
        float(s)
        return True
    except ValueError:
        return False


def fill_gaps(top, x, y, method="nearest"):
    mask = np.isnan(top)
    if np.any(mask):
        points = np.vstack((x[~mask], y[~mask])).T
        values = top[~mask]
        xi = np.vstack((x[mask], y[mask])).T
        top[mask] = scipy.interpolate.griddata(points, values, xi, method=method)
    return top


def shp2grid2(
    fname,
    mgrid=None,
    grid_ix=None,
    fields=None,
    method=None,
    progressbar=False,
    verbose=False,
):
    # read file
    assert os.path.exists(fname), "file does not exist"
    if isinstance(fname, str):
        if verbose:
            print(f"Reading file {fname}")
        shp = gpd.read_file(fname)
    else:
        shp = fname

    # convert to grid data
    # POINT
    if shp.type.iloc[0] == "Point":
        # geometry are points
        # use nearest interpolation
        points = np.array([[g.x, g.y] for g in shp.geometry])
        if len(fields) > 1:
            raise ValueError("For point data, only one field can be passed!")
        field = fields[0]
        values = shp[field].values
        xi = np.vstack((mgrid.xcellcenters.flatten(), mgrid.ycellcenters.flatten())).T
        if method is None:
            method = "nearest"
        vals = scipy.interpolate.griddata(points, values, xi, method=method)
        grid = np.reshape(vals, (mgrid.nrow, mgrid.ncol))

    # LINESTRING
    elif shp.type.iloc[0] == "LineString":
        raise (NotImplementedError)

    # POLYGON
    elif shp.type.iloc[0] in ["Polygon", "MultiPolygon"]:
        if method is None:
            method = "center_grid"

        if method in ["weighted_average", "most_common"]:
            gdf = geodataframe_to_grid(
                shp,
                grid_ix=grid_ix,
                mgrid=mgrid,
                keepcols=fields,
                progressbar=progressbar,
            )
            gr = gdf.groupby("cellids")
            aggdata = pd.DataFrame(index=gdf.cellids.unique(), columns=fields)

            for cid, group in tqdm(gr) if progressbar else gr:
                for icol in fields:
                    if method == "weighted_average":
                        area_weighted = (group.area * group.loc[:, icol]).sum() / group.area.sum()
                        aggdata.loc[[cid], icol] = area_weighted
                    elif method == "most_common":
                        most_common = group.area.idxmax()
                        aggdata.loc[[cid], icol] = group.loc[most_common, icol]

            grid = []
            for icol in fields:
                if mgrid is None:
                    igrid = np.full((grid_ix.mfgrid.nrow, grid_ix.mfgrid.ncol), np.nan)
                else:
                    igrid = np.full((mgrid.nrow, mgrid.ncol), np.nan)
                idx = tuple(zip(*aggdata.index))
                igrid[idx] = aggdata.loc[:, icol].values
                grid.append(igrid)

        elif method == "center_grid":
            # the value in each cell is detemined by the polygon
            # in which the cell-center falls
            field = fields[0]
            x = mgrid.xcellcenters.flatten()
            y = mgrid.ycellcenters.flatten()
            grid = np.full((mgrid.nrow, mgrid.ncol), np.NaN)
            for _, pol in shp.iterrows():
                if not pol.geometry.is_valid:
                    polygon = pol.geometry.buffer(0.0)
                else:
                    polygon = pol.geometry
                mask = [polygon.contains(Point(x[i], y[i])) for i in range(len(x))]
                if np.any(mask):
                    grid[np.reshape(mask, (mgrid.nrow, mgrid.ncol))] = pol[field]
        else:
            raise ValueError(f"'{method}' not a valid choice for method!'")

    # UNRECOGNIZED GEOMETRY
    else:
        raise (NotImplementedError(shp.type[0]))

    if len(grid) == 1 and isinstance(grid, list):
        return grid[0]
    return grid


def add_top_system_cell(layer, r, c, e, ci, cd, b, riv_spd, ghb_spd, drn_spd):
    if np.isnan(b) or b < 100.0:  # when b is more than 100 the cell is turned of by the user
        # first the riv or ghb cells
        if not np.isnan(b):
            if e <= b:  # infiltration will only take place if Hp>BD
                ci = 0.0
        if ci > 0.0:
            if np.isnan(b):
                ghb_spd = np.vstack((ghb_spd, [layer, r, c, e, ci]))
            else:
                riv_spd = np.vstack((riv_spd, [layer, r, c, e, ci, b]))
        # the riv and ghb packages assume an equal conductance for infiltration and drainage
        # therefore add drains to incorporate different conductances
        # substract the infiltration conductance
        cd = cd - ci
        if True:
            if cd < 0.0:
                # assume for now that the drainage conductance should allways
                # be higher than the infiltration conductance
                cd = 0.0
        if cd != 0.0:
            if b > e:
                # use the bottom when it is higher than the elevation
                e = b
            drn_spd = np.vstack((drn_spd, [layer, r, c, e, cd]))
    return riv_spd, ghb_spd, drn_spd


def add_point_data_to_riv_gdf(fname, riv_gdf, column, fill_value):
    CD1 = gpd.read_file(fname)
    riv_gdf[column] = np.NaN
    # LINK represents a river-number
    # loop over rivers
    for i in riv_gdf.index:
        mask = CD1["LINK"] == i
        if np.any(mask):
            riv_gdf.loc[i, column] = CD1.loc[mask, "VALUE"].mean()
    riv_gdf.loc[np.isnan(riv_gdf[column]), column] = fill_value

    """
    # loop over CD1
    for i,row in CD1.iterrows():
        mask = riv_gdf.index == row.LINK
        if np.any(mask):
            # check if there is only one level
            assert np.all(np.isnan(riv_gdf.loc[mask,column]))
            riv_gdf.loc[mask,column] = row.VALUE
    # LINK represents a node-number
    rivernumber = []
    for i in range(grid['NUMBER RIVERS']):
        rivernumber.extend([grid['RIVERNUMBER'][i]]*grid['NUMBER NODES/RIVER'][i])
    rivernumber = np.array(rivernumber)
    if False:
        # loop over rivers
        for i in riv_gdf.index:
            overlap = np.intersect1d(grid['LIST RIVER NODES'][rivernumber==i],CD1['LINK'])
            assert np.any(mask)
    else:
        # loop over CD1
        for i,row in CD1.iterrows():
            mask = grid['LIST RIVER NODES'] == row.LINK
            rn = rivernumber[mask]
            assert np.all(np.isnan(riv_gdf.loc[rn,column]) |
                    (riv_gdf.loc[rn,column]==row.VALUE))
            riv_gdf.loc[rivernumber[mask],column] = row.VALUE
    """


def read_polygon_shape(fname, val_new, VALUE="VALUE", make_valid=False):
    gdf = gpd.read_file(fname)
    if make_valid:
        # make invalid polygons valid
        gdf.geometry = gdf.buffer(0)
    # remove all otther columns than 'VALUE' and 'geometry'
    gdf = gdf[[VALUE, "geometry"]]
    # rename 'Value' to user specified name
    gdf = gdf.rename(columns={VALUE: val_new})
    return gdf


def get_bounds_polygon(m):
    # make a polygon of the model boundary
    extent = m.modelgrid.extent
    return nlmod.util.polygon_from_extent(extent)
