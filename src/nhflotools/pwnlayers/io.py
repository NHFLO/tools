"""Read PWN data from Mensink and Bergen."""

import logging
import os

import dask
import geopandas as gpd
import nlmod
import numpy as np
import pykrige
import shapely
import xarray as xr
from flopy.utils.gridintersect import GridIntersect
from nlmod import cache
from shapely import MultiPolygon, Polygon, make_valid

logger = logging.getLogger(__name__)

# TODO: mask and transition in read_kv_area


def read_pwn_data2(
    ds=None,
    datadir_mensink=None,
    datadir_bergen=None,
    length_transition=100.0,
    cachedir=None,
    parallel=True,
):
    """Read PWN data from Mensink and Bergen.

    Parameters
    ----------
    ds : xarray Dataset
        model dataset.
    datadir_mensink : str, optional
        directory with modeldata of mensink. The default is None.
    datadir_bergen : str, optional
        directory with modeldata of bergen. The default is None.
    length_transition : float, optional
        length of transition zone, by default 100.
    cachedir : str, optional
        cachedir used to cache files using the decorator
        nlmod.cache.cache_netcdf. The default is None.
    parallel : bool, optional
        If True, much is computed in parallel but the logging is scrambled. The default is True.

    Returns
    -------
    ds : xarray Dataset
        model dataset.
    ds_mask : xarray Dataset
        mask dataset. True where values are valid
    ds_mask_transition : xarray Dataset
        mask dataset. True in transition zone.
    """
    modelgrid = nlmod.dims.grid.modelgrid_from_ds(ds, rotated=False)
    ix = GridIntersect(modelgrid, method="vertex")

    ds_out = xr.Dataset(
        attrs={
            "extent": ds.attrs["extent"],
            "gridtype": ds.attrs["gridtype"],
        }
    )
    if parallel:
        logger.info("Using dask to compute in parallel. Logging is scrambled.")
        functions = []
        out_delayed = []

        if datadir_bergen is not None:
            functions = [
                _read_bergen_c_aquitards,
                _read_bergen_basis_aquitards,
                _read_bergen_thickness_aquitards,
            ]
            out_delayed += [
                dask.delayed(func)(
                    ds,
                    datadir_bergen,
                    length_transition=length_transition,
                    cachedir=cachedir,
                    cachename=f"triw_{func.__name__}",
                    ix=ix,
                )
                for func in functions
            ]

        if datadir_mensink is not None:
            functions = [
                _read_top_of_aquitards,
                _read_thickness_of_aquitards,
                _read_kd_of_aquitards,
                _read_mask_of_aquifers,
                _read_layer_kh,
                _read_kv_area,
            ]
            out_delayed += [
                dask.delayed(func)(
                    ds,
                    datadir_mensink,
                    length_transition=length_transition,
                    cachedir=cachedir,
                    cachename=f"triw_{func.__name__}",
                    ix=ix,
                )
                for func in functions
            ]

        out = dask.compute(out_delayed)[0]

        for outi in out:
            ds_out.update(outi)

    else:
        if datadir_bergen is not None:
            logger.info("Reading PWN data from Bergen")
            functions = [
                _read_bergen_c_aquitards,
                _read_bergen_basis_aquitards,
                _read_bergen_thickness_aquitards,
            ]
            for func in functions:
                logger.info("Gathering PWN layer info with: %s", func.__name__)

                out = func(
                    ds,
                    datadir_bergen,
                    length_transition=length_transition,
                    cachedir=cachedir,
                    cachename=f"triw_{func.__name__}",
                    ix=ix,
                )
                ds_out.update(out)

        if datadir_mensink is not None:
            logger.info("Reading PWN data from Mensink")
            functions = [
                _read_top_of_aquitards,
                _read_thickness_of_aquitards,
                _read_kd_of_aquitards,
                _read_mask_of_aquifers,
                _read_layer_kh,
                _read_kv_area,
            ]
            for func in functions:
                logger.info("Gathering PWN layer info with: %s", func.__name__)
                out = func(
                    ds,
                    datadir_mensink,
                    length_transition=length_transition,
                    cachedir=cachedir,
                    cachename=f"triw_{func.__name__}",
                    ix=ix,
                )
                ds_out.update(out)

    # Add top from ds
    ds_out["top"] = ds["top"]
    ds_out["top_mask"] = xr.ones_like(ds["top"], dtype=bool)
    ds_out["top_transition"] = xr.zeros_like(ds["top"], dtype=bool)

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_bergen_basis_aquitards(
    ds,
    pathname=None,
    length_transition=100.0,
    ix=None,
):
    """Read basis of aquitards.

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100. Incompatible with use_default_values_outside_polygons.
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
    iskrieged = -999.0
    polynames = {
        "1A": "BA1A",
        "1B": "BA1B_pol",
        "1C": "BA1C_pol",
        "1D": "BA1D_pol",
        "q2": "BAq2_pol",
    }
    pointnames = {
        "1A": None,
        "1B": "BA1B_point",
        "1C": "BA1C_point",
        "1D": "BA1D_point",
        "q2": "BAq2_point",
    }

    logging.info("read basis of Bergen aquitards")

    ds_out = xr.Dataset()

    fd = os.path.join(pathname, "Laagopbouw", "Basis_aquitard")

    # Load shapes for which no Kriging is applied
    for name, polyname in polynames.items():
        gdf = gpd.read_file(os.path.join(fd, f"{polyname}.shp"))
        gdf = make_valid_polygons(gdf)

        gdf_fill = gdf[iskrieged != gdf.VALUE]  # -999 is Krieged
        gdf_krieg = gdf[iskrieged == gdf.VALUE]
        array = nlmod.dims.grid.gdf_to_da(gdf_fill, ds, column="VALUE", agg_method="area_weighted", ix=ix)

        # Krieging
        if pointnames[name] is not None:
            # Ignore point indices to suppress warning and is not used here.
            gdf_pts = gpd.read_file(
                os.path.join(fd, f"{pointnames[name]}.shp"),
                columns=["VALUE"],
            )
            ok = pykrige.ok.OrdinaryKriging(
                gdf_pts.geometry.x.values,
                gdf_pts.geometry.y.values,
                gdf_pts.VALUE.values,
                variogram_model="linear",
                verbose=False,
                enable_plotting=False,
            )
            _multipolygon = MultiPolygon(
                gdf_krieg.geometry.explode("geometry", index_parts=True).values
            )  # returns Polygon or MultiPolygon
            _multipolygonl = [g for g in make_valid(_multipolygon).geoms if isinstance(g, MultiPolygon | Polygon)]
            if len(_multipolygonl) != 1:
                msg = "MultiPolygons in multipolygon"
                raise ValueError(msg)
            multipolygon = _multipolygonl[0]

            r = ix.intersect(
                multipolygon,
                contains_centroid=False,
                min_area_fraction=None,
                shapetype="multipolygon",
            ).astype([("icell2d", int), ("ixshapes", "O"), ("areas", float)])
            xpts = ds.x.sel(icell2d=r.icell2d).values
            ypts = ds.y.sel(icell2d=r.icell2d).values
            z, _ = ok.execute("points", xpts, ypts)
            array.loc[{"icell2d": r.icell2d}] = z

        # compute mask and transition zone
        ds_out[f"BER_BA{name}_mask"] = ~np.isnan(array)

        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"BER_BA{name}_transition"] = in_transition & ~ds_out[f"BER_BA{name}_mask"]

        ds_out[f"BER_BA{name}"] = array
    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_bergen_c_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read vertical resistance of layers.

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
    logging.info("read vertical resistance of aquitards")

    ds_out = xr.Dataset()

    # read kD-waarden of aquifers
    for _j, name in enumerate(["1A", "1B", "1C", "1D", "2"]):
        fname = os.path.join(pathname, "Bodemparams", f"C{name}.shp")
        gdf = gpd.read_file(fname)
        gdf = make_valid_polygons(gdf)
        ds_out[f"BER_C{name}"] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
        ds_out[f"BER_C{name}_mask"] = ~np.isnan(ds_out[f"BER_C{name}"])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"BER_C{name}_transition"] = in_transition & ~ds_out[f"BER_C{name}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_bergen_thickness_aquitards(
    ds,
    pathname=None,
    length_transition=100.0,
    ix=None,
):
    """Read thickness of aquitards.

    Parameters
    ----------
    ds : xr.DataSet
        xarray with model data
    pathname : str
        directory with model data.
    length_transition : float, optional
        length of transition zone, by default 100. Incompatible with use_default_values_outside_polygons.
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
    iskrieged = -999.0
    polynames = {
        "1A": "DI1A",
        "1B": "DI1B_pol",
        "1C": "DI1C_pol",
        "1D": "DI1D",
        "q2": "DI2",
    }
    pointnames = {
        "1A": None,
        "1B": "DI1B_point",
        "1C": "DI1C_point",
        "1D": None,
        "q2": None,
    }

    logging.info("read thickness of Bergen aquitards")

    ds_out = xr.Dataset()
    fd = os.path.join(pathname, "Laagopbouw", "Dikte_aquitard")

    # Load shapes for which no Kriging is applied
    for name, polyname in polynames.items():
        gdf = gpd.read_file(os.path.join(fd, f"{polyname}.shp"))
        gdf = make_valid_polygons(gdf)
        gdf_fill = gdf[iskrieged != gdf.VALUE]  # -999 is Krieged
        gdf_krieg = gdf[iskrieged == gdf.VALUE]
        array = nlmod.dims.grid.gdf_to_da(gdf_fill, ds, column="VALUE", agg_method="area_weighted", ix=ix)

        # Krieging
        if pointnames[name] is not None:
            # Ignore point indices to suppress warning and is not used here.
            gdf_pts = gpd.read_file(
                os.path.join(fd, f"{pointnames[name]}.shp"),
                columns=["VALUE"],
            )
            ok = pykrige.ok.OrdinaryKriging(
                gdf_pts.geometry.x.values,
                gdf_pts.geometry.y.values,
                gdf_pts.VALUE.values,
                variogram_model="linear",
                verbose=False,
                enable_plotting=False,
            )
            _multipolygon = MultiPolygon(
                gdf_krieg.geometry.explode("geometry", index_parts=True).values
            )  # returns Polygon or MultiPolygon
            _multipolygonl = [g for g in make_valid(_multipolygon).geoms if isinstance(g, MultiPolygon | Polygon)]

            if len(_multipolygonl) != 1:
                msg = "MultiPolygons in multipolygon"
                raise ValueError(msg)

            multipolygon = _multipolygonl[0]

            r = ix.intersect(multipolygon, contains_centroid=False, min_area_fraction=None).astype([
                ("icell2d", int),
                ("ixshapes", "O"),
                ("areas", float),
            ])
            xpts = ds.x.sel(icell2d=r.icell2d).values
            ypts = ds.y.sel(icell2d=r.icell2d).values
            z, _ = ok.execute("points", xpts, ypts)
            array.loc[{"icell2d": r.icell2d}] = z

        # compute mask and transition zone
        ds_out[f"BER_DI{name}_mask"] = ~np.isnan(array)

        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"BER_DI{name}_transition"] = in_transition & ~ds_out[f"BER_DI{name}_mask"]

        ds_out[f"BER_DI{name}"] = array
    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_top_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read top of aquitards.

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
        gdf = make_valid_polygons(gdf)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix)
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_thickness_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read thickness of aquitards.

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
        gdf = make_valid_polygons(gdf)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted", ix=ix)
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_kd_of_aquitards(ds, pathname, length_transition=100.0, ix=None):
    """Read kd of aquitards.

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
        gdf = make_valid_polygons(gdf)
        ds_out[name] = nlmod.dims.grid.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_mask_of_aquifers(ds, pathname, length_transition=100.0, ix=None):
    """Read mask of aquifers.

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
        gdf = gpd.read_file(fname, columns=["VALUE"])
        gdf = make_valid_polygons(gdf)
        ds_out[key] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest", ix=ix)
        ds_out[f"{key}_mask"] = ~np.isnan(ds_out[key])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{key}_transition"] = in_transition & ~ds_out[f"{key}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_layer_kh(ds, pathname, length_transition=100.0, ix=None):
    """Read hydraulic conductivity of layers.

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
        gdf = make_valid_polygons(gdf)
        ds_out[name] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted")
        ds_out[f"{name}_mask"] = ~np.isnan(ds_out[name])
        in_transition = nlmod.dims.grid.gdf_to_bool_da(gdf, ds, ix=ix, buffer=length_transition)
        ds_out[f"{name}_transition"] = in_transition & ~ds_out[f"{name}_mask"]

    return ds_out


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_kv_area(ds, pathname, length_transition=100.0, ix=None):  # noqa: ARG001
    """Read vertical resistance of layers.

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
        gdf = make_valid_polygons(gdf)

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


@cache.cache_netcdf(datavars=["top"], coords_2d=True)
def _read_topsysteem(ds, pathname):
    """Read topsysteem.

    Not tested yet after delivered by Artesia

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
    gdf = make_valid_polygons(gdf)
    ds_out["mvpolder"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    fname = os.path.join(pathname, "Topsyst", "MVdtm2007.shp")
    gdf = gpd.read_file(fname)
    ds_out["MVdtm"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")
    ds_out["mvDTM"] = ds_out["MVdtm"]  # both ways are used in expressions

    fname = os.path.join(pathname, "Topsyst", "gem_polderpeil2007.shp")
    gdf = gpd.read_file(fname)
    gdf = make_valid_polygons(gdf)
    ds_out["gempeil"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="area_weighted")

    # determine the top of the groundwater system
    top = ds_out["gempeil"].copy()
    # use nearest interpolation to fill gaps
    top = nlmod.dims.fillnan_da(top, ds=ds, method="nearest")
    ds_out["TOP"] = top

    fname = os.path.join(pathname, "Topsyst", "codes_voor_typedrainage.shp")
    gdf = gpd.read_file(fname)
    gdf = make_valid_polygons(gdf)
    ds_out["codesoort"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    fname = os.path.join(pathname, "Topsyst", "diepte_landbouw_drains.shp")
    gdf = gpd.read_file(fname)
    ds_out["draindiepte"] = nlmod.dims.gdf_to_da(gdf, ds, column="VALUE", agg_method="nearest")

    return ds_out


def make_valid_polygons(gdf):
    """Make polygons valid.

    And reduces geometrycollections that consist of not just polygons to
    multipolygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygons.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with valid polygons.

    """
    gdf = gdf.copy()
    gdf.geometry = gdf.make_valid()

    gdf_gc = gdf.loc[gdf.geometry.type == "GeometryCollection"].copy()
    gdf_gc_converted = gdf_gc.geometry.apply(
        lambda x: shapely.geometry.MultiPolygon([gg for gg in x.geoms if isinstance(gg, shapely.geometry.Polygon)])
    )
    gdf.loc[gdf_gc.index, "geometry"] = gdf_gc_converted

    return gdf
