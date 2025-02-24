import geopandas as gpd
import hydropandas as hpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nlmod
import numpy as np
import pandas as pd
import pastas as ps
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points


def get_parameters_perceel(name):
    """Get the parameters for a specific perceel.

    Parameters
    ----------
    name : str
        name of the perceel. Could be 'perceel1', 'perceel2' or 'perceel3'

    Returns
    -------
    meas_point : shapely.geometry.Point
        measurement point
    gdf_aoi : geopandas.GeoDataFrame
        area of interest
    extent : np.array
        extent of the area of interest
    tmin : str
        start date
    tmax : str
        end date
    o : hpd.GroundwaterObs
        groundwater observation
    polderpeil : float
        polderpeil
    K : float
        hydraulic conductivity
    K_drn : float
        drain conductivity
    """
    if name == "perceel1":
        meas_point = Point((126860, 508795))
        gdf_aoi = gpd.GeoDataFrame({"name": ["perceel1"]}, geometry=[meas_point.buffer(1000)], crs="EPSG:28992")

        # time settings
        tmin = "1978-1-1"
        tmax = "1986-1-1"

        o = hpd.GroundwaterObs.from_bro(bro_id="GMW000000046108", tube_nr=1)
        polderpeil = -4.8  # Perceel 1 https://www.hhnk.nl/peilbesluiten
        K = 8.0
        K_drn = 0.1
    elif name == "perceel2":
        meas_point = Point((120660, 506580))
        gdf_aoi = gpd.GeoDataFrame({"name": ["perceel2"]}, geometry=[meas_point.buffer(1000)], crs="EPSG:28992")

        # time settings
        tmin = "1964-1-1"
        tmax = "1973-1-1"

        o = hpd.GroundwaterObs.from_bro(bro_id="GMW000000053917", tube_nr=1)
        polderpeil = -4.7  # Perceel 2 https://www.hhnk.nl/peilbesluiten
        K = 0.6
        K_drn = 0.05
    elif name == "perceel3":
        meas_point = Point((137570, 521991))
        gdf_aoi = gpd.GeoDataFrame({"name": ["perceel3"]}, geometry=[meas_point.buffer(1000)], crs="EPSG:28992")

        # time settings
        tmin = "1980-1-1"
        tmax = "1994-1-1"

        o = hpd.GroundwaterObs.from_bro(bro_id="GMW000000053869", tube_nr=1)
        polderpeil = -2.4
        K = 0.6
        K_drn = 0.05
    else:
        raise ValueError(f"Perceel {name} not found")

    bbox = gdf_aoi.total_bounds
    extent = np.array([bbox[0], bbox[2], bbox[1], bbox[3]])

    gdf_aoi.loc[0, "tmin"] = tmin
    gdf_aoi.loc[0, "tmax"] = tmax

    return meas_point, gdf_aoi, extent, tmin, tmax, o, polderpeil, K, K_drn


def get_nearest_surfacewater_geometries(p, brt):
    """Get the nearest surface water geometries from a point.

    Parameters
    ----------
    p : shapely.geometry.Point
        point of interest

    brt : geopandas.GeoDataFrame
        surface water data

    Returns
    -------
    line_sloot : shapely.geometry.LineString
        nearest surface water line
    line_to_sloot : shapely.geometry.LineString
        line from point to nearest surface water
    psloot : shapely.geometry.Point
        projection of point on nearest surface water
    """
    # distance to nearest surface water
    distances = brt.distance(p)

    # dichtstbijzijnde sloot
    isloot = brt.index[distances.argmin()]
    line_sloot = brt.loc[isloot].geometry

    # verwijder sloot uit brt
    brt.drop(isloot, inplace=True)

    # projectie van punt op dichtstbijzijnde sloot
    psloot = nearest_points(line_sloot, p)[0]

    # lineString van punt naar dichtstbijzijnde sloot
    line_to_sloot = LineString([p, psloot])

    return line_sloot, line_to_sloot, psloot


def get_perpendicular_search_line_from_point(p, psloot, max_slootafstand=400):
    """Get a line perpendicular to the nearest surface-water line.

    Parameters
    ----------
    p : shapely.geometry.Point
        point of interest
    psloot : shapely.geometry.Point
        projection of point on nearest surface water
    max_slootafstand : float, optional
        maximum distance to search for a second surface water line, by default 400m

    Returns
    -------
    shapely.geometry.LineString
        line perpendicular to the nearest surface-water line
    """
    # get line perpendicular to the nearest surface-water line
    dx, dy = psloot.x - p.x, psloot.y - p.y
    f = max_slootafstand / (max(abs(dy), abs(dx)))
    return LineString([p, Point(p.x - (dx * f), p.y - (dy * f))])


def get_3search_lines_from_point(p, psloot, max_slootafstand=400):
    """Get 3 lines in all directions from a point.

    Parameters
    ----------
    p : shapely.geometry.Point
        point of interest
    psloot : shapely.geometry.Point
        projection of point on nearest surface water
    max_slootafstand : float, optional
        maximum distance to search for a second surface water line, by default 400m

    Returns
    -------
    l2 : shapely.geometry.LineString
        line perpendicular to the nearest surface-water line from p with a distance of max_slootafstand
    l3 : shapely.geometry.LineString
        line parallel to the nearest surface-water line from p with a distance of max_slootafstand
    l4 : shapely.geometry.LineString
        line parallel to the nearest surface-water line from p with a distance of max_slootafstand
    """
    # 3 lines in all directions
    dx, dy = psloot.x - p.x, psloot.y - p.y
    f = max_slootafstand / (max(abs(dy), abs(dx)))
    l2 = LineString([p, Point(p.x - (dx * f), p.y - (dy * f))])
    l3 = LineString([p, Point(p.x - (dy * f), p.y + (dx * f))])
    l4 = LineString([p, Point(p.x + (dy * f), p.y - (dx * f))])

    return l2, l3, l4


def get_2nd_closest_surfacewater_geometries(brt, p, l2, l3=None, l4=None, max_slootafstand=400):
    """Get the second closest surface water geometries from a point.

    Parameters
    ----------
    brt : geopandas.GeoDataFrame
        surface water data
    p : shapely.geometry.Point
        point of interest
    l2 : shapely.geometry.LineString
        line perpendicular to the nearest surface-water line
    l3 : shapely.geometry.LineString, optional
        line parallel to the nearest surface-water line, by default None
    l4 : shapely.geometry.LineString, optional
        line parallel to the nearest surface-water line, by default None
    max_slootafstand : float, optional
        maximum distance to search for a second surface water line, by default 400m

    Returns
    -------
    l_sloot2 : shapely.geometry.LineString
        second closest surface water line
    psloot2 : shapely.geometry.Point
        projection of point on second closest surface water
    """
    # create gdf with nearest intersection point for each line and surface water
    gdf_int = gpd.GeoDataFrame()
    for ld in [l2, l3, l4]:
        if ld is not None:
            brt_int = brt.loc[brt.intersects(ld)].copy()
            if brt_int.empty:  # no intersection between line and brt
                continue
            brt_int_points = brt_int.intersection(ld)  # intersection points
            brt_distance_p = brt_int_points.distance(p)  # distance to intersection points
            min_dis = brt_distance_p == brt_distance_p.min()  # not using argmin because it can be multiple values
            brt_nearest_is = brt_int.loc[min_dis].copy()  # brt lines with closest intersection
            brt_nearest_is.loc[:, "line_geom_is"] = ld  # save intersected line
            brt_nearest_is.loc[:, "point_is"] = brt_int_points.loc[min_dis]  # save intersection point
            brt_nearest_is.loc[:, "distance_is"] = brt_distance_p.min()  # save distance to intersection point
            gdf_int = pd.concat([gdf_int, brt_nearest_is], axis=0)  # add info to new gdf

    # if there are no intersection points return None
    if gdf_int.empty:
        print(f"geen geometries gevonden bij max slootafstand van {max_slootafstand}!")

        return None, None

    # find the closest intersection point for all lines
    min_dist_geom = gdf_int.iloc[gdf_int["distance_is"].argmin()]
    l_sloot2 = min_dist_geom["geometry"]
    psloot2 = min_dist_geom["point_is"]

    # if l_sloot2 is een polygon, neem het dichtsbijzijnde punt
    if isinstance(psloot2, (LineString, MultiLineString)):
        psloot2 = nearest_points(p, psloot2)[1]

    return l_sloot2, psloot2


def get_gdf_poi(p, brt, perp=False, centerpoint=None, max_slootafstand=400):
    """Get a geodataframe with points of interest.

    Parameters
    ----------
    p : shapely.geometry.Point
        point of interest
    brt : geopandas.GeoDataFrame
        surface water data
    perp : bool, optional
        search only for a perpendicular line. If False 3 search lines are used to find
        the 2nd closest surface water, by default False.

    Returns
    -------
    gdf_poi : geopandas.GeoDataFrame
        geodataframe with points of interest
    """
    if perp and centerpoint is None:
        raise ValueError("Centerpoint is required when perp is True")

    brt = brt.reset_index(drop=True)  # make sure brt has a unique index and is a copy

    if perp:
        l_sloot1, l1, psloot1 = get_nearest_surfacewater_geometries(centerpoint, brt)
        l2 = get_perpendicular_search_line_from_point(centerpoint, psloot1, max_slootafstand=max_slootafstand)
        l_sloot2, psloot2 = get_2nd_closest_surfacewater_geometries(brt, centerpoint, l2, max_slootafstand=max_slootafstand)
        if psloot2 is None:
            gdf_poi = gpd.GeoDataFrame(
                geometry=[p, l_sloot1, psloot1, l_sloot2, psloot2, centerpoint],
                index=["punt", "sloot 1", "projectie sloot 1", "sloot 2", "projectie sloot 2", "middelpunt perceel"],
                crs="EPSG:28992",
            )
            return gdf_poi
        if isinstance(psloot2, MultiPoint):
            distmp = [p2.distance(p) for p2 in psloot2.geoms]
            psloot2 = psloot2.geoms[np.argmin(distmp)]

        if psloot1.x < psloot2.x:
            xsec_center = LineString([psloot1, psloot2])  # cross section at center point
        else:
            xsec_center = LineString([psloot2, psloot1])

        # create cross section at point parallel to cross section at center point
        _p_proj = xsec_center.interpolate(xsec_center.project(p))
        xsec_p1 = xsec_center.parallel_offset(p.distance(xsec_center), side="right")
        xsec_p2 = xsec_center.parallel_offset(-p.distance(xsec_center), side="right")
        if xsec_p1.distance(p) < xsec_p2.distance(p):
            xsec_p = xsec_p1
        else:
            xsec_p = xsec_p2

        # extend cross section to intersect with surface water
        pl = []
        for line_sloot in [l_sloot1, l_sloot2]:
            if xsec_p.intersects(line_sloot):
                pl.append(xsec_p.intersection(line_sloot))
            else:
                pl.append(nearest_points(xsec_p, line_sloot)[0])
            # make sure pl[-1] is a point not a line
            if isinstance(pl[-1], (LineString, MultiLineString)):
                pl[-1] = nearest_points(pl[-1], line_sloot)[1]

        if pl[0].x < pl[1].x:
            xsec_p = LineString(pl)
        else:
            xsec_p = LineString(pl[::-1])

        gdf_poi = gpd.GeoDataFrame(
            geometry=[p, l_sloot1, psloot1, l_sloot2, psloot2, centerpoint, xsec_center, xsec_p],
            index=[
                "punt",
                "sloot 1",
                "projectie sloot 1",
                "sloot 2",
                "projectie sloot 2",
                "middelpunt perceel",
                "doorsnede middelpunt",
                "doorsnede punt",
            ],
            crs="EPSG:28992",
        )

    else:
        l_sloot1, l1, psloot1 = get_nearest_surfacewater_geometries(p, brt)
        l2, l3, l4 = get_3search_lines_from_point(p, psloot1, max_slootafstand=max_slootafstand)
        l_sloot2, psloot2 = get_2nd_closest_surfacewater_geometries(brt, p, l2, l3, l4, max_slootafstand=max_slootafstand)
        gdf_poi = gpd.GeoDataFrame(
            geometry=[p, l_sloot1, psloot1, l_sloot2, psloot2, l1, l4, l2, l3],
            index=[
                "punt",
                "sloot 1",
                "projectie sloot 1",
                "sloot 2",
                "projectie sloot 2",
                "0",
                "90",
                "180",
                "270",
            ],
            crs="EPSG:28992",
        )

    return gdf_poi


def get_slootafstanden(p, brt, perp=False, max_slootafstand=400):
    """Get the distance to the nearest and 2nd closest surface water.

    Parameters
    ----------
    p : shapely.geometry.Point
        point of interest
    brt : geopandas.GeoDataFrame
        surface water data
    perp : bool, optional
        search only for a perpendicular line. If False 3 search lines are used to find
        the 2nd closest surface water, by default False
    max_slootafstand : float, optional
        maximum distance to search for a second surface water line, by default 400

    Returns
    -------
    slootafstand : float
        distance to the 2nd closest surface water
    afstand_tov_midden : float
        distance to the middle of the two surface water lines

    """
    brt = brt.reset_index(drop=True)  # make sure brt has a unique index and is a copy
    line_sloot, _, psloot = get_nearest_surfacewater_geometries(p, brt)
    if p.distance(line_sloot) < 1e-6:  # point p is on line l
        return np.nan, np.nan

    if perp:
        line_from_sloot = get_perpendicular_search_line_from_point(p, psloot, max_slootafstand=max_slootafstand)
        l_sloot2, psloot2 = get_2nd_closest_surfacewater_geometries(brt, p, line_from_sloot, max_slootafstand=max_slootafstand)
        slootafstand = max_slootafstand if l_sloot2 is None else psloot.distance(psloot2)
    else:
        l2, l3, l4 = get_3search_lines_from_point(p, psloot, max_slootafstand=max_slootafstand)
        l_sloot2, psloot2 = get_2nd_closest_surfacewater_geometries(
            brt, p, l2, l3, l4, max_slootafstand=max_slootafstand
        )
        slootafstand = max_slootafstand if l_sloot2 is None else p.distance(psloot2) + p.distance(psloot)

    afstand_tot_verste_sloot = p.distance(psloot2)  # distance from point to sloot 2
    afstand_tov_midden = afstand_tot_verste_sloot - slootafstand / 2

    return slootafstand, afstand_tov_midden


def get_kraijenhoff_par(x: float, L: float, K: float, D: float, Sy: float, N: float):
    """Get parameters A, a & b for the Pastas Kraijenhoff de Leur from original KdL.

    Parameters
    ----------
    x : int or float
        afstand tot sloot [m].
    L : int or float
        slootafstand, afstand tussen sloten [m].
    K : int or float
        doorlatendheid [m/dag].
    D : int or float
        dikte watervoerende laag [m].
    Sy : int or float
        Bergingscoefficient [-]
    N : _type_
        _description_
        ..

    Returns
    -------
    A, a, b : pastas parameters
    """
    b = x / L
    A = -N * L**2 / (2 * K * D) * (b**2 - (1 / 4))
    a = Sy * L**2 / (np.pi**2 * K * D)
    return A, a, b


def get_kvl_model(oc_knmi, D, polderpeil, kwel=None, oseries=None, ev_factor=None):
    """Get a pastas Kraijenhoff van de Leur model to simulate the groundwater head.

    Parameters
    ----------
    oc_knmi : ObsCollection
        knmi data with precipitation and evaporation
    D : int or float
        dikte watervoerende laag [m].
    polderpeil : int or float
        oppervlaktewaterpeil in de polder
    kwel : float or None
        the kwel flux [mm/day]. Only added if kwel is not None. The default is
        None.
    ev_factor : float or None:
        The evaporation is multiplied by this factor (verdampingsfactor). Only applied
        if not None. The default is None

    Returns
    -------
    ml
        pastas model
    """
    # create model
    prec_indexname = oc_knmi.index[oc_knmi.index.str.startswith("RD")][0]
    if oseries is None:
        oseries = pd.Series(D, index=oc_knmi.loc[prec_indexname, "obs"].resample("D").first().index)

    ml = ps.Model(
        oseries=oseries,
        name="khoff",
    )
    tss = {
        "sample_up": "bfill",
        "sample_down": "mean",
        "fill_nan": 0.0,
        "fill_before": 0.0,
        "fill_after": 0.0,
    }
    prec_indexname = oc_knmi.index[oc_knmi.index.str.startswith("RD")][0]
    if kwel is None:
        prec = oc_knmi.loc[prec_indexname, "obs"]["RD"].resample("D").first()
    else:
        prec = oc_knmi.loc[prec_indexname, "obs"]["RD"].resample("D").first() + kwel / 1000.0

    ev_name = oc_knmi.index[oc_knmi.index.str.startswith("EV24")][0]

    rem = ps.RechargeModel(
        prec=prec,
        evap=oc_knmi.loc[ev_name, "obs"]["EV24"].resample("D").first(),
        rfunc=ps.Kraijenhoff(nterms=100, cutoff=0.9999),
        settings=(tss, tss),
    )

    ml.add_stressmodel(rem)
    ml.set_parameter("constant_d", initial=polderpeil, vary=False)
    if ev_factor is not None:
        ml.set_parameter("recharge_f", initial=-ev_factor, vary=False)
    else:
        ml.set_parameter("recharge_f", initial=-1.0, vary=False)

    return ml


def get_kvl_head(ml, x, L, K, D, Sy, N, tmin, tmax):
    """Get a simulation of the heads using Kraijenhoff van de Leur.

    Parameters
    ----------
    ml : Pastas Model
        Pastas model to do the simulation.
    x : int or float
        afstand tot sloot [m].
    L : int or float
        slootafstand, afstand tussen sloten [m].
    K : int or float
        doorlatendheid [m/dag].
    D : int or float
        dikte watervoerende laag [m].
    Sy : int or float
        Bergingscoefficient [-]
    N : _type_
        _description_
    tmin : str
        start date
    tmax : str
        end date

    Returns
    -------
    pd.Series
        simulated groundwater heads.
    """
    A, a, b = get_kraijenhoff_par(x, L, K, D, Sy, N)

    ml.set_parameter("recharge_A", initial=A, vary=False)
    ml.set_parameter("recharge_a", initial=a, vary=False)
    ml.set_parameter("recharge_b", initial=b, vary=False)

    return ml.simulate(tmin=tmin, tmax=tmax)


def plot_kh_kv(
    ds,
    line,
    layer="layer",
    variables=None,
    zmin=-50.25,
    min_label_area=None,
    cmap=None,
    norm=None,
):
    """Plot de doorlatendheid in een doorsnede.

    Parameters
    ----------
    ds : xarray Dataset
        met kh en kv data
    line : _type_
        doorsnedelijn
    layer : str, optional
        naam van layer dimensie in ds, by default "layer"
    variables : list of str, optional
        variable names, by default None
    zmin : float, optional
        bottom of plot, by default -50.25
    min_label_area : _type_, optional
        _description_, by default None
    cmap : _type_, optional
        colormap, by default None
    norm : _type_, optional
        _description_, by default None
    """
    if variables is None:
        variables = ["kh", "kv"]
    if cmap is None:
        cmap = plt.get_cmap("turbo_r")
    if norm is None:
        boundaries = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    for var in variables:
        f, ax = plt.subplots(figsize=(10, 5))
        cs = nlmod.plot.DatasetCrossSection(ds, line, layer=layer, ax=ax, zmin=zmin)
        pc = cs.plot_array(ds[var], norm=norm, cmap=cmap)
        if min_label_area is not None:
            cs.plot_layers(alpha=0.0, min_label_area=min_label_area)
            cs.plot_grid(vertical=False)
        fmat = mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}")
        nlmod.plot.colorbar_inside(pc, bounds=[0.05, 0.05, 0.02, 0.9], format=fmat)
        nlmod.plot.title_inside(var, ax=ax)
        ax.set_xlabel("afstand langs doorsnede (m)")
        ax.set_ylabel("z (m NAP)")
        f.tight_layout(pad=0.0)


def get_drn_distance(p, brt, draindistance, max_slootafstand=400):
    """Get drain geometries for a point p.

    Parameters
    ----------
    p : Point
        point for which the drain geometries are calculated
    brt : GeoDataFrame
        GeoDataFrame with surface water data
    draindistance : float
        distance between drains
    max_slootafstand : float
        maximum distance to surface water

    Returns
    -------
    drain_dists : array
        array with drain distances
    """
    slootafstand, _ = get_slootafstanden(p, brt, perp=True, max_slootafstand=max_slootafstand)
    if np.isnan(slootafstand) or slootafstand == max_slootafstand:
        return np.nan, np.nan
    drain_dists = np.linspace(0, slootafstand, int(slootafstand / draindistance) + 2)[1:-1]

    return drain_dists, slootafstand


def get_drn_geometries(p, brt, drain_dists, gdf_perceel, max_slootafstand=400):
    """Get drain geometries for a point p.

    Parameters
    ----------
    p : Point
        point for which the drain geometries are calculated
    brt : GeoDataFrame
        GeoDataFrame with surface water data
    drain_dists : array
        array with drain distances
    gdf_perceel : gpd.GeoDataFrame
        GeoDataFrame with the geometry of the parcel
    max_slootafstand : float
        maximum distance to surface water

    Returns
    -------
    gdf_drn : GeoDataFrame
        GeoDataFrame with drain geometries

    """
    brt = brt.reset_index(drop=True)  # make sure brt has a unique index and is a copy
    gdf_poi = get_gdf_poi(p, brt, perp=True, centerpoint=p, max_slootafstand=max_slootafstand)

    if gdf_poi.loc["sloot 2", "geometry"] is None:
        print("cannot fit drain geometry, because there is no second surface water line")
        return gpd.GeoDataFrame()
    psloot2 = gdf_poi.loc["projectie sloot 2", "geometry"]

    line_between_sloten = gdf_poi.loc["doorsnede middelpunt", "geometry"]

    # create drain geometry
    geometries_drn = []
    for drndist in drain_dists:
        p_start = line_between_sloten.interpolate(drndist)
        dx, dy = psloot2.x - p_start.x, psloot2.y - p_start.y
        f = max_slootafstand / (max(abs(dy), abs(dx)))
        ldrn = LineString([
            Point(p_start.x + (dy * f), p_start.y - (dx * f)),
            Point(p_start.x - (dy * f), p_start.y + (dx * f)),
        ])
        geometries_drn.append(ldrn)

    gdf_drn = gpd.GeoDataFrame(geometry=geometries_drn, crs="EPSG:28992")

    # slice with edge of perceel
    gdf_drn["geometry"] = gdf_drn.intersection(gdf_perceel.geometry.values[0])

    return gdf_drn
