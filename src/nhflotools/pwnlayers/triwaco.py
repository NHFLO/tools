# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:52:16 2018

@author: Artesia
"""

import os
from collections import defaultdict

import geopandas as gpd
import nlmod
import numpy as np
import scipy.interpolate
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.strtree import STRtree


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


def read_ado(fname, eof_char=""):
    """read a .ado file from a Triwaco model


    Parameters
    ----------
    fname : str
        file path of .ado file.
    eof_char : str, optional
        character combination that indicates the end of the file.
        The default is ''.

    Returns
    -------
    data : dict
        data from the .ado file.

    """
    data = {}
    with open(fname, "r") as f:
        # ------------------------------------------------------------------------
        f.readline()
        r = read_sets(f, data)
        assert r.strip() == eof_char
    return data


def read_teo(fname):
    """read a .teo file with triwaco grid data

    Parameters
    ----------
    fname : str
        path of the .teo file.

    Returns
    -------
    data : dict
        griddata from the .teo file.

    """
    data = {}
    with open(fname, "r") as f:
        data["identification"] = f.readline().strip()
        for i in range(7):
            p = f.readline().strip().split("=")
            data[p[0].strip()] = int(p[1])
        # ------------------------------------------------------------------------
        f.readline()
        r = read_sets(f, data)
        # check to see if file was read until the end:
        assert r.strip() == "END FILE GRIDFL"
    return data


def read_flo(fname, eof_char="******"):
    data = {}
    with open(fname, "r") as f:
        data["identification"] = f.readline().strip()
        # ------------------------------------------------------------------------
        f.readline()
        p = f.readline().strip().split("=")
        data[p[0].strip()] = p[1]
        p = f.readline().strip().split("=")
        data[p[0].strip()] = int(p[1])
        # ------------------------------------------------------------------------
        f.readline()
        # ------------------------------------------------------------------------
        f.readline()
        r = read_sets(f, data)
        # check to see if file was read until the end:
        assert r.strip() == eof_char
    return data


def read_sets(f, data):
    """read data from a triwaco file


    Parameters
    ----------
    f : _io.TextIOWrapper
        reference to triwaco file.
    data : dict
        dictionary that is filled with data.

    Returns
    -------
    r : str
        last line of the file.

    """
    r = f.readline().strip()
    while r.startswith("*SET*") or r.startswith("*TEXT*"):
        is_set = r.startswith("*SET*")
        if is_set:
            set_name = r[5:].strip("=")
        else:
            set_name = r[6:].strip("=")
        # the next line is allways 2
        r = int(f.readline().strip())
        if r == 1:
            # data is a single constant
            ls = [f.readline().strip()]
            fmt_letter = "F"
        elif r == 2:
            # data is an array
            # read number and format
            r = f.readline()
            p1 = r.strip().split()
            nv = int(p1[0])  # number of values
            fmt = p1[1].strip("()")
            fmt_letter = fmt[np.where([not x.isdigit() for x in fmt])[0][0]]
            p2 = fmt.split(fmt_letter)
            npl = int(p2[0])  # number of values per line
            nc = int(p2[1].split(".")[0])  # number of characters per value
            nl = int(np.ceil(nv / npl))  # number of lines
            ls = []
            for il in range(nl):
                # ls.extend([x for x in f.readline().strip().split()])
                r = f.readline()
                if il == nl - 1:
                    # calculate how many values there are on the last line
                    npl = nv - (nl - 1) * npl
                ls.extend([r[nc * iv : nc * (iv + 1)] for iv in range(npl)])
        if is_set:
            if fmt_letter == "I":
                ls = [int(x) for x in ls]
            else:
                ls = [float(x) for x in ls]
            assert f.readline().strip() == "ENDSET"
        else:
            assert f.readline().strip() == "ENDTEXT"
        data[set_name] = np.array(ls)
        # ------------------------------------------------------------------------
        f.readline()
        r = f.readline().strip()
    return r


def get_node_gdf(grid, extent=None):
    """


    Parameters
    ----------
    grid : TYPE
        DESCRIPTION.
    extent : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    gdf : TYPE
        DESCRIPTION.

    """
    xy = np.vstack((grid["X-COORDINATES NODES"], grid["Y-COORDINATES NODES"])).T
    vor = Voronoi(xy)
    gdf = gpd.GeoDataFrame(geometry=list(voronoi_polygons(vor, 100.0)))
    if extent is not None:
        gdf = gdf.loc[gdf.intersects(nlmod.util.polygon_from_extent(extent))]
    return gdf


def voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p]  # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)  # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)  # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)]  # Index of previous vertex.
        k = region[(inf + 1) % len(region)]  # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            (dir_j,) = ridge_direction[i, j]
            (dir_k,) = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1 :] + region[:inf]]
        extra_edge = [
            voronoi.vertices[j] + dir_j * length,
            voronoi.vertices[k] + dir_k * length,
        ]
        yield Polygon(np.concatenate((finite_part, extra_edge)))


def get_figpaths(figpath):
    # get the figure-path of an input and an output folder
    # and make sure the path exists
    ifigpath = os.path.join(figpath, "input")
    if not os.path.isdir(ifigpath):
        os.makedirs(ifigpath)
    ofigpath = os.path.join(figpath, "output")
    if not os.path.isdir(ofigpath):
        os.makedirs(ofigpath)
    return ifigpath, ofigpath


def get_strtree(gdf):
    polygons = gdf.geometry
    for index, polygon in polygons.iteritems():
        # add the index to the polygon, so we can find it back later
        polygon.index = index
    strtree = STRtree(gdf.geometry)
    return strtree
