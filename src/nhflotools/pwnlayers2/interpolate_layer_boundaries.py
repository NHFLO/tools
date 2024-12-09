import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from nhflotools.pwnlayers2.interpolation_helper_functions import (
    CRS_RD,
    get_point_values,
    interpolate_gdf,
    polyline_from_points,
)

try:
    import pyvista as pv
except ImportError as e:
    msg = "pyvista is not installed. Please install it to run this script."
    raise ImportError(msg) from e

# Define the interpolation grid (to be replaced with the model grid in NHFLO)
xmin, ymin = 95000, 496000
xmax, ymax = 115000, 533000
dx = 100.0
xi = np.arange(xmin, xmax + dx, dx)
yi = np.arange(ymin, ymax + dx, dx)
X, Y = np.meshgrid(xi, yi)

# Create a GeoDataFrame with the points of the interpolation
# grid. The values are set to zero, which get used as helper
# points in the interpolation of the thickness in the areas
# where the layer is reported as absent by Kosten (1997)

pts = gpd.points_from_xy(X.ravel(), Y.ravel())
gdf_pt = gpd.GeoDataFrame(
    geometry=pts,
    data={"value": [0] * len(pts)},
    crs=CRS_RD,
)

# Names of the layers to be interpolated
layer_names = ["S11", "S12", "S13", "S21", "S22", "S31", "S32"]

# Define colours for the 3D plot
cmap_t = mpl.colormaps["Blues"]
colors_t = cmap_t(np.linspace(0.5, 1, len(layer_names)))

cmap_b = mpl.colormaps["Oranges"]
colors_b = cmap_b(np.linspace(0.5, 1, len(layer_names)))

fig, ax = plt.subplots()
# cf = plt.contourf(X, Y, zint_t.reshape(X.shape))
# plt.colorbar(mappable=cf)
# plt.plot(x, y,'o', ms=3, mec='w', mfc='none' )
# plt.axis('square')
# plt.show()

# Define a line for creating a cross section that shows the
# projection of the layer top/bottoms along a line.
ln_xs = polyline_from_points(
    np.array([
        [(xmin + xmax) / 2, ymin, 0],
        [(xmin + xmax) / 2, ymax, 0],
        [(xmin + xmax) / 2, ymax, -15000],
        [(xmin + xmax) / 2, ymin, -15000],
    ])
)

# Create a plotter instance for the 3D plot
plotter = pv.Plotter()
# Same for the cross section
plotter_xs = pv.Plotter()

overlap = np.zeros(X.shape)

da = xr.DataArray(
    data=overlap,
    dims=["lat", "lon"],
    coords={
        "lat": yi,
        "lon": xi,
    },
)

# Load the polygon to fill the nans below the North Sea with nearest neighbour interpolation values
fpath_shp = os.path.join("..", "gis", "kaarten_2024_voor_interpolatie", "noordzee_clip", "noordzee_clip.shp")
gdf_ns = gpd.read_file(fpath_shp)

# Create a list with the names of the subfolders where the interpolation result will be stored
subdirs = ["top_aquitard", "dikte_aquitard", "bot_aquitard"]

# Loop over the layers
fpath_gpkg = os.path.join("..", "gis", "kaarten_2024_voor_interpolatie", "interpolation_points.gpkg")
for c, layer_name in enumerate(layer_names):
    # Create GeoDataFrames with the data points of the top and thicknesses
    gdf_t = get_point_values(f"T{layer_name}")
    # gdf_t.set_crs(CRS_RD)
    gdf_d = get_point_values(f"D{layer_name}")
    # gdf_d.set_crs(CRS_RD)

    # Read the polygons that indicate the absence of a layer (0.01 m polygons in the Koster (1997) shapefiles)
    fpath_shp = os.path.join(
        "..", "gis", "kaarten_2024_voor_interpolatie", "dikte_aquitard", f"D{layer_name}", f"D{layer_name}_mask.shp"
    )
    gdf_msk = gpd.read_file(fpath_shp)
    gdf_msk = gdf_msk[["geometry", "VALUE"]]
    gdf_msk = gdf_msk.rename(columns={"VALUE": "value"})
    gdf_within = gpd.sjoin(gdf_pt, gdf_msk, predicate="within")
    gdf_d = pd.concat([gdf_d, gdf_within[["geometry", "value_left"]]])

    gdf_t = gdf_t.drop_duplicates()
    gdf_d = gdf_d.drop_duplicates()

    # Store the interpolation points (layer top) so that they can be visualised in QGIS
    # Experimental, commented out for the time being
    # gdf_t.to_file(
    #     fpath_gpkg,
    #     driver="GPKG",
    #     mode="a",
    #     layer=layer_name,
    # )

    # gdf_d.to_file(
    #     fpath_gpkg,
    #     driver="GPKG",
    #     mode="a",
    #     layer=layer_name,
    # )

    # # Store the interpolation points (layer top) so that they can be visualised in QGIS
    fpath_shp = os.path.join(
        "..",
        "gis",
        "kaarten_2024_voor_interpolatie",
        "top_aquitard",
        f"T{layer_name}",
        f"T{layer_name}_interpolation_points.shp",
    )
    gdf_t.set_crs(CRS_RD)
    # gdf_t.to_file(fpath_shp)

    # # Store the interpolation points (layer thickness) so that they can be visualised in QGIS
    fpath_shp = os.path.join(
        "..",
        "gis",
        "kaarten_2024_voor_interpolatie",
        "dikte_aquitard",
        f"D{layer_name}",
        f"D{layer_name}_interpolation_points.shp",
    )
    gdf_d.set_crs(CRS_RD)
    # gdf_d.to_file(fpath_shp)

    # Interpolate the top
    zint_t = interpolate_gdf(gdf_pt, gdf_t, gdf_ns=gdf_ns)
    # Interpolate the thickness
    zint_d = interpolate_gdf(gdf_pt, gdf_d, gdf_ns=gdf_ns, gdf_msk=gdf_msk)

    # Check if a mask exists for the Bergen area
    fpath_shp = os.path.join(
        "..",
        "gis",
        "kaarten_2024_voor_interpolatie",
        "dikte_aquitard",
        f"D{layer_name}",
        f"D{layer_name}_mask_bergen_area.shp",
    )
    if os.path.isfile(fpath_shp):
        # Read the shapefile
        gdf_msk_bergen = gpd.read_file(fpath_shp)
        # Check which grid points are within the clipping polygons
        gdf_within = gpd.sjoin(gdf_pt, gdf_msk_bergen, predicate="within")
        # Convert their indices to a list
        idx_msk = gdf_within.index.to_list()
        # Set the interpolated values to NaN
        zint_t[idx_msk] = np.nan
        zint_d[idx_msk] = np.nan

    # Calculate the layer bottom using the interpolated values
    zint_b = zint_t - zint_d

    # Store the interpolated values for visualization in QGIS
    for subdir, zint in zip(subdirs, [zint_t, zint_d, zint_b], strict=False):
        da.values = zint.reshape(X.shape)
        fstem = f"{subdir[0].capitalize()}{layer_name}"
        fpath = os.path.join("..", "gis", "kaarten_2024_geinterpoleerd", subdir, fstem)
        os.makedirs(fpath, exist_ok=True)
        fpath = os.path.join(fpath, f"{fstem}.nc")
        da.to_netcdf(fpath)

    # Determine the areas where the bottom of a layer is below the top of the underlying layer
    if c > 0:
        dz = zint_b0 - zint_t  # Note that zint_b0 is not defined until the first layer has been processed  # noqa: F821
        dz[dz > 0] = np.nan
        da.values = dz.reshape(X.shape)

        fpath = os.path.join(
            "..", "gis", "kaarten_2024_geinterpoleerd", "overlap", f"overlap_{layer_names[c - 1]}_{layer_name}.nc"
        )
        da.to_netcdf(fpath)

    zint_b0 = zint_b  # Store the bottom of the current layer for comparison with the top of the next layer
    # zint_t[np.isnan(zint_t)] = 0
    # zint_b[np.isnan(zint_b)] = 0

    # Add the top to the 3D plot
    grid_t = pv.StructuredGrid(X, Y, zint_t.reshape(X.shape) * 100)
    for i in np.where(np.isnan(zint_t))[0]:
        grid_t.BlankPoint(i)

    plotter.add_mesh(
        grid_t,
        color=colors_t[c],
        style="surface",
        show_edges=False,
        nan_opacity=0,
        # scalars=grid.points[:, -1],
        # scalar_bar_args={'vertical': True},
    )

    # Add the top to the 3D cross section with the projected top and bottom elevations
    line_slice_t = grid_t.slice_along_line(ln_xs)
    plotter_xs.add_mesh(
        line_slice_t,
        line_width=1,
        render_lines_as_tubes=False,
        color=colors_t[c],
    )

    # Add the bottom to the 3D plot
    grid_b = pv.StructuredGrid(X, Y, zint_b.reshape(X.shape) * 100)
    for i in np.where(np.isnan(zint_b))[0]:
        grid_b.BlankPoint(i)

    plotter.add_mesh(
        grid_b,
        color=colors_b[c],
        style="surface",
        show_edges=False,
        nan_opacity=0,
        # scalars=grid.points[:, -1],
        # scalar_bar_args={'vertical': True},
    )

    # Add the bottom to the 3D cross section with the projected top and bottom elevations
    line_slice_b = grid_b.slice_along_line(ln_xs)
    plotter_xs.add_mesh(
        line_slice_b,
        line_width=1,
        render_lines_as_tubes=False,
        color=colors_b[c],
    )

# Activate the 3D plots
plotter.show_grid()
plotter.show()

plotter_xs.add_mesh(ln_xs, line_width=1, color="grey")
plotter_xs.show()
