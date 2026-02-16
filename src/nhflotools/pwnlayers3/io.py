import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from nhflotools.pwnlayers2.prepare_data.interpolation_helper_functions import (
    CRS_RD,
    get_point_values,
    interpolate_gdf,
    polyline_from_points,
)

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
fpath_shp = os.path.join(__file__, "..", "noordzee_clip", "noordzee_clip.shp")
gdf_ns = gpd.read_file(fpath_shp)

# Create a list with the names of the subfolders where the interpolation result will be stored
subdirs = ["top_aquitard", "dikte_aquitard", "bot_aquitard"]

# Loop over the layers
fpath_gpkg = os.path.join(__file__, "..", "interpolation_points.gpkg")
for c, layer_name in enumerate(layer_names):
    # Create GeoDataFrames with the data points of the top and thicknesses
    gdf_t = get_point_values(f"T{layer_name}")
    # gdf_t.set_crs(CRS_RD)
    gdf_d = get_point_values(f"D{layer_name}")
    # gdf_d.set_crs(CRS_RD)

    # Read the polygons that indicate the absence of a layer (0.01 m polygons in the Koster (1997) shapefiles)
    fpath_shp = os.path.join(__file__, "..", "dikte_aquitard", f"D{layer_name}", f"D{layer_name}_mask.shp")
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
        __file__,
        "..",
        "top_aquitard",
        f"T{layer_name}",
        f"T{layer_name}_interpolation_points.shp",
    )
    gdf_t.set_crs(CRS_RD)
    # gdf_t.to_file(fpath_shp)

    # # Store the interpolation points (layer thickness) so that they can be visualised in QGIS
    fpath_shp = os.path.join(
        __file__,
        "..",
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