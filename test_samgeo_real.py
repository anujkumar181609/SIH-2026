"""
Step 2 (SAM-Geo half): download a real satellite image tile and run SAM-Geo's
automatic segmentation on it, to verify the model produces sensible building/
object outlines -- not just that it initializes.
"""

import os
from samgeo import SamGeo
from samgeo.common import tms_to_geotiff

bbox = [-122.4194, 37.7749, -122.4174, 37.7769]  # ~San Francisco sample

image_path = "test_tile.tif"
output_path = "test_tile_segmented.tif"
vector_path = "test_tile_footprints.geojson"

print("Downloading satellite image tile...")
tms_to_geotiff(output=image_path, bbox=bbox, zoom=18, source="Satellite", overwrite=True)
print(f"Saved image to {image_path}")

print("Running SAM-Geo segmentation...")
sam = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    automatic=True,
)

sam.generate(image_path, output_path, batch=True, foreground=True, erosion_kernel=(3, 3), mask_multiplier=255)
print(f"Saved segmentation mask to {output_path}")

sam.raster_to_vector(output_path, vector_path)
print(f"Saved vectorized footprints to {vector_path}")

import geopandas as gpd
gdf = gpd.read_file(vector_path)
print(f"Number of detected regions/footprints: {len(gdf)}")
print(gdf.head())