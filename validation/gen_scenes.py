import sys
import site
site_path = site.getusersitepackages()
if site_path not in sys.path:
    sys.path.append(site_path)
import os
import numpy as np
from skimage import io, exposure

import os
import numpy as np
import math

model_name = "test3_norm01_colorsrgb"  
splats = False # True

# ========== CONFIG ==========
# pointcloud, dsm FALSE -> TREE
# dsm TRUE -> DSM
# pointcloud TRUE, dsm FALSE -> POINTCLOUD
pointcloud = True # False
dsm = False # True
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
tree_folder = f"models//{model_name}"
if splats:
    tree_folder = f"models//SPLATS//{model_name}"
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = f"models//{model_name}/pointclouds-landmarks"  
    output_folder = f"outputs/trees-pointclouds/{model_name}"
    if splats:
        tree_folder = f"models//SPLATS//{model_name}//"
        os.makedirs(os.path.join(base_folder, "outputs/trees-pointclouds/SPLATS"), exist_ok=True)
        output_folder = f"outputs/trees-pointclouds/SPLATS/{model_name}"
if dsm: 
    tree_folder = "DSM_OBJ"  
    output_folder = f"outputs/dsm/{model_name}"
ortho_folder = "ORTHOPHOTOS"
texture_folder = "textures"
temp_folder = "temp"
tree_folder = os.path.join(base_folder, tree_folder)
ortho_folder = os.path.join(base_folder, ortho_folder)
texture_folder = os.path.join(base_folder, texture_folder)
output_folder = os.path.join(base_folder, output_folder)
temp_folder = os.path.join(base_folder, temp_folder)
# template_texture = os.path.join(texture_folder, "coniferous.jpg")  # or deciduous.jpg
render_resolution = (220, 220) # (512, 512) # (250, 250)
background_color = (1, 1, 1, 1)  # White RGBA


from osgeo import osr, gdal, gdalconst, ogr
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import numpy as np
def transformCoordinates(src_epsg_code, dst_epsg_code, lon, lat):
    
    # Create spatial reference objects for the source (WGS84) and target coordinate systems
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(src_epsg_code)

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(dst_epsg_code)


    # Create a coordinate transformation object
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    # Perform the transformation
    dst_x, dst_y, _ = transform.TransformPoint(lat, lon)
    
    return dst_x, dst_y

def cropOrthophoto(x, y, offset, wmts_url, output_file, output_width=128, output_height=128):
    # Specify the coordinates for cropping
    min_x = x - offset
    min_y = y - offset
    max_x = x + offset
    max_y = y + offset
    
    # Open the WMTS dataset
    ds = gdal.Open(wmts_url)
    
    # Create a new GeoTIFF dataset for the cropped image
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, output_width, output_height, ds.RasterCount, ds.GetRasterBand(1).DataType)
    
    # Copy the geotransform and projection
    output_ds.SetGeoTransform([min_x, (max_x - min_x) / output_width, 0, max_y, 0, -(max_y - min_y) / output_height])
    output_ds.SetProjection(ds.GetProjection())

    gdal.Warp(output_ds, ds, outputBounds=(min_x, min_y, max_x, max_y), width=output_width, height=output_height)

    # Clean up
    output_ds = None
    ds = None

def pixel_to_latlon(file_path, pixel_x, pixel_y):
    # Open the georeferenced image
    dataset = gdal.Open(file_path)

    # Transform pixel coordinates to map coordinates
    gt = dataset.GetGeoTransform()
    map_x = gt[0] + (pixel_x * gt[1]) + (pixel_y * gt[2])
    map_y = gt[3] + (pixel_x * gt[4]) + (pixel_y * gt[5])

    # Convert map coordinates to latitude and longitude
    spatial_ref = dataset.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(spatial_ref)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)  # EPSG code for WGS84

    transform = osr.CoordinateTransformation(src, target)
    lat, lon, _ = transform.TransformPoint(map_x, map_y)

    return lat, lon

def cropDEMFromIMG(input_img_path, min_x, min_y, max_x, max_y, output_file, width, height):
    # Read .img file
    with rasterio.open(input_img_path) as src:
        # Transform bounds to match source CRS if needed
        if src.crs.to_string() != "EPSG:3857":
            bounds = transform_bounds("EPSG:3857", src.crs, min_x, min_y, max_x, max_y)
        else:
            bounds = (min_x, min_y, max_x, max_y)

        # Compute window to read
        window = from_bounds(*bounds, transform=src.transform)

        # Read windowed data at native resolution
        data = src.read(1, window=window)

        # Get transform for the cropped window
        transform = rasterio.windows.transform(window, src.transform)

        # Write output GeoTIFF
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    print(f"Saved cropped DEM to {output_file}")

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def subtract_dsm_dtm(dsm_path, dtm_path, output_path):
    # Open DSM
    with rasterio.open(dsm_path) as dsm:
        dsm_data = dsm.read(1).astype(np.float32)
        dsm_meta = dsm.meta.copy()
        dsm_transform = dsm.transform
        dsm_crs = dsm.crs

    # Open DTM
    with rasterio.open(dtm_path) as dtm:
        dtm_data = dtm.read(1).astype(np.float32)
        dtm_transform = dtm.transform
        dtm_crs = dtm.crs
        dtm_nodata = dtm.nodata

        # Reproject and resample DTM to match DSM
        reprojected_dtm = np.zeros_like(dsm_data, dtype=np.float32)
        dtm_mask = np.ones_like(dsm_data, dtype=bool)  # Track where DTM has valid values

        reproject(
            source=dtm_data,
            destination=reprojected_dtm,
            src_transform=dtm_transform,
            src_crs=dtm_crs,
            dst_transform=dsm_transform,
            dst_crs=dsm_crs,
            resampling=Resampling.bilinear,
            src_nodata=dtm_nodata,
            dst_nodata=np.nan
        )

        # Create a mask for valid DTM pixels (not NaN after reproject)
        valid_mask = ~np.isnan(reprojected_dtm)

    # Do DSM - DTM where valid, else use DSM
    height_diff = np.where(valid_mask, dsm_data - reprojected_dtm, dsm_data)
    height_diff[height_diff < 0] = 0  # Optional: remove negative values

    # Save the result
    with rasterio.open(output_path, 'w', **dsm_meta) as out:
        out.write(height_diff, 1)

    print(f"Saved height difference (DSM - DTM) to: {output_path}")

def add_dtm_to_chm(chm_path, dtm_path, output_path):
    # Open CHM (canopy height model)
    with rasterio.open(chm_path) as chm:
        chm_data = chm.read(1).astype(np.float32)
        chm_meta = chm.meta.copy()
        chm_transform = chm.transform
        chm_crs = chm.crs

    # Open DTM
    with rasterio.open(dtm_path) as dtm:
        dtm_data = dtm.read(1).astype(np.float32)
        dtm_transform = dtm.transform
        dtm_crs = dtm.crs
        dtm_nodata = dtm.nodata

        # Reproject and resample DTM to match CHM
        reprojected_dtm = np.zeros_like(chm_data, dtype=np.float32)
        reproject(
            source=dtm_data,
            destination=reprojected_dtm,
            src_transform=dtm_transform,
            src_crs=dtm_crs,
            dst_transform=chm_transform,
            dst_crs=chm_crs,
            resampling=Resampling.bilinear,
            src_nodata=dtm_nodata,
            dst_nodata=np.nan
        )

    # Add DTM where valid; otherwise keep CHM value
    valid_mask = ~np.isnan(reprojected_dtm)
    dsm_reconstructed = np.where(valid_mask, chm_data + reprojected_dtm, chm_data)

    # Optional: Clip small values
    dsm_reconstructed[dsm_reconstructed < 0] = 0

    # Save the result
    with rasterio.open(output_path, 'w', **chm_meta) as out:
        out.write(dsm_reconstructed, 1)

    print(f"Saved DSM (CHM + DTM) to: {output_path}")

def extract_dsm_and_ortho_patch(lat, lon, area_meters_dsm, area_meters_dtm, dsm_out_path, ortho_out_path, zoom_level=19, zoom_level_dsm=18, size_tiles=256):
    print(f"Cropping orthophoto at lat/lon: ({lat}, {lon}), area={area_meters_dsm}m²")
    ortho_x, ortho_y = transformCoordinates(4326, 3857, lon, lat)
    wmts_url = f"WMTS:https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml,layer=bmaporthofoto30cm,zoom_level={zoom_level}"
    cropOrthophoto(ortho_x, ortho_y, area_meters_dsm, wmts_url, ortho_out_path, output_width=124, output_height=124)
    ortho_dtm_path = ortho_out_path.replace(".png", "_dtm.png")
    cropOrthophoto(ortho_x, ortho_y, area_meters_dtm, wmts_url, ortho_dtm_path, output_width=124, output_height=124)
    print(f"Orthophoto patch saved to {ortho_out_path}")

    img = Image.open(ortho_out_path)
    width, height = img.size
    nw_lat, nw_lon = pixel_to_latlon(ortho_out_path, 0, height)
    se_lat, se_lon = pixel_to_latlon(ortho_out_path, width, 0)

    ulx, uly = transformCoordinates(4326, 3857, nw_lon, nw_lat)
    lrx, lry = transformCoordinates(4326, 3857, se_lon, se_lat)


    img = Image.open(ortho_out_path)
    width, height = img.size
    nw_lat, nw_lon = pixel_to_latlon(ortho_out_path, 0, height)
    se_lat, se_lon = pixel_to_latlon(ortho_out_path, width, 0)

    ulx, uly = transformCoordinates(4326, 3857, nw_lon, nw_lat)
    lrx, lry = transformCoordinates(4326, 3857, se_lon, se_lat)

    path = 'https://gataki.cg.tuwien.ac.at/raw/Oe_2020/OeRect_01m_gs_31287.img'
    cropDEMFromIMG(path, ulx, lry, lrx, uly, dsm_out_path, width=size_tiles, height=size_tiles)
    # cropDEMBox(path, ulx, lry, lrx, uly, dsm_out_path)

    # DTM 
    img = Image.open(ortho_dtm_path)
    width, height = img.size
    nw_lat, nw_lon = pixel_to_latlon(ortho_dtm_path, 0, height)
    se_lat, se_lon = pixel_to_latlon(ortho_dtm_path, width, 0)
    ulx, uly = transformCoordinates(4326, 3857, nw_lon, nw_lat)
    lrx, lry = transformCoordinates(4326, 3857, se_lon, se_lat)
    img = Image.open(ortho_dtm_path)
    width, height = img.size
    nw_lat, nw_lon = pixel_to_latlon(ortho_dtm_path, 0, height)
    se_lat, se_lon = pixel_to_latlon(ortho_dtm_path, width, 0)
    ulx, uly = transformCoordinates(4326, 3857, nw_lon, nw_lat)
    lrx, lry = transformCoordinates(4326, 3857, se_lon, se_lat)

    dsm_img_path_small = 'https://gataki.cg.tuwien.ac.at/raw/Oe_2020/OeRect_01m_gs_31287.img'
    dsm_path_small = dsm_out_path.replace(".tif", "_dsm_small.tif")
    cropDEMFromIMG(dsm_img_path_small, ulx, lry, lrx, uly, dsm_path_small, width=size_tiles, height=size_tiles)

    dsm_minus_dtm_path = dsm_out_path.replace(".tif", "_minus_dtm.tif")
    subtract_dsm_dtm(
        dsm_path=dsm_out_path,
        dtm_path=dsm_path_small,
        output_path=dsm_minus_dtm_path
    )

    dtm_img_path_small = 'https://gataki.cg.tuwien.ac.at/raw/Oe_2020/OeRect_01m_gt_31287.img'
    dtm_path_small = dsm_out_path.replace(".tif", "_dtm_small.tif")
    cropDEMFromIMG(dtm_img_path_small, ulx, lry, lrx, uly, dtm_path_small, width=size_tiles, height=size_tiles)

    dsm_final_path = dsm_out_path.replace(".tif", "_final.tif")

    add_dtm_to_chm(
        chm_path=dsm_minus_dtm_path,
        dtm_path=dtm_path_small,
        output_path=dsm_final_path
    )

def create_large_dsm_ortho(lat, lon, area_meters_dsm, dsm_out_path, ortho_out_path, zoom_level=19, zoom_level_dsm=18, size_tiles=256):
    print(f"Cropping orthophoto at lat/lon: ({lat}, {lon}), area={area_meters_dsm}m²")
    ortho_x, ortho_y = transformCoordinates(4326, 3857, lon, lat)
    wmts_url = f"WMTS:https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml,layer=bmaporthofoto30cm,zoom_level={zoom_level}"
    cropOrthophoto(ortho_x, ortho_y, area_meters_dsm, wmts_url, ortho_out_path, output_width=124, output_height=124)

    img = Image.open(ortho_out_path)
    width, height = img.size
    nw_lat, nw_lon = pixel_to_latlon(ortho_out_path, 0, height)
    se_lat, se_lon = pixel_to_latlon(ortho_out_path, width, 0)
    ulx, uly = transformCoordinates(4326, 3857, nw_lon, nw_lat)
    lrx, lry = transformCoordinates(4326, 3857, se_lon, se_lat)

    path = 'https://gataki.cg.tuwien.ac.at/raw/Oe_2020/OeRect_01m_gs_31287.img'
    cropDEMFromIMG(path, ulx, lry, lrx, uly, dsm_out_path, width=size_tiles, height=size_tiles)

    dsm_minus_dtm_path = dsm_out_path.replace(".tif", "_large_terrain.tif")
    dsm_path_small = dsm_out_path.replace(".tif", "_final.tif")
    subtract_dsm_dtm(
        dsm_path=dsm_out_path,
        dtm_path=dsm_path_small,
        output_path=dsm_minus_dtm_path
    )


#####

import os
import csv
# Load metadata
csv_path = os.path.join(base_folder, "trees-data.csv")
category_map = {}

output_folder = os.path.join(base_folder, "temp")
os.makedirs(output_folder, exist_ok=True)
DSM_FOLDER = os.path.join(output_folder, "DSM_TIF")
ORTHO_FOLDER = os.path.join(output_folder, "ORTHOPHOTOS")
os.makedirs(DSM_FOLDER, exist_ok=True)
os.makedirs(ORTHO_FOLDER, exist_ok=True)
AREA_METERS_DSM = 400
AREA_METERS_DTM = 35 # 25
ZOOM_LEVEL = 20
ZOOM_LEVEL_DSM = 17
TILE_SIZE = 256
# === SETUP OUTPUT FOLDERS ===
os.makedirs(DSM_FOLDER, exist_ok=True)
os.makedirs(ORTHO_FOLDER, exist_ok=True)
# === PROCESS EACH TREE ===
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tree_id = row["ID"].zfill(3)  # zero-pad to match '001' format
        category = row["Category"]
        category_map[tree_id] = category
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        dsm_tif_path = os.path.join(DSM_FOLDER, f"tree_{tree_id}.tif")
        # dsm_mat_path = os.path.join(DSM_FOLDER, f"tree_{tree_id}.mat")
        ortho_path = os.path.join(ORTHO_FOLDER, f"tree_{tree_id}.png")

        print(f"/n▶ Processing Tree ID {tree_id} at lat={lat}, lon={lon}")

        try:
            extract_dsm_and_ortho_patch(
                lat, lon, AREA_METERS_DSM, AREA_METERS_DTM,
                dsm_out_path=dsm_tif_path,
                ortho_out_path=ortho_path,
                zoom_level=ZOOM_LEVEL,
                zoom_level_dsm=ZOOM_LEVEL_DSM,
                size_tiles=TILE_SIZE
            )

            create_large_dsm_ortho(
                lat, lon, 2000,
                dsm_out_path=dsm_tif_path.replace(".png", "_large.png"),
                ortho_out_path=ortho_path.replace(".png", "_large.png"),
                zoom_level=ZOOM_LEVEL,
                zoom_level_dsm=ZOOM_LEVEL_DSM,
                size_tiles=TILE_SIZE
            )

        except Exception as e:
            print(f"❌ Failed for Tree ID {tree_id}: {e}")