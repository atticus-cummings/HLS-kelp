import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import re
#from scipy.ndimage import binary_dilation, uniform_filter
from rasterio.errors import RasterioIOError
from skimage import io
from pyproj import Transformer
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
#from cuml.model_selection import train_test_split
from scipy.stats import randint
from cupyx.scipy.ndimage import binary_dilation, convolve
import time

import sys
import pickle
import csv
import cupy as cp
import random

# def load_granule_data(path):
#     granules = os.listdir(path)
#     for item in granules:
#         if os.path.isdir(os.path.join(path, item)):
#             img_path = os.path.join(path, item)
#             granule = item
#             break
#     files = os.listdir(img_path)
#     return img_path, granule, files

def get_extent(transform, width, height):
    return [
        transform[2], 
        transform[2] + width * transform[0], 
        transform[5] + height * transform[4], 
        transform[5]
    ]


def quartile_filter(value, quartile):
    return (value > quartile[0]) and value < quartile[1]

def valid_endmember(endmember):
    em_quartile = [(73.0, 220.0), (72.0, 158.0), (-2.0, 51.0), (-25.0, 4.0)]
    return (quartile_filter(endmember[0],em_quartile[0])
    and quartile_filter(endmember[1],em_quartile[1])
    and quartile_filter(endmember[2],em_quartile[2])
    and quartile_filter(endmember[3],em_quartile[3]))


def get_sensor(granule):
    file_data = granule.split('.')
    return file_data[1]

def get_sensor_bands(granule):
    file_data = granule.split('.')
    sensor = file_data[1]
    if sensor == 'L30':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
    else:
        return ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    
def load_rf(rf_path):
    with open(rf_path, 'rb') as f:
        cu_rf = pickle.load(f)
    return cu_rf

def filter_and_sort_files(img_path, granule):

    sensor_bands = get_sensor_bands(granule)
    
    img_files = os.listdir(img_path)
    
    filtered_files = [f for f in img_files if any(band in f for band in sensor_bands)]
    
    final_files = []
    for band in sensor_bands:
        for file in filtered_files:
            if band in file:
                final_files.append(file)
    
    # Sort files based on the order in sensor_bands
    def sort_key(filename):
        for band in sensor_bands:
            if band in filename:
                return sensor_bands.index(band)
        return len(sensor_bands)  # Place files with no band information at the end
    
    sorted_files = sorted(final_files, key=sort_key)
    
    return sorted_files

def valid_granule(img_files, sensor_bands, files, item):
    f_mask = [f for f in files if re.search(r'Fmask\.tif$', f)]
    if not f_mask or len(img_files) != len(sensor_bands):
        print(f"Incomplete or invalid granule: {item}")
        return False
    return True
   
def create_qa_mask(land_mask, img_path):
    files = os.listdir(img_path)
    f_mask = [f for f in files if re.search(r'Fmask\.tif$', f)]
    if not f_mask:
        return False

##==========Fmask Cloud mask==========##
    #bitwise operations are weird. Far outside my comfort zone. Need to take CS33 first.........
    try:
        with rasterio.open(os.path.join(img_path,f_mask[0])) as fmask:
            qa_band = fmask.read(1)
        # qa_bit = (1 << 1) - 1
        # qa_cloud_mask = ((qa_band >> 1) & qa_bit) == 1  # Bit 1 for cloud
        # qa_adjacent_to_cloud_mask = ((qa_band >> 2) & qa_bit) == 1  # Bit 2 for cloud adjacent
        # qa_cloud_shadow = ((qa_band >> 3) & qa_bit) == 1 
        # qa_ice = ((qa_band >> 4) & qa_bit) == 1 
        # qa_aerosol = (((qa_band >> 6) & 1) == 1) & (((qa_band >> 7) & 1) == 1)
        # cloud_mask = qa_cloud_mask | qa_adjacent_to_cloud_mask | qa_cloud_shadow | qa_ice | qa_aerosol#Mask out Clouds and cloud-adjacent pixels 
        cloud_mask = ((qa_band >> 1) & 1) #| ((qa_band >> 2) & 1) | ((qa_band >> 3) & 1) | ((qa_band >> 4) & 1) | (((qa_band >> 6) & 1) & ((qa_band >> 7) & 1))
        #cloud_mask_2D = cloud_mask.reshape(-1).T
    except RasterioIOError as e:
        print(f"Error reading file {f_mask}: {e}")
        return False  # Skip to the next granule if a file cannot be read
    #may not be necessary to mask out the cloud-adjacent pixels 

##========== Determine percentage of ocean covered by clouds ==========##
    cloud_land_mask = cp.asarray(cloud_mask | land_mask)
    cloud_but_not_land_mask = cp.asarray(cloud_mask & ~land_mask)
    num_pixels_cloud_not_land = cp.count_nonzero(cloud_but_not_land_mask)
    num_pixels_not_land = np.count_nonzero(~land_mask)
    percent_cloud_covered = num_pixels_cloud_not_land/num_pixels_not_land
    # print(f'{granule} Percent cloud covered: {percent_cloud_covered}')
    return cloud_land_mask, cloud_but_not_land_mask, percent_cloud_covered

def calculate_local_variance(image_gpu, window_size):

    mean_kernel = cp.ones((window_size, window_size), dtype=cp.float32) / (window_size * window_size)
    local_mean_gpu = convolve(image_gpu.astype(cp.float32), mean_kernel, mode='constant', cval=0.0)

    squared_image_gpu = cp.square(image_gpu.astype(cp.float32))
    mean_squared_gpu = convolve(squared_image_gpu, mean_kernel, mode='constant', cval=0.0)
    local_variance_gpu = mean_squared_gpu - cp.square(local_mean_gpu)
    
    return local_variance_gpu

def reproject_dem_to_hls(hls_path, dem_path='/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff'):
    with rasterio.open(hls_path) as dst:
        hls = dst.read()
        dem = rasterio.open(dem_path)
        if dem.crs != dst.crs:
            reprojected_dem = np.zeros((hls.shape[1], hls.shape[2]), dtype=hls.dtype)
            reproject(
                source=dem.read(),
                destination=reprojected_dem,
                src_transform=dem.transform,
                src_crs=dem.crs,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
                resampling=Resampling.bilinear)
            if reprojected_dem.any():
                return reprojected_dem
            else:
                return False

def generate_land_mask(reprojected_dem, land_dilation=3, show_image=False, as_numpy=True):
    if reprojected_dem.any():
        struct = cp.ones((land_dilation, land_dilation))
        reprojected_dem_gpu = cp.asarray(reprojected_dem)
        land_mask = binary_dilation(reprojected_dem_gpu > 0, structure=struct)
        if as_numpy:
            land_mask = cp.asnumpy(land_mask)
        if show_image:
            plt.figure(figsize=(6, 6))
            if as_numpy:
                plt.imshow(land_mask, cmap='gray')
            else:
                plt.imshow(cp.asnumpy(land_mask))
            plt.show()
        return land_mask
    else:
        print("Something failed, you better go check...")
        sys.exit()

def create_land_mask(hls_path, dem_path='/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff', show_image=False, as_numpy=True):
    reprojected_dem = reproject_dem_to_hls(hls_path, dem_path)
    return generate_land_mask(reprojected_dem, show_image=show_image, as_numpy=as_numpy)

def create_mesma_mask(
    classified_img,
    img,
    land_mask,
    cloud_but_not_land_mask,
    ocean_dilation_size=100,  # Struct size for dilation
    kelp_neighborhood=5,
    min_kelp_count=4,
    kelp_dilation_size=15,
    variance_window_size=15,
    variance_threshold=0.95,
    variance_mask=True
):
    ocean_dilation = cp.ones((ocean_dilation_size, ocean_dilation_size))
    structuring_element = cp.ones((kelp_dilation_size, kelp_dilation_size))
    
    #time_st = time.time()
    classified_img_gpu = cp.array(classified_img)
    land_dilated_gpu = cp.asarray(land_mask)
    #time_val = time.time()
    cloud_not_land_gpu = cp.asarray(cloud_but_not_land_mask)
    #land_dilated_gpu = cp.where(land_mask, True, False)
    clouds_dilated_gpu = cp.where(classified_img_gpu == 2, True, cloud_not_land_gpu)
    land_dilated_gpu = binary_dilation(land_dilated_gpu, structure=ocean_dilation)
    #print(f'land finished: {time.time()-time_val}')

    ocean_dilated_gpu = land_dilated_gpu | clouds_dilated_gpu 

    kelp_dilated_gpu = cp.where(classified_img_gpu == 0, True, False)  # This is expanding the kelp_mask so the TF is reversed
    kernel = cp.ones((kelp_neighborhood, kelp_neighborhood), dtype=cp.int32)

    time_val = time.time()
    kelp_count_gpu = convolve(kelp_dilated_gpu.astype(cp.int32), kernel, mode='constant', cval=0.0)
    #print(f'kelp moving average finished: {time.time()-time_val}')

    kelp_dilated_gpu = cp.where(((~kelp_dilated_gpu) | (kelp_count_gpu <= min_kelp_count)), 0, 1)  # If there's no kelp, or the kelp count is <=4, set pixel == false
    #time_val = time.time()
    kelp_dilated_gpu = binary_dilation(kelp_dilated_gpu, structure=structuring_element)  # I may not want to do this. we'll see
    #print(f'kelp dilation finished: {time.time()-time_val}')
    #time_val = time.time()
    kelp_mask = [None] * 4
    ocean_mask = [None] * 4

    for i in range(4):
        band_data = img[i]
        band_data_gpu = cp.asarray(band_data)  # Ensure it's a CuPy array

        kmask_gpu = cp.where(kelp_dilated_gpu == 1, band_data_gpu, cp.nan)
        if variance_mask:
            local_variance_gpu = calculate_local_variance(band_data_gpu, variance_window_size)
            max_local_variance = cp.percentile(local_variance_gpu, 100 * variance_threshold)
            variance_mask_gpu = cp.where(local_variance_gpu > max_local_variance, cp.nan, band_data_gpu)

        if(variance_mask):
            final_omask_gpu = cp.where((ocean_dilated_gpu == True) | cp.isnan(variance_mask_gpu), cp.nan, band_data_gpu)
        else:
            final_omask_gpu = cp.where((ocean_dilated_gpu == True), cp.nan, band_data_gpu)
        kelp_mask[i] = kmask_gpu
        ocean_mask[i] = final_omask_gpu
    #print(f'kBand masking and variance masking complete: {time.time()-time_val}')
    kelp_mask = cp.stack(kelp_mask, axis=0)
    ocean_mask = cp.stack(ocean_mask,axis=0)
    return kelp_mask, ocean_mask


def normalize_img(img, flatten=True):
    img_2D = img.reshape(img.shape[0], -1).T
    img_sum = img_2D.sum(axis=1)
    #print(img_sum.shape)
    epsilon = 1e-10  
    #mask = img_sum[:, None] != 0
    #mask = cp.broadcast_to(mask, img_2D.shape)
    #print(mask.shape)
    #print(type(mask))
    img_2D_nor = cp.divide(img_2D, img_sum[:, None] + epsilon)#, where=mask)
    img_2D_nor = (img_2D_nor * 255).astype(cp.uint8)
    img_2D_nor = img_2D_nor.T
    if flatten:
        img_data= img_2D_nor.reshape(img_2D_nor.shape[0], -1).T
        return img_data
    return img_2D_nor
   

def select_ocean_endmembers(ocean_mask=None, ocean_data=None, print_average=False, n=30, min_pixels=1000, check_EM=False):
    ocean_EM_n = 0
    if ocean_mask is not None:
        ocean_data = ocean_mask.reshape(ocean_mask.shape[0], -1)
    
    nan_columns = cp.isnan(ocean_data).any(axis=0)  # Remove columns with nan 
    ocean_EM_stack =[]
    filtered_ocean = ocean_data[:, ~nan_columns]
    if len(filtered_ocean[0,:]) < min_pixels:
        print("Too few valid ocean end-members")
        return None
    i = 0
    while len(ocean_EM_stack) < n and i < 3000:
        index = random.randint(0,len(filtered_ocean[0])-1)
        if valid_endmember(filtered_ocean[:,index]) or not check_EM:
            ocean_EM_stack.append(filtered_ocean[:,index])
        i = i+1
    if(len(ocean_EM_stack) < 30):
        print("Invalid ocean EM selection")
        return None

    ocean_EM = cp.stack(ocean_EM_stack, axis=1)
    #print(ocean_EM_array)
    if(print_average):
        average_val = cp.nanmean(filtered_ocean, axis=1)
        average_endmember = cp.nanmean(ocean_EM, axis=1)
        print(f"average EM Val: {average_endmember}")
        print(f"average    Val: {average_val}")
    return ocean_EM

def run_mesma(kelp_mask, ocean_EM, n=30, kelp_EM=[459, 556, 437, 1227], print_status=True):
    height = kelp_mask.shape[1]
    width = kelp_mask.shape[2]
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0], -1)
    
    filtered_kelp_data,filtered_indices,original_indices = reduce_matrix_nan(kelp_data)

    num_pixels = len(filtered_indices)
    frac1 = cp.full((num_pixels, n), cp.nan)
    frac2 = cp.full((num_pixels, n), cp.nan)
    rmse = cp.full((num_pixels, n), cp.nan)

    #kelp_mask = cp.asarray(kelp_mask)
    ocean_EM = cp.asarray(ocean_EM)
    kelp_EM = cp.asarray(kelp_EM)
    #kelp_data = cp.asarray(kelp_data)
    if(print_status):
        print("Running MESMA")
    for k in range(n):
        B = cp.column_stack((ocean_EM[:, k], kelp_EM))
        U, S, Vt = cp.linalg.svd(B, full_matrices=False)
        IS = Vt.T / S
        em_inv = IS @ U.T
        F = em_inv @ filtered_kelp_data
        #print(F)
        model = (F.T @ B.T).T
        resids = (filtered_kelp_data - model) / 10000
        rmse[:, k] = cp.sqrt(cp.mean(resids**2,axis=0 ))
        frac1[:, k] = F[0,:]
        frac2[:, k] = F[1,:]

    rmse = cp.asarray(rmse)
    minVals = cp.nanmin(rmse, axis=1)
    # print(minVals)
    PageIdx = cp.nanargmin(rmse, axis=1)
    rows = cp.arange(rmse.shape[0])

    PageIdx = cp.expand_dims(PageIdx, axis=0)
    Zindex = cp.ravel_multi_index((rows,  PageIdx), dims=rmse.shape)
    Mes2 = frac2.ravel()[Zindex]

    mesma_full = revert_matrix_nan(Mes2,filtered_indices,original_indices)
    min_vals_full = revert_matrix_nan(minVals,filtered_indices,original_indices)
    # mesma_full[filtered_indices] = Mes2
    # min_vals_full[filtered_indices] = minVals

    mesma_2D = mesma_full.reshape(height,width)
    min_vals_2D = min_vals_full.reshape(height,width)
    

    mesma_2D = mesma_2D
    mesma_2D = -0.229 * mesma_2D**2 + 1.449 * mesma_2D - 0.018 #Landsat mesma corrections 
    mesma_2D = cp.clip(mesma_2D, 0, None)  # Ensure no negative values
    mesma_2D = cp.round(mesma_2D * 100).astype(cp.int16)
    return mesma_2D, min_vals_2D
    
def plot_mesma(mesma_array, kelp_mask=None, crop=False):
    kelp_img = cp.asnumpy(kelp_mask).astype(np.float32)
    Mes_array_vis = np.where(mesma_array <5 , np.nan, mesma_array)
    #Mes_array_vis = np.where(Mes_array_vis >70 , np.nan, Mes_array_vis)
    kelp_vis = np.where(kelp_img == 0, np.nan, kelp_img)
    plt.figure(figsize=(12, 12), dpi=400)
    plt.imshow(kelp_img[1,2800:3250, 875:1300], cmap='Greys', alpha=1, extent=extent_latlon, vmax=600)
    plt.imshow(Mes_array_vis[2800:3250, 875:1300], alpha=1, extent=extent_latlon )#, vmax=80)
    cbar = plt.colorbar( shrink=.75)
    cbar.ax.tick_params(labelsize=16)  # Set the font size for colorbar ticks
    cbar.set_label('Kelp Endmember Number Multiple', fontsize=20)
    plt.xlabel('Longitude', fontsize=30)
    plt.ylabel('Latitude', fontsize=30)
    #plt.axis('off')
    plt.title("MESMA Output", fontsize=36)
    plt.xticks([],fontsize=16)
    plt.yticks([],fontsize=16)
    plt.show()

def get_metadata(path, files=None):
    if files==None:
        files = os.listdir(path)
    metadata_file = [f for f in files if re.search(r'metadata\.csv$', f)]
    if metadata_file:
        with open(os.path.join(path, metadata_file[0]), mode='r') as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)  
            values = next(csv_reader) 
        return dict(zip(keys, values))
    

def reduce_matrix_nan(array, zeros=False):
    if zeros:
        non_nan_colums = cp.nonzero(array).any(axis=0)
    else:
        non_nan_columns = ~cp.isnan(array).any(axis=0)
        
    original_indices = cp.arange(array.shape[1])
    filtered_indices = original_indices[non_nan_columns]
    filtered_data = array[:, non_nan_columns]
    return filtered_data, filtered_indices, original_indices
def revert_matrix_nan(array,indices, original_indices):
    array_full = cp.full((original_indices.shape), cp.nan)
    array_full[indices] = array
    return array_full

def cropped_extent(transform, crs, width, height):
        new_transform = transform * rasterio.Affine.translation(875, 2800)
        extent_cropped = get_extent(new_transform,3250-2800,1300-875)
        return extent_cropped, new_transform
def get_lat_lon(extent, transform, crs):
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        # Transform the extent coordinates
        lon_min, lat_min = transformer.transform(extent[0], extent[2])
        lon_max, lat_max = transformer.transform(extent[1], extent[3])
        # Define the new extent in lat/lon
        extent_latlon = [lon_min, lon_max, lat_min, lat_max]
        return extent_latlon



        num_bands = 7
        data_type = rasterio.int16
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': num_bands,  # one band  B02, B03, B04, and B05, classified, mesma (Blue, Green, Red, and NIR).
            'dtype': data_type,  # assuming binary mask, adjust dtype if needed
            'crs': src.crs,
            'transform': src.transform,
            'nodata': 0,  # assuming no data is 0
            'tags': {'TIMESTAMP': metadata['SENSING_TIME'], 'CLOUD_COVERAGE': percent_cloud_covered, 'RF_MODEL': rf_model, 'VIS_LINK': metadata['data_vis_url']}
        }
        if not os.path.isdir(os.path.join(save_to_path, tile)):
            os.mkdir(os.path.join(save_to_path, tile))
        img_path = os.path.join(save_to_path, tile, f'{item}_processed.tif')

        # Write the land mask array to geotiff
        with rasterio.open(img_path, 'w', **profile) as dst:
            dst.write(img[0].astype(data_type), 1)
            dst.write(img[1].astype(data_type), 2)
            dst.write(img[2].astype(data_type), 3)
            dst.write(img[3].astype(data_type), 4)
            dst.write(classified_img_cpu.astype(data_type), 5)
            dst.write(mesma_array.astype(data_type), 6)
            dst.write(min_vals.astype(np.float16), 7)
            dst.update_tags(TIMESTAMP=metadata['SENSING_TIME'], CLOUD_COVERAGE=percent_cloud_covered, RF_MODEL=rf_model, VIS_LINK=metadata['data_vis_url'])