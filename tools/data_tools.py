import rasterio
import os
#import kelp_tools as kt
import pickle
import numpy as np
from rasterio.errors import RasterioIOError
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import clear_output
import matplotlib.dates as mdates
import re
from rasterio.plot import show
import requests
from PIL import Image
from io import BytesIO
import csv



def view_img(path):
#granule = 'HLS.L30.T11SKU.2023040T183427.v2.0'
    try:
        files = os.listdir(path)
    except: 
        print("Invalid Path?")
        return
    metadata_file = [f for f in files if re.search(r'metadata\.csv$', f)]
    if metadata_file :
        with open(os.path.join(path, metadata_file[0]), mode='r') as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)  
            values = next(csv_reader) 
        metadata = dict(zip(keys, values))
    print(metadata['SENSING_TIME'])
    urls = metadata['data_vis_url']
    img_urls = urls.strip("[]").replace("'", "").split(", ")
    print(img_urls)
    response = requests.get(img_urls[0])
    img = Image.open(BytesIO(response.content))
    img.show()


def get_stats(array):
    return np.mean(array), np.std(array), np.percentile(array, [25, 75])

def get_pair_change(df, category=None):
    mesma_change = []
    if category is None:
        category = 'mesma'
    for i, pair in df.iterrows():
        if(pair[f'f_{category}'] > pair[f's_{category}']):
            high_pixel =  pair[f'f_{category}']
            low_pixel =  pair[f's_{category}']
        else:
            low_pixel =  pair[f'f_{category}']
            high_pixel = pair[f's_{category}']
        mean = float((high_pixel+low_pixel)/2)
        if(mean == 0 ):
            percent_change= 0
        else:
            percent_change = float(2*(high_pixel-low_pixel)/(low_pixel + high_pixel))
        change = high_pixel-low_pixel
        mesma_change.append([low_pixel,high_pixel,mean,percent_change, change])
    mesma_change=np.stack(mesma_change)
    return mesma_change

def get_extent(transform, width, height):
    return [
        transform[2], 
        transform[2] + width * transform[0], 
        transform[5] + height * transform[4], 
        transform[5]
    ]

def load_processed_img(path, file, bands=None, just_data=False, geo_info=False, tide_current=True, cloud_coverage=True, crop=False, date_return=False):
    try:
        with rasterio.open(os.path.join(path, file), 'r') as src:
            if bands is None:
                data = src.read()
            else:
                data = src.read(bands)
            metadata = src.tags()
            transform = src.transform
            crs = src.crs
            
        if crop:
            data = data[:, 2700:3300, 900:1500]
            # Update the transform for the cropped image
            new_transform = transform * transform.translation(875, 2800)
        else:
            new_transform = transform

    except RasterioIOError as e:
        print(f"Error reading file {file}: {e}")
        return None

    if just_data:
        return data
    if tide_current:
        try:
            tide = float(metadata['TIDE'])
            current = float(metadata['CURRENT'])
        except KeyError:
            print(f'{file} has no TIDE or CURRENT metadata')
            return None
    if cloud_coverage:
        try:
            clouds = float(metadata['CLOUD_COVERAGE'])
        except KeyError:
            print(f'{file} has no CLOUD metadata')
            return None
    return_vals = [data]
    if date_return:
        date = metadata['TIMESTAMP']
        date = date[:24]
        print(date)
        date = date.rstrip('Z')
        date = date[:26]
        print(date)
        date_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
        day_num = date_obj.timestamp() / 86400 
        return_vals.append(day_num)
        return_vals.append(date)
    if tide_current:
        return_vals.append(tide)
        return_vals.append(current)
    if geo_info:
        return_vals.append(new_transform)
        return_vals.append(crs)
    if cloud_coverage:
        return_vals.append(clouds)
    return return_vals


def plot_image_with_coords(data, transform, crs):
    fig, ax = plt.subplots(figsize=(10, 10))
    show((data, 1), transform=transform, ax=ax)
    ax.set_title('Cropped Image with Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
def get_sensor(granule):
    file_data = granule.split('.')
    return file_data[1]

def extract_date(filename):
    parts = filename.split('.')
    if len(parts) <4:
        return None
    date_str = parts[3]
    date = datetime.strptime(date_str, '%Y%jT%H%M%S')
    return date

def group_by_date(filenames, max_days=4, max_pair_size = 100):
    dates_and_files = [(extract_date(filename), filename) for filename in filenames]
    dates_and_files.sort()  # Sort by date

    neighborhood = []
    max_pair_size = 2
    neighbors = []
    last_date = None

    for date, filename in dates_and_files:
        if last_date is None:
            last_date = date
        if last_date is None or (date - last_date).days <= max_days:
            neighbors.append(filename)
            if(len(neighbors) >=max_pair_size):
                neighborhood.append((len(neighbors), neighbors))
                neighbors = [filename]
                last_date = date
        else:
            if (len(neighbors) > 1):
                neighborhood.append((len(neighbors), neighbors))
            neighbors = [filename]
            last_date = date

    if neighbors and len(neighbors) > 1:
         neighborhood.append((len(neighbors), neighbors))
    neighborhood.sort(key=lambda x: x[0], reverse=True)
    return neighborhood

def get_mesma_pixel_sums(path, file1, file2, mesma_residuals=False, crop=False, bands=[5,6], only_overlap=False, kelp_map=None):
    f_data = load_processed_img(path,file1, bands=bands, geo_info=False,cloud_coverage=True, crop=crop, date_return=True)
    if f_data is None:
        return None
    #print(f_data)
    f_img, f_daynum, f_date, f_tide, f_current, f_clouds = f_data
    s_data = load_processed_img(path,file2, bands=bands, geo_info=False,cloud_coverage=True, crop=crop, date_return=True)
    if s_data is None:
        return None
    s_img, s_daynum, s_date, s_tide, s_current, s_clouds = s_data
    # if crop:
    #     s_img = s_img[:,2800:3050,850:1600]
    #     f_img = f_img[:,2800:3050,850:1600]
    
    # Process First Image
    f_mesma = np.array(f_img[1])
    f_mesma = np.where(f_mesma < 5, 0 , f_mesma)
    f_mesma = np.where(f_mesma > 200, 0, f_mesma)

    if only_overlap:
        f_mesma = np.where(s_img[0] == 0, f_mesma, 0)
        f_mesma = np.where(f_img[0] == 0, f_mesma, 0)
    else:
        f_mesma = np.where(s_img[0] == 2, 0, f_mesma)

    f_kelp = np.where(f_img[0] == 0, 1, 0)
    f_kelp = np.where(s_img[0] == 2, 0, f_kelp)

    f_cloud_correction_factor = None
    if kelp_map is not None:
        f_mesma = np.where(kelp_map, f_mesma, 0)
        f_kelp = np.where(kelp_map, f_kelp,0)
        cloud_over_kelp = np.where(f_img[0] == 2,kelp_map, 0)
        clouds_over_kelp_sum = np.sum(cloud_over_kelp)
        kelp_pixels = np.sum(kelp_map)
        f_cloud_correction_factor = (kelp_pixels/(kelp_pixels-clouds_over_kelp_sum)).astype(float)

    f_kelp_pixels = np.sum(f_kelp)
    f_sum = np.sum(f_mesma)
    # Process Second Image
    s_mesma = np.array(s_img[1])
    s_mesma = np.where(s_mesma < 5, 0 , s_mesma)
    s_mesma = np.where(s_mesma > 200, 0, s_mesma)

    f_clouds = np.sum(np.where(f_img[0] ==2, 1, 0))
    s_clouds = np.sum(np.where(s_img[0] ==2,1,0))


    if only_overlap:
        s_mesma = np.where(f_img[0] == 0, s_mesma, 0)
        s_mesma = np.where(s_img[0] == 0, s_mesma, 0)
    else:
        s_mesma = np.where(f_img[0] == 2, 0, s_mesma)

    s_kelp = np.where(s_img[0] == 0, 1, 0)
    s_kelp = np.where(f_img[0] == 2, 0, s_kelp)
    s_cloud_correction_factor= None
    if kelp_map is not None:
        s_mesma = np.where(kelp_map, s_mesma, 0)
        s_kelp = np.where(kelp_map, s_kelp,0)
        cloud_over_kelp = np.where(s_img[0] == 2,kelp_map, 0)
        clouds_over_kelp_sum = np.sum(cloud_over_kelp)
        if(kelp_pixels - clouds_over_kelp_sum == 0):
            s_cloud_correction_factor = 999
        else:
            s_cloud_correction_factor = (kelp_pixels/(kelp_pixels-clouds_over_kelp_sum)).astype(float)

    s_kelp_pixels = np.sum(s_kelp)
    s_sum = np.sum(s_mesma)
    #print(s_sum)
    
    data = [file1, f_daynum, f_sum, f_kelp_pixels,f_current,f_tide,f_clouds,f_cloud_correction_factor, file2, s_daynum,s_sum,s_kelp_pixels,s_current,s_tide, s_clouds, s_cloud_correction_factor]
    if mesma_residuals:
        mesma_res = f_mesma - s_mesma
        return data , mesma_res, f_mesma, s_mesma
    return data

def get_mesma_residuals(path, file1, file2, crop=False, only_overlap=False):
    bands=[5,6]
    f_data = load_processed_img(path,file1, bands=bands, just_data=True)
    if f_data is None:
        return None
    f_img, f_tide, f_current = f_data
    s_data = load_processed_img(path,file2, bands=bands, just_data=True)
    if s_data is None:
        return None
    s_img, s_tide, s_current = s_data
    if crop:
        s_img = s_img[:,2800:3050,850:1600]
        f_img = f_img[:,2800:3050,850:1600]

    f_mesma = np.array(f_img[1])
    f_mesma = np.where(f_mesma < 5, 0 , f_mesma)
    f_mesma = np.where(f_mesma > 200, 0, f_mesma)
    if only_overlap:
        f_mesma = np.where(s_img[0] == 0, f_mesma, 0)
    else:
        f_mesma = np.where(s_img[0] == 2, 0, f_mesma)


    s_mesma = np.array(s_img[1])
    s_mesma = np.where(s_mesma < 5, 0 , s_mesma)
    s_mesma = np.where(s_mesma > 200, 0, s_mesma)
    #
    if only_overlap:
        s_mesma = np.where(f_img[0] == 0, s_mesma, 0)
    else:
        s_mesma = np.where(f_img[0] == 2, 0, s_mesma)


    mesma_res = f_mesma - s_mesma
    return mesma_res, f_mesma, s_mesma

def get_col_keys():

    return ['img1', 'f_date','f_mesma', 'f_kelp_pixels', 'f_current', 'f_tide','f_clouds', 'f_cloud_factor', 'img2', 's_date','s_mesma', 's_kelp_pixels', 's_current','s_tide','s_clouds','s_cloud_factor']

def plot_pair_values(df, show_color=False, color_basis='', color_title='', title1='Image 1',  title2='Image 2', single_color_var=False, vmin=None, vmax=None):
    f_mesma = df['f_mesma'].astype(int)
    s_mesma= df['s_mesma'].astype(int)
    f_kelp = df['f_kelp_pixels'].astype(int)
    s_kelp= df['s_kelp_pixels'].astype(int)

    # Calculating cloud cover percent (your previous task)
    df['cloud_cover_percent'] = df.apply(lambda row: np.max([(1 - 1 / row['f_cloud_factor']) * 100, (1 - 1 / row['s_cloud_factor']) * 100]), axis=1)

    # Assuming show_color, single_color_var, color_basis, vmin, vmax, color_title, title1, title2 are defined
    # Handling colors
    if show_color:
        if single_color_var and color_basis != '':
            colors = df[color_basis].astype(float)
        elif color_basis == '':
            colors = df['f_clouds'].astype(float) - df['s_clouds'].astype(float)
        else:
            colors = df[f'{color_basis}'].astype(float)
        if vmin is None:
            vmin = np.min(colors)
        if vmax is None:
            vmax = np.max(colors)

    # Best fit line and R^2 for mesma
    slope_mesma, intercept_mesma, r_value_mesma, p_value_mesma, std_err_mesma = linregress(f_mesma, s_mesma)
    y_fit_mesma = slope_mesma * np.linspace(f_mesma.min(), f_mesma.max(), 100) + intercept_mesma

    # Best fit line and R^2 for kelp
    slope_kelp, intercept_kelp, r_value_kelp, p_value_kelp, std_err_kelp = linregress(f_kelp, s_kelp)
    y_fit_kelp = slope_kelp * np.linspace(f_kelp.min(), f_kelp.max(), 100) + intercept_kelp

    # Plotting
    plt.figure(figsize=(18, 6))

    # Mesma plot
    plt.subplot(1, 2, 1)
    if show_color:
        scatter_1 = plt.scatter(f_mesma, s_mesma, c=colors, vmin=vmin, vmax=vmax, alpha=1, label=f'R²={r_value_mesma**2:.2f}')
    else:
        scatter_1 = plt.scatter(f_mesma, s_mesma, alpha=1, label=f'R²={r_value_mesma**2:.2f}')
    plt.plot(np.linspace(f_mesma.min(), f_mesma.max(), 100), np.linspace(f_mesma.min(), f_mesma.max(), 100), color='red', label='y = x')
    #plt.plot(np.linspace(f_mesma.min(), f_mesma.max(), 100), y_fit_mesma, color='blue', linestyle='--', label=f'Fit: y={slope_mesma:.2f}x+{intercept_mesma:.2f}, R²={r_value_mesma**2:.2f}')
    # cbar = plt.colorbar(scatter_1)
    # cbar.ax.tick_params(labelsize=14)  # Set the font size for colorbar ticks
    # cbar.set_label(color_title, fontsize=16)
    plt.legend(fontsize=16)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim((0,100000))
    plt.xlim((0,100000))
    plt.gca().tick_params(axis='y', which='major', pad=10)  # Adjust 'pad' value as needed
    plt.gca().tick_params(axis='x', which='major', pad=10)  # Adjust 'pad' value as needed
    plt.xlabel("Image 1 (MESMA Kelp Value)", fontsize=16)
    plt.ylabel("Image 2 (MESMA Kelp Value)", fontsize=16)
    plt.title('Mesma Value Comparison', fontsize=22)

    # Kelp plot
    plt.subplot(1, 2, 2)
    if show_color:
        scatter_2 = plt.scatter(f_kelp, s_kelp, c=colors, vmin=vmin, vmax=vmax, alpha=1, label=f'R²={r_value_kelp**2:.2f}')
    else:
        scatter_2 = plt.scatter(f_kelp, s_kelp, alpha=1, label=f'R²={r_value_kelp**2:.2f}')
    plt.plot(np.linspace(f_kelp.min(), f_kelp.max(), 100), np.linspace(f_kelp.min(), f_kelp.max(), 100), color='red', label='y = x')
    #plt.plot(np.linspace(f_kelp.min(), f_kelp.max(), 100), y_fit_kelp, color='blue', linestyle='--', label=f'Fit: y={slope_kelp:.2f}x+{intercept_kelp:.2f}, R²={r_value_kelp**2:.2f}')
    cbar = plt.colorbar(scatter_2)
    cbar.ax.tick_params(labelsize=14)  # Set the font size for colorbar ticks
    cbar.set_label(color_title, fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel("Image 1 (Number of Pixels) ", fontsize=16)
    plt.ylabel("Image 2 (Number of Pixels)", fontsize=16)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Adjust 'nbins' value as needed
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
    plt.title('Kelp Pixel Count Comparison', fontsize=22)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.gca().tick_params(axis='y', which='major', pad=10)  # Adjust 'pad' value as needed
    plt.gca().tick_params(axis='x', which='major', pad=10)  # Adjust 'pad' value as needed
    plt.show()

    # Print the slope, intercept, and R² values
    print(f"Mesma: slope = {slope_mesma}, intercept = {intercept_mesma}, R² = {r_value_mesma**2}")
    print(f"Kelp: slope = {slope_kelp}, intercept = {intercept_kelp}, R² = {r_value_kelp**2}")

def view_rgb(path, file1, file2, crop=False,  title_1='rgb1', title_2='rgb2'):
    img_1 = load_processed_img(path,file1, bands=[1,2,3,5,6], just_data=True, crop=crop)
    img_2 = load_processed_img(path,file2, bands=[1,2,3,5,6], just_data=True, crop=crop)
    rgb_1 = np.stack([img_1[2], img_1[1], img_1[0]], axis=-1)
    rgb_2 = np.stack([img_2[2], img_2[1], img_2[0]], axis=-1)
    mesma1 = img_1[3]
    mesma2 = img_2[3]
    kelp1 = img_1[4]
    kelp2 = img_2[4]

    plt.figure(figsize=(15, 15)) 
    plt.subplot(3, 2, 1) 
    plt.imshow(rgb_1)
    plt.title(title_1)
    plt.subplot(3, 2, 2) 
    plt.imshow(rgb_2)
    plt.title(title_2)
    plt.subplot(3, 2, 3) 
    plt.imshow(kelp1)
    plt.title(title_1)
    plt.subplot(3, 2, 4) 
    plt.imshow(kelp2)
    plt.title(title_2)
    plt.subplot(3, 2, 5) 
    plt.imshow(mesma1)
    plt.title(title_1)
    plt.subplot(3, 2, 6) 
    plt.imshow(mesma2)
    plt.title(title_2)
    plt.show()

def plot_four(plot1, plot2, plot3, plot4=None, title1='plot 1', title2 = 'plot 2', title3='plot 3', title4='plot4'):
    v_min = np.min([np.min(plot1), np.min(plot2),np.min(plot3)])
    v_max = np.max([np.max(plot1), np.max(plot2),np.max(plot3)])
    plt.figure(figsize=(20,10))
    plt.subplot(2, 2, 1)
    plt.imshow(plot1, vmax = v_max)
    plt.colorbar()
    plt.title(title1)
    plt.subplot(2,2,2)
    plt.imshow(plot2, vmax = v_max)
    plt.colorbar()
    plt.title(title2)
    plt.subplot(2, 2, 3)
    plt.imshow(plot3, vmax = v_max, vmin=-v_max)
    plt.colorbar()
    plt.title(title3)
    if(plot4 is not None):
        plt.subplot(2, 2, 4)
        plt.imshow(plot4)
        plt.colorbar()
        plt.title(title4)
    plt.show()
    
def plot_tide(df):
    tide_diff = df['f_tide'] - df['s_tide']

    mesma_ht = np.where(tide_diff > 0, df['f_mesma'], df['s_mesma'])
    mesma_lt = np.where(tide_diff <= 0, df['f_mesma'], df['s_mesma'])
    tide_diff = abs(tide_diff)
    mesma_diff = (mesma_lt - mesma_ht) / mesma_ht

    plt.figure()
    plt.scatter(tide_diff, mesma_diff)
    plt.title("Water Height difference vs Kelp Detection")
    plt.ylabel("Percent Change in Kelp Biomass Detection")
    plt.xlabel("Difference in Water Height")
    #plt.ylim(0, 5)
    plt.show()

def plot_current(df):
    current_diff = df['f_current'] - df['s_current']

    mesma_hc = np.where(current_diff > 0, df['f_mesma'], df['s_mesma'])
    mesma_lc = np.where(current_diff <= 0, df['f_mesma'], df['s_mesma'])
    tide_diff = abs(current_diff)
    mesma_diff = (mesma_lc - mesma_hc) / mesma_hc

    plt.figure()
    plt.scatter(current_diff, mesma_diff)
    plt.title("Water Height difference vs Kelp Detection")
    plt.ylabel("Percent Change in Kelp Biomass Detection")
    plt.xlabel("Difference in Current Magnitude")
    #plt.ylim(0, 5)
    plt.show()

def plot_tide_current(df):
    # Assuming df is your DataFrame
    current_diff = df['f_current'] - df['s_current']
    mesma_hc = np.where(current_diff > 0, df['f_mesma'], df['s_mesma'])
    mesma_lc = np.where(current_diff <= 0, df['f_mesma'], df['s_mesma'])
    current_diff = abs(current_diff)
    mesma_diff_current = (mesma_lc - mesma_hc) / mesma_hc

    tide_diff = df['f_tide'] - df['s_tide']
    mesma_ht = np.where(tide_diff > 0, df['f_mesma'], df['s_mesma'])
    mesma_lt = np.where(tide_diff <= 0, df['f_mesma'], df['s_mesma'])
    tide_diff = abs(tide_diff)
    mesma_diff_tide = (mesma_lt - mesma_ht) / mesma_ht

    valid_indices_tide = np.isfinite(mesma_diff_tide) & ~np.isnan(mesma_diff_tide)
    valid_indices_current = np.isfinite(mesma_diff_current) & ~np.isnan(mesma_diff_current)

    tide_diff_clean = tide_diff[valid_indices_tide]
    mesma_diff_tide_clean = mesma_diff_tide[valid_indices_tide] * 100

    current_diff_clean = current_diff[valid_indices_current]
    mesma_diff_current_clean = mesma_diff_current[valid_indices_current] * 100
    # print(tide_diff_clean)
    # print(mesma_diff_current_clean)
    # Check if there is variation in the data

    slope_tide, intercept_tide, r_value_tide, p_value_tide, std_err_tide = linregress(tide_diff_clean, mesma_diff_tide_clean)

    slope_current, intercept_current, r_value_current, p_value_current, std_err_current = linregress(current_diff_clean, mesma_diff_current_clean)

    plt.figure(figsize=(18, 6))

    # Plot for Tide Difference vs Kelp Detection
    plt.subplot(1, 2, 1)
    plt.title("Tide Vs Change in Kelp Detection", fontsize=20)
    plt.ylabel("Change in Kelp (%)", fontsize=16)
    plt.xlabel("Difference in Water Height (m) | (High - Low Tide)", fontsize=14)
    plt.scatter(tide_diff_clean, mesma_diff_tide_clean, label=f'R²={r_value_tide**2:.2f}')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # if not np.isnan(r_value_tide):
    #     plt.plot(tide_diff_clean, intercept_tide + slope_tide * tide_diff_clean, 'r', label=f'Fit line, R²={r_value_tide**2:.2f}')
    #     plt.legend()

    # Plot for Current Difference vs Kelp Detection
    plt.subplot(1, 2, 2)
    plt.title("Current Vs Change in Kelp Detection", fontsize=20)
    plt.ylabel("Change in Kelp (%)", fontsize=16)
    plt.xlabel("Difference in Current Magnitude (m/s) | (High - Low Current)", fontsize=14)
    plt.scatter(current_diff_clean, mesma_diff_current_clean, label=f'R²={r_value_current**2:.2f}')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # if not np.isnan(r_value_current):
    #     plt.plot(current_diff_clean, intercept_current + slope_current * current_diff_clean, 'r', label=f'Fit line, R²={r_value_current**2:.2f}')
    #     plt.legend()

    plt.show()


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

def filter_and_sort_files(img_path, granule):
    # Split the granule to get the sensor information
    sensor_bands = get_sensor_bands(granule)
    
    # Get all files in the directory
    img_files = os.listdir(img_path)
    
    # Filter files to keep only those that are in the sensor_bands list
    filtered_files = [f for f in img_files if any(band in f for band in sensor_bands)]
    
    # Remove files that don't match the bands in sensor_bands
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


def analyze_mesma_pixel(path, file1, file2, bands=[5,6], crop=False, residuals=False, kelp_map=None):
    f_data = load_processed_img(path,file1, bands=bands, just_data=True, crop=crop)
    if f_data is None:
        return None
    #print(f_data)
    f_img = f_data
    s_data = load_processed_img(path,file2, bands=bands, just_data=True, crop=crop)
    if s_data is None:
        return None
    s_img = s_data
    
    f_mesma = np.array(f_img[1])
    f_mesma = np.where(f_mesma < 5, 0 , f_mesma) #remove values < 10
    f_mesma = np.where(f_mesma > 200, 0 , f_mesma)
    f_mesma = np.where(s_img[0] == 2, 0, f_mesma) #Remove values that are clouds in other image
    if kelp_map is not None:
        f_mesma = np.where(kelp_map, f_mesma, 0)
    f_sum = np.sum(f_mesma)

    s_mesma = np.array(s_img[1])
    s_mesma = np.where(s_mesma < 5, 0 , s_mesma) #Remove values <10
    s_mesma = np.where(s_mesma > 200, 0 , s_mesma) 
    s_mesma = np.where(f_img[0] == 2, 0, s_mesma) #remove pixels clouded in other image
    if kelp_map is not None:
        s_mesma = np.where(kelp_map, s_mesma, 0)
    s_sum = np.sum(s_mesma)

    s_mesma_binary = np.where(s_mesma > 0 , 1, 0)
    f_mesma_binary = np.where(f_mesma > 0,1 ,0)

    sf_mesma_binary = np.where(s_mesma_binary, f_mesma_binary, 0)

    s_count = np.sum(s_mesma_binary)
    f_count = np.sum(f_mesma_binary)
    sf_count = np.sum(sf_mesma_binary)
    #s_mesma = np.where(s_mesma==0, np.nan, s_mesma)
    #f_mesma = np.where(f_mesma==0, np.nan,f_mesma)
    if residuals:
        resids = f_mesma - s_mesma
        abs_resids = abs(resids)
        resids_sum = np.sum(abs_resids)
        #abs_resids = np.where(resids == 0, np.nan, resids)
        return sf_count, s_sum, f_sum, s_count, f_count, s_mesma, f_mesma, resids, resids_sum
    else:
        return sf_count, s_sum, f_sum, s_count, f_count, s_mesma, f_mesma

def get_mesma_EMs(file='EM_reformatted_dict_v4.pkl', path=r'C:\Users\attic\HLS_Kelp\python_objects'):
    endmember_path = os.path.join(path,file)
    with open(endmember_path, 'rb') as f:
        endmember_dict = pickle.load(f)
    return endmember_dict

def get_granule(filename):
    match = re.match(r'^(.*)_processed\.tif$', filename)
    if match:
        extracted_part = match.group(1)
        return extracted_part
    else:
        print("invalid file name")
        return None
    
def get_image_pixel_sums(path, file, crop=False, bands=[5,6], kelp_map=None , cloud_correction=False, tide_current=False ):
    f_data = load_processed_img(path,file, bands=bands, geo_info=False,cloud_coverage=True, tide_current=tide_current, crop=crop, date_return=True)
    if f_data is None:
        return None
    #print(f_data)
    if tide_current:
        f_img, day_num, date, f_tide, f_current, f_clouds = f_data
    else: 
        f_img, day_num, date, f_clouds = f_data
    # Process First Image
    f_mesma = np.array(f_img[1])
    f_mesma = np.where(f_mesma < 5, 0 , f_mesma)
    f_mesma = np.where(f_mesma > 200, 0, f_mesma)

    f_kelp = np.where(f_img[0] == 0, 1, 0)
    f_kelp = np.where(kelp_map,f_kelp, 0 )
    cloud_correction_factor = None
    if kelp_map is not None:
        f_mesma = np.where(kelp_map, f_mesma, 0)
        f_kelp = np.where(kelp_map, f_kelp,0)
        cloud_over_kelp = np.where(f_img[0] == 2,kelp_map, 0)
        clouds_over_kelp_sum = np.sum(cloud_over_kelp)
        kelp_pixels = np.sum(kelp_map)
        cloud_correction_factor = kelp_pixels/(kelp_pixels-clouds_over_kelp_sum).astype(float)
    f_kelp_pixels = np.sum(f_kelp)
    f_sum = np.sum(f_mesma)
    if cloud_correction and cloud_correction_factor is not None:
        f_sum = f_sum * cloud_correction_factor
    if tide_current:    
        data = [file, day_num, date, f_sum, f_kelp_pixels,f_current,f_tide,f_clouds, cloud_correction_factor]
    else:
        data = [file, day_num, date, f_sum, f_kelp_pixels,f_clouds, cloud_correction_factor]
    return data

def extract_date(filename):
    match = re.search(r'\.(\d{7})T', filename)
    if match:
        date_str = match.group(1)
        date = datetime.strptime(date_str, '%Y%j')
        return date
    return None

def sort_filenames_by_date(filenames):
    date_filename_pairs = [(extract_date(filename), filename) for filename in filenames]
    date_filename_pairs.sort(key=lambda x: x[0])
    sorted_filenames = [filename for _, filename in date_filename_pairs]
    return sorted_filenames

def generate_binary_kelp_map(tile, tile_path=r'C:\Users\attic\HLS Kelp Detection\processed imagery\tiles', version=0, save=True, binary_threshold=10, show_image=False):
    path= os.path.join(tile_path,tile)
    filenames = set(os.listdir(path))
    filenames.discard('kelp_map.tif')
    filenames = list(filenames)
    length = len(filenames)
    image = []
    data, crs, transform = load_processed_img(path,filenames[2],bands=[1], geo_info=True,cloud_coverage=False, tide_current=False)
    for i,file in enumerate(filenames):
        data = load_processed_img(path,file,bands=[1],just_data=True, cloud_coverage=False)
        if data is None:
            continue
        kelp = np.where(data==0, 1,0)
        image.append(kelp)
 
        clear_output()
        print(f'{i+1}/{length}')    

    image = np.array(image)  # Convert list to 3D NumPy array
    summed_image = np.sum(image, axis=0)

    kelp_map = np.where(summed_image > binary_threshold, 1,0)
    print(kelp_map.shape)
    if show_image:
        plt.figure(figsize=(25,25))
        plt.imshow(kelp_map[0])
        plt.show()
        plt.figure(figsize=(25,25))
        plt.imshow(summed_image[0])
        plt.show()
    
    files = os.listdir(path)
    for file in files:
        match = re.match(r'^.*\.tif$', file)
        if match is not None:
            img_file = file
            break
    print(img_file)
    packet = load_processed_img(path,img_file,bands=[1],geo_info=True, tide_current=False, cloud_coverage=False)

    data, transform, crs = packet
    bands,width,height = data.shape
    if save:
        data_type = rasterio.uint8
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 2,  # one band  B02, B03, B04, and B05, classified, mesma (Blue, Green, Red, and NIR).
            'dtype': data_type,  # assuming binary mask, adjust dtype if needed
            'crs': crs,
            'transform': transform,
            'nodata': 0,  # assuming no data is 0
            'tags': { 'VERSION':version }
        }
        try:
            new_path = os.path.join(path,'kelp_map.tif')
            with rasterio.open(new_path, 'w', **profile) as dst:
                dst.write((kelp_map[0]).astype(data_type), 1)
                dst.write((summed_image[0]).astype(data_type), 2)
            print(f'saved to: {new_path}')
        except RasterioIOError as e:
            print(f"Error reading file {file}: {e}")
    return kelp_map[0]


def convert_df_types(df, col_types=None):
    if col_types is None:
        col_types = {
            'kelp_pixels': int,
            'day_num': float,
            'mesma': float,
            'tide': float,
            'clouds': float,
            'current': float,
            'cloud_correction_factor': float,
            'date': 'datetime'
        }
    for col, dtype in col_types.items():
        if dtype == 'datetime':
            df[col] = [datetime.strptime(date_str[:26], "%Y-%m-%dT%H:%M:%S.%f") for date_str in df[col]]
        else:
            df[col] = df[col].astype(dtype)
    return df

