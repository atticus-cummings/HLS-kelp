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
        date = date.rstrip('Z')
        date = date[:26]
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
    if(show_color):
        if(single_color_var and not color_basis == ''):
            colors = df[color_basis].astype(float)
        elif color_basis == '':
            colors= df['f_clouds'].astype(float) - df['s_clouds'].astype(float) 
        else:
            colors= df[f's_{color_basis}'].astype(float) + df[f'f_{color_basis}'].astype(float)

        if vmin == None:
            vmin= np.min(colors)
        if vmax == None:
            vmax = np.max(colors)
    min_val = min(f_mesma.min(), s_mesma.min())
    max_val = max(f_mesma.max(), s_mesma.max())
    x_m = np.linspace(min_val, max_val, 100)
    y_m = x_m

    min_val = min(f_kelp.min(), s_kelp.min())
    max_val = max(f_kelp.max(), s_kelp.max())
    x_k = np.linspace(min_val, max_val, 100)
    y_k = x_k


    slope, intercept = np.polyfit(f_mesma, s_mesma, 1)
    print(slope, intercept)
    y_fit = slope * x_m + intercept
    plt.figure(figsize=(18,6))
    plt.subplot(1, 2, 1) 
    if(show_color):
        scatter_1 = plt.scatter(f_mesma, s_mesma, c=colors, vmin=vmin, vmax=vmax, alpha=1)
    else:
        scatter_1 = plt.scatter(f_mesma, s_mesma)
    plt.plot(x_m, y_m, color='red', label='y = x')
    plt.colorbar(scatter_1, label=color_title)
    plt.legend()
    plt.xlabel(title1)
    plt.ylabel(title2)
    plt.title('Mesma Pixel Summation Comparison')
    plt.subplot(1,2,2)
    if(show_color):
        scatter_2 = plt.scatter(f_kelp, s_kelp, c=colors, vmin=vmin, vmax=vmax, alpha=1)
    else:
        scatter_2 = plt.scatter(f_kelp, s_kelp)
    plt.plot(x_k, y_k, color='red', label='y = x')
    plt.colorbar(scatter_2, label=color_title)
    plt.legend()
    plt.xlabel(title1)
    plt.ylabel(title2)
    plt.title('Classified Pixel Count Comparison')
    plt.show()

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

    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1) 
    plt.title("Water Height difference vs Kelp Detection")
    plt.ylabel("Percent Change in MESMA Value")
    plt.xlabel("Difference in Water Height (m) | (High Tide - Low Tide)")
    plt.scatter(tide_diff, mesma_diff_tide*100, c=current_diff)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.scatter(current_diff, mesma_diff_current*100, c=tide_diff)
    plt.ylabel("Percent Change MESMA Value")
    plt.xlabel("Difference in Current Magnitude (m/s) | (High Current - Low Current)")
    plt.colorbar()
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
    
def get_image_pixel_sums(path, file, crop=False, bands=[5,6], kelp_map=None , cloud_correction=False ):
    f_data = load_processed_img(path,file, bands=bands, geo_info=False,cloud_coverage=True,crop=crop, date_return=True)
    if f_data is None:
        return None
    #print(f_data)
    f_img, day_num, date, f_tide, f_current, f_clouds = f_data
    
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
        
    data = [file, day_num, date, f_sum, f_kelp_pixels,f_current,f_tide,f_clouds, cloud_correction_factor]
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

def sum_kelp(path, filenames, version=0, save=True, binary_threshold=10):
    length = len(filenames)
    image = []
    data, crs, transform = load_processed_img(path,file,bands[1], geo_info=True,cloud_coverage=False, tide_current=False)
    for i,file in enumerate(filenames):
        data = load_processed_img(path,file,bands=[5],just_data=True, cloud_coverage=False)
        if data is None:
            continue
        kelp = np.where(data==0, 1,0)
        image.append(kelp)
 
        clear_output()
        print(f'{i}/{length}')    

    image = np.array(image)  # Convert list to 3D NumPy array
    summed_image = np.sum(image, axis=0)

    kelp_map = np.where(summed_image > binary_threshold, 1,0)
    print(kelp_map.shape)
    plt.figure(figsize=(25,25))
    plt.imshow(kelp_map[0,2500:3600,200:1750])
    plt.show()
    plt.figure(figsize=(25,25))
    plt.imshow(summed_image[0,2500:3600,200:1750])
    plt.show()
    
    files = os.listdir(path)
    for file in files:
        match = re.match(r'^.*\.tif$', file)
        if match is not None:
            img_file = file
            break
    print(img_file)
    packet = load_processed_img(path,img_file,bands=[1],geo_info=True, cloud_coverage=False)

    data, tide, current, transform, crs = packet
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
            new_path = os.path.join(rf'H:\HLS_data\imagery\Isla_vista_kelp\processed_v{version}','kelp_map.tif')
            with rasterio.open(new_path, 'w', **profile) as dst:
                dst.write((kelp_map[0]).astype(data_type), 1)
                dst.write((summed_image[0]).astype(data_type), 2)
            print(f'saved to: {new_path}')
        except RasterioIOError as e:
            print(f"Error reading file {file}: {e}")


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

