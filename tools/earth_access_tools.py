import os
import geopandas as gp
import matplotlib.pyplot as plt
import json
from contextlib import redirect_stdout
import csv
import earthaccess
import xml.etree.ElementTree as ET
import csv

def extract_metadata(granule):
    ## ======= Parse metadata ======= ##
    json_str = json.dumps(granule.__dict__)
    metadata = json.loads(json_str) 
    meta = metadata['render_dict']['meta']
    #For some reason, attributes are parsed into a list in the HLS metadata. This reformats it into a dictionary.
    attributes_list = metadata['render_dict']['umm']['AdditionalAttributes']
    attributes = {attr['Name']: attr['Values'][0] for attr in attributes_list}
    vis_urls = granule.dataviz_links()
    meta['data_vis_url'] = vis_urls[0]
    return {**attributes, **meta}
def save_metadata_csv(metadata, file_path,folder_name):
    ## ======= write metadata csv ======= ##
    csv_file = os.path.join(file_path, (f'{folder_name}_metadata.csv'))
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metadata.keys())
        writer.writerow(metadata.values())
def get_sensor_bands(granule_name):
    file_data = granule_name.split('.')
    sensor = file_data[1]
    if sensor == 'L30':
        bands =  ['B02', 'B03', 'B04', 'B05', 'B06', 'B07','Fmask']
    else:
        bands =  ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12','Fmask']
    return bands
def get_band_links(data_links, granule_name):
    bands = get_sensor_bands(granule_name=granule_name)
    filtered_links = [f for f in data_links if any(band in f for band in bands)]
    return filtered_links

def get_sorted_files(folder_path, granule_name):

    bands = get_sensor_bands(granule_name)
    img_files = os.listdir(folder_path)
    filtered_files = [f for f in img_files if any(band in f for band in bands)]
    
    final_files = []
    for band in bands:
        for file in filtered_files:
            if band in file:
                final_files.append(file)
    
    # Sort files based on the order in sensor_bands
    def sort_key(filename):
        for band in bands:
            if band in filename:
                return bands.index(band)
        return len(bands)  # put other files at the end
    
    sorted_files = sorted(final_files, key=sort_key)
    
    return sorted_files

def create_tiling_reference(data_xml=r'C:\Users\attic\HLS Kelp Detection\maps\S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml', csv_path=r'C:\Users\attic\HLS Kelp Detection\maps'):
    tree = ET.parse(data_xml)
    root = tree.getroot()
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

    bounding_boxes = {}
    # Iterate through each Placemark
    for placemark in root.findall('.//kml:Placemark', namespace):
        tile_name = placemark.find('kml:name', namespace).text
        
        coordinates = placemark.find('.//kml:coordinates', namespace)
        if coordinates is not None:
            coords = coordinates.text.strip().split(' ')
            lon_lat = [tuple(map(float, coord.split(','))) for coord in coords]
            bounding_boxes[tile_name] = lon_lat

    csv_path = r'C:\Users\attic\HLS Kelp Detection\maps'
    csv_file = os.path.join(csv_path, 'tiling_boxes.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for tile, coords in bounding_boxes.items():
            line = [tile] + [item for coord in coords[0:4] for item in coord]
            writer.writerow(line)
    return csv_file


def search_bounding_box(tile_id, csv_file=r'C:\Users\attic\HLS Kelp Detection\maps\tiling_boxes.csv'):
        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            coordinates = None
            for row in csv_reader:
                if row[0] == tile_id:
                    coords = row[1:]
                    break
        bounding_box=(float(coords[9]), float(coords[10]), float(coords[3]),float(coords[4]))
        return bounding_box


def download_dem (tile_id, dem_path): 
    bounding_box = search_bounding_box(tile_id)
    earthaccess.login(persist=True)
    dem_results = earthaccess.search_data(
        short_name="ASTGTM",
        bounding_box=bounding_box)
    for result in dem_results:
        earthaccess.download(result, local_path=dem_path)