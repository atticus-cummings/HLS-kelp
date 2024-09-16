from noaa_coops import Station
from pprint import pprint
import os
import rasterio
from datetime import datetime, timedelta
from dateutil.parser import parse
import pandas as pd
import csv
import pytz
import numpy as np
from rasterio.errors import RasterioIOError

def get_date(path, file):
    with rasterio.open(os.path.join(path, file), 'r') as src:
        metadata = src.tags()
    date_string = metadata['TIMESTAMP']
    try:
        date = parse(date_string)
    except:
        print(f"File: {file} | Invalid date: {date_string} ")
        return None
    return date

def get_tide_data(path,file, station=None):
    if station is None:
        station =Station(id=9411340)
    time = get_date(path, file)
    if time is None:
        return None
    begin_date = (time - timedelta(days=1)).strftime('%Y%m%d')
    end_date = (time + timedelta(days=1)).strftime('%Y%m%d')
    try:
        df_water_levels = station.get_data(
            begin_date=begin_date,  
            end_date=end_date,
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt"
        )
    except:
        return None
    # Reset index and convert time column to datetime
    df = df_water_levels.reset_index()
    df['datetime'] = pd.to_datetime(df['t'])
    df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    df['time_diff'] = abs(df['datetime'] - time)
    nearest_index = df['time_diff'].idxmin()
    return df.loc[nearest_index,'v']

def load_lter(lter_path):
    lter_df = pd.read_csv(lter_path)
    lter_df['datetime'] = lter_df['matlab_datenum'].apply(matlab_datenum_to_datetime)
    return lter_df


def matlab_datenum_to_datetime(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    return python_datetime

def index_from_date(df, date):

    # Ensure the 'date' parameter is timezone-aware
    if date.tzinfo is None:
        date = date.tz_localize('UTC')

    lower_bound = 0
    upper_bound = len(df.index) - 1
    index = 0
    date_found = False

    while not date_found and lower_bound <= upper_bound:
        index = (upper_bound + lower_bound) // 2
        time = df.iloc[index]['datetime']
        
        # Ensure the 'time' from the dataframe is timezone-aware
        if time.tzinfo is None:
            time = time.tz_localize('UTC')
        
        if time == date:
            date_found = True
            break
        elif time < date:
            lower_bound = index + 1
        else:
            upper_bound = index - 1

    difference = (time - date).total_seconds()
    return index, time, difference

def get_current_data(path,file, df=None, lter_path=None):
    if(lter_path is None and df is None):
        return None
    if df is None and lter_path is not None:
        df = pd.read(lter_path)
    date = get_date(path,file)
    if date is None:
        return False
    index, time,difference = index_from_date(df,date)
    if(difference > 20*60):
        print(f"data for {date} not found. Time difference: {difference}s")
    current = np.sqrt((df.iloc[index]['E_Vel_02.5m_bin'] + df.iloc[index]['E_Vel_03.5m_bin'])**2 + (df.iloc[index]['E_Vel_02.5m_bin'] + df.iloc[index]['E_Vel_03.5m_bin'])**2)
    return current

def insert_tide_current_metadata(path, file, df, station):
    current = get_current_data(path,file,df=df)
    if(current is None):
        print("Error getting current data")
        return False
    tide = get_tide_data(path,file, station=station)
    if(tide is None):
        print("Error getting tide data")
        return False
    
    current_str = str(current)
    tide_str = str(tide)
    try:
        with rasterio.open(os.path.join(path,file), 'r+') as src:
            src.update_tags(
            CURRENT=current_str,
            TIDE=tide_str
            )
    except RasterioIOError as e:
        print(f"Error updating file metadata {file}: {e}")
        return False
    return True
    
def insert_tide_current_dataset(path, lter_path, station=Station(id=9411340)):
    lter_df = load_lter(lter_path)
    files = os.listdir(path)
    for i, file in  enumerate(files):
        if not insert_tide_current_metadata(path,file, df=lter_df, station=station):
            print(f'{file} failed. Continuing.')