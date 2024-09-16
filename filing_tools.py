from datetime import datetime, timedelta
import os

def extract_date(filename):
    parts = filename.split('.')
    date_str = parts[3]
    date = datetime.strptime(date_str, '%Y%jT%H%M%S')
    return date

def group_by_date(filenames, max_days=4):
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

def get_paired_filenames(path,max_days=2):
    files = os.listdir(path)
    grouped_files = group_by_date(files, max_days=max_days)
    file_list = []
    for group in grouped_files:
        files = group[1]
        for file in files:
            if file not in file_list:
                file_list.append(file)
    return file_list