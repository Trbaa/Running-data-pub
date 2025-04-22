import os
import gpxpy
import pandas as pd

def parse_gpx_file(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'file': os.path.basename(file_path),
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation,
                    'time': point.time
                })
    return pd.DataFrame(data)

def load_all_gpx(folder_path):
    all_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.gpx'):
                full_path = os.path.join(root, file)
                df = parse_gpx_file(full_path)
                all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values('time')
    return combined_df
