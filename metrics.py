from geopy.distance import geodesic

def compute_metrics(df):
    df['delta_km'] = df.apply(
        lambda row: geodesic(
            (df.loc[row.name - 1, 'lat'], df.loc[row.name - 1, 'lon']),
            (row['lat'], row['lon'])
        ).km if row.name > 0 and df.loc[row.name, 'file'] == df.loc[row.name - 1, 'file'] else 0,
        axis=1
    )

    df['delta_time_s'] = df['time'].diff().dt.total_seconds().fillna(0)
    df.loc[df['file'] != df['file'].shift(1), 'delta_time_s'] = 0

    df['speed_kmh'] = df['delta_km'] / (df['delta_time_s'] / 3600 + 1e-9)
    df['ele_diff'] = df['ele'].diff().fillna(0)
    df.loc[df['file'] != df['file'].shift(1), 'ele_diff'] = 0

    df['grade_percent'] = (df['ele_diff'] / (df['delta_km'] * 1000 + 1e-9)) * 100
    df['distance_km'] = df.groupby('file')['delta_km'].cumsum()

    # Čišćenje
    df = df[(df['speed_kmh'] < 30) & (df['grade_percent'].abs() < 50)]
    return df
