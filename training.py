# training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import train_model, evaluate_model
import joblib

def prepare_data(df):
    
    df['time_to_finish'] = df['distance_km'] / df['speed_kmh'].replace(0, 1)

    # Odabir feature-a (svi relevantni podaci koji će uticati na model)
    features = [
    'speed_kmh',
    'ele_diff',
    'grade_percent',
    'delta_km',
    'delta_time_s',
    'distance_km'
]
    X = df[features]
    
    # Ciljna promenljiva (npr. vreme do cilja ili ETA)
    y = df['time_to_finish']  # Ova kolona je sada dodata

    # Standardizacija podataka
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Podela na trening i test skup (80% za trening, 20% za testiranje)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate(df):
    # Priprema podataka
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Treniranje modela
    model = train_model(X_train, y_train)

    # Evaluacija modela
    evaluate_model(model, X_test, y_test)
