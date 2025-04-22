import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_model():

    model = joblib.load('trained_model.pkl')
    return model

def predict_finish_time(df):

    model = load_model()

    features = [
    'speed_kmh',
    'ele_diff',
    'grade_percent',
    'delta_km',
    'delta_time_s',
    'distance_km'
]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)

    df['predicted_time_to_finish'] = predictions
    return df