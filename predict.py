import joblib

def load_model():

    lr_model = joblib.load('model/trained_model_lr.pkl')
    rf_model = joblib.load('model/trained_model_rf.pkl')
    return lr_model,rf_model

def predict_finish_time(df):

    lr_model,rf_model = load_model()

    features = [
    'speed_kmh',
    'ele_diff',
    'grade_percent',
    'delta_km',
    'delta_time_s',
    'distance_km'
]
    X = df[features]

    scaler = joblib.load('scaler.pkl')
    X_scaled = scaler.transform(X)

    df['lr_predicted_time'] = lr_model.predict(X_scaled)
    df['rf_predicted_time'] = rf_model.predict(X_scaled)
    return df