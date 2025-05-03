from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import train_model_lr, evaluate_model, train_model_rf
import joblib

def prepare_data(df):
    
    total_distance_km = df['distance_km'].max()

    df['remaining_distance_km'] = total_distance_km - df['distance_km']
    df['speed_kmh'] = df['speed_kmh'].replace(0, 1e-3)  #deljenje 0
    df['time_to_finish'] = df['remaining_distance_km'] / df['speed_kmh']

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
    y = df['time_to_finish'] 

    # Standardizacija podataka
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, 'scaler.pkl')

    # Podela na trening i test skup (80% za trening, 20% za testiranje)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate(df):
    # Priprema podataka
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Treniranje modela
    model_lr = train_model_lr(X_train, y_train)
    model_rf = train_model_rf(X_train,y_train)

    # Evaluacija modela
    evaluate_model(model_lr, X_test, y_test,'Linear Regression')
    evaluate_model(model_rf,X_test,y_test,'Random Forest')
