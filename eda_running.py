import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    # Konfiguracija za prikaz grafika
    sns.set(style='whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)

    # Učitavanje podataka
    df = pd.read_csv('output/combined_data.csv', parse_dates=['time'])

    # ----------------- Histogram brzine -----------------
    plt.figure()
    sns.histplot(df['speed_kmh'], bins=40, kde=True, color='skyblue')
    plt.title('Distribucija brzine (km/h)')
    plt.xlabel('Brzina [km/h]')
    plt.ylabel('Učestalost')
    plt.savefig('diagrams/speed_distribution.png')
    plt.close()

    # ----------------- Elevation change (uspon/silazak) -----------------
    plt.figure()
    sns.histplot(df['ele_diff'], bins=40, kde=True, color='orange')
    plt.title('Promena nadmorske visine (ele_diff)')
    plt.xlabel('Promena visine (m)')
    plt.ylabel('Učestalost')
    plt.savefig('diagrams/elevation_diff_distribution.png')
    plt.close()

    # ----------------- Ukupna distanca po sesiji -----------------
    session_distance = df.groupby('file')['delta_km'].sum().sort_values()

    plt.figure()
    session_distance.plot(kind='bar', color='green')
    plt.title('Distanca po trening sesiji')
    plt.ylabel('Distanca (km)')
    plt.xlabel('Fajl (trening)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('diagrams/distance_per_session.png')
    plt.close()

    # ----------------- Brzina tokom vremena -----------------
    plt.figure()
    sns.lineplot(data=df, x='time', y='speed_kmh', hue='file', legend=False)
    plt.title('Brzina tokom vremena (po sesijama)')
    plt.xlabel('Vreme')
    plt.ylabel('Brzina (km/h)')
    plt.savefig('diagrams/speed_over_time.png')
    plt.close()

    # ----------------- Elevacija kroz distancu -----------------
    plt.figure()
    sns.lineplot(data=df, x='distance_km', y='ele', hue='file', legend=False)
    plt.title('Profil staze (nadmorska visina u odnosu na distancu)')
    plt.xlabel('Distanca (km)')
    plt.ylabel('Nadmorska visina (m)')
    plt.savefig('diagrams/elevation_profile.png')
    plt.close()