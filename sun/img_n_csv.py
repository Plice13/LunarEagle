import numpy as np
import math
import sunpy.coordinates
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_l_b():
    def str_to_dec_degree(string):
        string = str(string)
        parts = string.split('d')
        degrees = float(parts[0])

        minutes_str, seconds_str = parts[1].split('m')
        minutes = float(minutes_str)

        seconds_str = seconds_str.rstrip('s')
        seconds = float(seconds_str)

        # Převeďte na základní desetinný tvar ve stupních
        decimal_degrees = degrees + minutes / 60 + seconds / 3600
        return decimal_degrees

    def calculate_rho():
        radius_of_sun = 750 #px
        middle_point = (1000, 900)  
        point2 = (830, 666)
        distance_to_middle = np.sqrt((point2[0] - middle_point[0]) ** 2 + (point2[1] - middle_point[1]) ** 2)
        rho=np.arcsin(distance_to_middle/radius_of_sun)
        print(distance_to_middle, radius_of_sun)
        print(f'Rho je přesněji: {rho} psích jazíčků')
        return rho

    def calculate_Q():
        middle_point = (1000, 900)    
        point2 = (830, 666)

        # Spočtěte úhel vůči svislé ose
        angle = np.degrees(np.arctan2(point2[0] - middle_point[0], -(point2[1] - middle_point[1])))

        if angle < 0:
            angle = angle+360
        elif angle >360:
            angle = angle-360
        else:
            angle=angle

        print(f'Úhel vůči svislé ose: {angle} stupňů')
        return angle

    calculate_Q()
    calculate_rho()

    B0 = math.radians(str_to_dec_degree(sunpy.coordinates.sun.B0(time='20230926084800')))
    P = math.radians(str_to_dec_degree(sunpy.coordinates.sun.P(time='20230926084800')))
    L0 = math.radians(str_to_dec_degree(sunpy.coordinates.sun.L0(time='20230926084800')))
    rho=math.asin(calculate_rho())
    Q=math.radians(calculate_Q())
    b = math.asin(math.sin(B0) * math.cos(rho) + math.cos(B0) * math.sin(rho) * math.cos(P - Q))
    print(f"b = {math.degrees(b)}")

    l = (math.asin((math.sin(rho) * math.sin(P - Q)) / math.cos(b)) + L0)
    print(f"l = {math.degrees(l)}")
    return math.degrees(l), math.degrees(b)

def scan_csv(target_l, target_b, target_date = '2023-09-26'):
    df = pd.read_csv('Ondrejov_data_kresba.csv', delimiter=';', encoding='Windows-1250')
    df.columns = ['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB']

    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['b'] = pd.to_numeric(df['b'], errors='coerce')
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

    # Filtrujte řádky pro konkrétní datum
    df_filtered = df[df['Datum'] == target_date]

    print('\nfiltrovááno',df_filtered)
    print('\n\n\n', (df_filtered['l']))
    # Vytvořte sloupec pro vzdálenost
    df_filtered['distance'] = np.sqrt((df_filtered['l'] - target_l) ** 2 + (df_filtered['b'] - target_b) ** 2)

    # Najděte řádek s nejmenší vzdáleností
    nearest_row = df_filtered[df_filtered['distance'] == df_filtered['distance'].min()]

    # Výpis nejbližšího řádku s hodnotou vzdálenosti
    print("Nejbližší řádek:")
    print(nearest_row[['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB', 'distance']])

l, b = get_l_b()
print(l,b)
scan_csv(l,b)
