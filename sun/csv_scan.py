import pandas as pd
import numpy as np
from tqdm import tqdm
# Nahrajte CSV soubor s určením správného kódování a oddělovače
df = pd.read_csv('Ondrejov_data_kresba.csv', delimiter=';', encoding='Windows-1250')

# Nastavte názvy sloupců ručně
df.columns = ['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB']


'''# Příprava logovacího souboru pro zápis
log_file = open('sun\log.txt', 'w')

# Iterace přes DataFrame a zápis řádků do logu
for index, row in tqdm(df.iterrows(), total=len(df)):
    log_file.write(f"{row.to_string(index=False)}\n")

# Uzavření logovacího souboru
log_file.close()
'''
# Převod sloupců 'l' a 'b' na číselné hodnoty
df['l'] = pd.to_numeric(df['l'], errors='coerce')
df['b'] = pd.to_numeric(df['b'], errors='coerce')
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

'''log_file = open('sun\log_KONV.txt', 'w')

# Iterace přes DataFrame a zápis řádků do logu
for index, row in tqdm(df.iterrows(), total=len(df)):
    log_file.write(f"{row.to_string(index=False)}\n")

# Uzavření logovacího souboru
log_file.close()
'''

# Zadejte cílové hodnoty l a b
target_l = 86
target_b = 17
target_date = '2023-09-26'  # Zadejte konkrétní datum

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

