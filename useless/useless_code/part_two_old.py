import os
from tqdm import tqdm
import cv2
import numpy as np
import statistics
from PIL import Image, ImageDraw
import pandas as pd
import math
import sunpy.coordinates
from datetime import datetime
import shutil


class Maintenance:
    def erase_log(log_path):
        log_file = open(log_path, 'w')
        log_file.close()

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

class Calculations:

    def calculate_Q(point2, middle_point, sun_date):
        print(f'Points are, middle: {middle_point}, and point2 is: {point2}')
        # Spočtěte úhel vůči svislé ose
        # print(f'Čas obrázku je: {sun_date}')
        if int(sun_date) < 20170816000000:
            # north down
            print('Sever dole')
            angle = np.degrees(np.arctan2(-(point2[0] - middle_point[0]), (point2[1] - middle_point[1])))
            # print(f'Správný úhel je: {angle}, druhý úhel je {angle2}')
        else:
            # north at top
            print('Sever nahoře')
            angle = np.degrees(np.arctan2(point2[0] - middle_point[0], -(point2[1] - middle_point[1])))
            # print(f'Správný úhel je: {angle}, druhý úhel je {angle2}')

        if angle < 0:
            angle = angle+360
        elif angle >360:
            angle = angle-360
        else:
            angle=angle

        return angle

    def calculate_middle_of_sunspot(part_of_name):
        values = part_of_name.split(',')
        x, y, w, h = map(int, values)
        middle = (x+w/2, y+h/2)
        return middle

    def calculate_rho(point2, middle_point):
        # print(f'Střed ve funkci je: {middle_point}')
        radius_of_sun = int(middle_point[0])*0.735
        # print(f'Radius Slunce je: {radius_of_sun}')
        distance_to_middle = np.sqrt((point2[0] - middle_point[0]) ** 2 + (point2[1] - middle_point[1]) ** 2)
        print(f'Distance od centra je: {distance_to_middle} a poměr je tedy: {distance_to_middle/radius_of_sun}')
        rho=np.arcsin(distance_to_middle/radius_of_sun)
        # print(f'Z toho rho je: {rho}')
        return rho

class Reading:
    def get_day_from_image(image_name):
        image_string = str(image_name).replace('dr.jpg','')
        if int(image_string[:2]) < 40:
            image_string = '20'+image_string
        else:
            image_string = '19' + image_string
        return image_string
    
    def get_time_from_csv(image_string, csv_path):
        target_date = f'{image_string[:4]}-{image_string[4:6]}-{image_string[6:]}'

        df = pd.read_csv(csv_path, delimiter=';', encoding='Windows-1250')
        df.columns = ['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB']

        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

        df_filtered = df[df['Datum'] == target_date]

        hour_from_csv = df_filtered['čas'].iloc[0]
        hour_from_csv = str(hour_from_csv)

        min_from_csv = df_filtered['min'].iloc[0]
        min_from_csv = str(min_from_csv)
       
        return hour_from_csv.zfill(2)+min_from_csv.zfill(2)+'00'

    def get_closest_match(target_b, target_l, full_date, csv_path):
        print(target_b, target_l, '\n')
        input_date  = datetime.strptime(full_date, '%Y%m%d%H%M%S')  # Parse the input string as a datetime object
        target_date = input_date.strftime('%Y-%m-%d')  # Format the datetime object to 'YYYY-MM-DD' format

        df = pd.read_csv(csv_path, delimiter=';', encoding='Windows-1250')
        df.columns = ['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB']

        df['l'] = pd.to_numeric(df['l'], errors='coerce')
        df['b'] = pd.to_numeric(df['b'], errors='coerce')
        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

        df_filtered = df[df['Datum'] == target_date].copy()

        df_filtered['distance'] = np.sqrt((df_filtered['l'] - target_l) ** 2 + (df_filtered['b'] - target_b) ** 2)
        print(df_filtered,'\n')
        min_distance = df_filtered['distance'].min()
        nearest_row = df_filtered[df_filtered['distance'] == min_distance]

        sunspot_classification = nearest_row['typ'].values[0]
        min_distance = round(min_distance)
        print(sunspot_classification, min_distance)
        return sunspot_classification, min_distance


if __name__ == '__main__':
    sunspot_path = r'C:\Users\PlicEduard\sunspots_full'
    save_path = r'C:\Users\PlicEduard\clasifics\classification_mega'
    log_path = 'log10.txt'

    #Maintenance.erase_log(log_path)

    sunspots = [sunspot for sunspot in os.listdir(sunspot_path) if sunspot.endswith(".png")]

    for sunspot in tqdm(os.listdir(sunspot_path), total=len(os.listdir(sunspot_path))):
        if True:
            try:
                sunspot_date = sunspot.split('_')[0]
                sunspot_coordinates = sunspot.split('_')[1]
                
                # get sun information in that time
                B0 = math.radians(Maintenance.str_to_dec_degree(sunpy.coordinates.sun.B0(time=sunspot_date)))
                P = math.radians(Maintenance.str_to_dec_degree(sunpy.coordinates.sun.P(time=sunspot_date)))
                L0 = math.radians(Maintenance.str_to_dec_degree(sunpy.coordinates.sun.L0(time=sunspot_date)))

                # get Q and rho
                midpoint_of_sunspot = Calculations.calculate_middle_of_sunspot(sunspot_coordinates)
                midpoint_of_image = (1000,900)
                Q = math.radians(Calculations.calculate_Q(midpoint_of_sunspot, midpoint_of_image, sunspot_date))
                rho = Calculations.calculate_rho(midpoint_of_sunspot, midpoint_of_image)

                #get b and l
                b = math.asin(math.sin(B0) * math.cos(rho) + math.cos(B0) * math.sin(rho) * math.cos(P- Q))
                l = (math.asin((math.sin(rho) * math.sin(P-Q)) / math.cos(b)) + L0)
                print(f'\nB je součet z {math.sin(B0) * math.cos(rho)} a {math.cos(B0) * math.sin(rho) * math.cos(P- Q)}')
                print(f'L je zlomek, kde nahoře je {(math.sin(rho) * math.sin(P-Q))} a dole jen {math.cos(b)}, což znamená že zlomek vyjde {(math.sin(rho) * math.sin(P-Q)) / math.cos(b)}, po aplikace asinu {math.asin((math.sin(rho) * math.sin(P-Q)) / math.cos(b))} ve stupních pak {math.degrees(math.asin((math.sin(rho) * math.sin(P-Q)) / math.cos(b)))}')
                b = math.degrees(b)
                l = math.degrees(l)
                print(f'\nÚdaje o Slunci v den {sunspot_date[:8]} jsou L: {round(math.degrees(L0),1)}, B: {round(math.degrees(B0),1)}, P: {round(math.degrees(P),1)}')
                print(f'Q je: {math.degrees(Q)} a rho je: {rho}')
                #get match
                sunspot_clasification, min_distance = Reading.get_closest_match(b, l, sunspot_date, 'Ondrejov_data_kresba.CSV')
                source_path = sunspot_path+'/'+sunspot  # Replace with the path to your source image
                if not os.path.exists(save_path+'/'+sunspot_clasification):
                    # If it doesn't exist, create the directory
                    os.makedirs(save_path +'/'+sunspot_clasification)
                destination_path =  save_path +'/'+sunspot_clasification+'/'+sunspot+f'_{round(math.degrees(P))}__{round(math.degrees(Q))}_{round(rho,2)}__{round(b)}_{round(l)}__min_dist={min_distance}_.png'  # Replace with the path where you want to copy the image

                shutil.copyfile(source_path, destination_path)
            except Exception as e:
                log_file = open(log_path, 'a', encoding='utf-8')
                log_file.write(f'Skvrna {sunspot} nemohla být zpracován protože vyhodila chybu: {e}\n')
                log_file.close()
