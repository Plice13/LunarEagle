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

class Maintenance:
    def erase_log(log_path):
        log_file = open(log_path, 'w')
        log_file.close()

    def cv2_to_PIL(image_cv2):
        return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

    def PIL_to_cv2(image_PIL):
        return cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGBA2BGR)
    
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
class Adjustment:
    def center_the_image_and_remove_big_circle_cv2(image_cv2, visualisation=False):
        x_move, y_move = Adjustment.find_circles_in_image_cv2(image_cv2, visualisation=visualisation)
        image_PIL = Maintenance.cv2_to_PIL(image_cv2)
        image_PIL = Adjustment.move_image_PIL(image_PIL, x_move, y_move)
        final_image_cv2 = Maintenance.PIL_to_cv2(image_PIL)

        return final_image_cv2

    def find_circles_in_image_cv2(image, visualisation=False):
        x_list = []
        y_list = []
        r_list = []
        height, width = image.shape[:2]

        # some adjustments
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

        # find big circle
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=730, maxRadius=800)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                x_list.append(x)
                y_list.append(y)    
                r_list.append(r)
                if visualisation == True:
                    cv2.circle(image, (x, y), r, (0, 255, 0), 1)
                    cv2.circle(image, (x, y), radius=2, color=(0, 128, 255), thickness=-1)
        if visualisation == True:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find small circle
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=300, maxRadius=400)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                x_list.append(x)
                y_list.append(y)    
                if visualisation == True:
                    cv2.circle(image, (x, y), r, (255, 0, 0), 1)
                    cv2.circle(image, (x, y), radius=2, color=(128, 0, 255), thickness=-1)
        if visualisation == True:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # get center of all circles
        x_med,y_med = (int(statistics.median(x_list)), int(statistics.median(y_list)))
        move_x = int(width/2-x_med)
        move_y = int(height/2-y_med)
        r_min = int(min(r_list))
        r_max = int(max(r_list))
        if visualisation == False:    
            pass
            #cv2.circle(image, (x_med,y_med), int(statistics.median(r_list)), (255, 255, 255), int((r_max-r_min)/2)+15)
        else:
            cv2.circle(image, (x_med,y_med), 4, (0, 0, 255), -1)
            cv2.circle(image, (x_med,y_med), int(statistics.median(r_list)), (0, 0, 255), int((r_max-r_min)/2)+15)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return move_x, move_y, 

    def move_image_PIL(image_PIL, move_x, move_y):
        new_image = Image.new('RGB', image_PIL.size)

        # paste the original image with offset
        new_image.paste(image_PIL, (move_x, move_y))

        return new_image

    def remove_tables_PIL(image_PIL, mask = Image.open("mask_table_pro.png"), visualisation=False):
        mask = mask.convert("RGBA")

        # make sure that mask and image are same size
        width, height = image_PIL.size
        mask = mask.resize((width, height))

        # combine
        result = Image.new("RGBA", image_PIL.size)
        result.paste(image_PIL.convert("RGBA"), (0,0))
        result.paste((255,255,255,255), (0, 0), mask)
        if visualisation == True:
            # having the mask purple 
            result.paste((128,0,128,255), (0, 0), mask)
            result.show()   
        return result

    def enhance_image_cv2(img, visualisation=False):
        # if not too dark then make it white
        k=220
        low = (0,0,0)
        high = (k,k,k)

        mask = cv2.inRange(img, low, high)
        mask = 255 - mask
        if visualisation == True:
            cv2.imshow("mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        blured = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)
        if visualisation == True:
            cv2.imshow("blured", blured)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        mask2 = cv2.inRange(blured, 0, 200)

        mask2 = 255 - mask2
        if visualisation == True:
            cv2.imshow("mask2", mask2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        return mask2

class Calculations:
    def find_rectangles(enhanced, base, visualisation=False):
        # some more adjustments
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # scanning contours
        for i, cnt in enumerate(contours):
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if visualisation == True:
                cv2.drawContours(base, [cnt], -1, (0, 0, 122), 1)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if visualisation == True:
                    cv2.drawContours(base, [cnt], -1, (0, 255, 122), 3)

                if 10 < w < 1600 and 10 < h < 1600:
                    # make mask of rectangle with its coordinates # hodne dlouhomi to zabralo :(((
                    mask = np.zeros_like(base, dtype=np.uint8)
                    cv2.drawContours(mask, [approx], -1, color=(255,255,255), thickness=cv2.FILLED)                    
                    mask_inverted = ~mask
                    roi_whole_image = cv2.bitwise_or(base, mask_inverted)

                    # Extract the region of interest
                    roi_small = roi_whole_image[y:y+h, x:x+w]

                    filename = sunspot_path +'/'+ picture_date +'_'+ str(x)+','+str(y)+','+str(w)+','+str(h)+ '_'+ '.png'
                    cv2.imwrite(filename, roi_small)

        if visualisation == True:
            cv2.imshow("base", base)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
        print(target_date)

        df = pd.read_csv(csv_path, delimiter=';', encoding='Windows-1250')
        df.columns = ['Datum', 'čas', 'min', 'Q', 'č', 'typ', 'skv', 'l', 'b', 'plo', 'pol', 'CV', 'SN', 'RB']

        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

        df_filtered = df[df['Datum'] == target_date]

        hour_from_csv = df_filtered['čas'].iloc[0]
        hour_from_csv = str(hour_from_csv)

        min_from_csv = df_filtered['min'].iloc[0]
        min_from_csv = str(min_from_csv)
       
        return hour_from_csv.zfill(2)+min_from_csv.zfill(2)+'00'


if __name__ == '__main__':
    visualisation = False

    folder_path = r'C:\Users\PlicEduard\ondrejov'
    sunspot_path = r'C:\Users\PlicEduard\proof\sunspots'
    log_path = 'log.txt'

    #Maintenance.erase_log(log_path)
    pictures = [pic for pic in os.listdir(folder_path) if pic.endswith(".jpg")]
    x=3

    for pic in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
        # process only every ...th picture
        if x == 3:
            # repeat code for every image in folder
            try:
                picture_day = Reading.get_day_from_image(pic) #yyyymmdd
                picture_time = Reading.get_time_from_csv(picture_day, 'Ondrejov_data_kresba.CSV') #hhmmss
                global picture_date
                picture_date = picture_day+ picture_time #yyyymmmddhhmmss
                picture_full_path=folder_path+'/'+picture_day[2:]+'dr.jpg' #picture_day[2:] for format yymmdd
                
                picture_cv2 = cv2.imread(picture_full_path)
                picture = Adjustment.center_the_image_and_remove_big_circle_cv2(picture_cv2)
                picture = Maintenance.cv2_to_PIL(picture)
                picture = Adjustment.remove_tables_PIL(picture)
                picture = Maintenance.PIL_to_cv2(picture)
                enhanced_picture = Adjustment.enhance_image_cv2(picture)
                Calculations.find_rectangles(enhanced_picture,picture,visualisation=visualisation)
                #saveing every sunspot groop              
            except Exception as e:
                log_file = open(log_path, 'a', encoding='utf-8')
                log_file.write(f'Obrázek {pic} nemohl být zpracován protože vyhodil chybu: {e}\n')
                log_file.close()
            x=0
        else:
            x+=1
