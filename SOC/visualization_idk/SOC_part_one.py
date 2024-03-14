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
import csv
import shutil

class Maintenance:
    def show_image(picture, name='resized_picture', screen=1, final_height=None):
        #screen 0 notebook, screen 1 monitor
        height, width = picture.shape[:2]
        if final_height == None:
            if screen == 0:
                final_height=750
            else:
                final_height=1000
            final_dimension=(round((final_height/height)*width),final_height)
        else:
            final_dimension=(round((final_height/height)*width),final_height)
        
        resized_picture = cv2.resize(picture, dsize=final_dimension)
        cv2.imshow(name, resized_picture)
        cv2.waitKey()
        Maintenance.make_dir(os.path.join(sunspot_path, 'visualization'))
        cv2.imwrite(os.path.join(sunspot_path, f'visualization/{name}.png'), resized_picture)
        cv2.destroyWindow(name)

    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def remove_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)

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
    
    def erase_csv(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

class Adjustment:
    def resize_PIL(img, save_folder_path):
        # Resize image to width 2000 while maintaining the aspect ratio
        basewidth = 2000
        width_scale = basewidth / float(img.size[0])
        hsize = int((float(img.size[1]) * float(width_scale)))
        img = img.resize((basewidth, hsize), Image.LANCZOS)

        # Create a white canvas of size 2000x1800
        background = Image.new('RGB', (2000, 1800), (255, 212, 0))

        # Calculate position to paste the resized image at the center
        paste_x = (2000 - img.size[0]) // 2
        paste_y = (1800 - img.size[1]) // 2

        # Paste the resized image onto the white canvas
        background.paste(img, (paste_x, paste_y))

        # Save the new image
        #save_path = os.path.join(save_folder_path, f"resized_{picture_date}.jpg")
        #background.save(save_path)
        Maintenance.show_image(picture=Maintenance.PIL_to_cv2(background), name='1_resized')
        return background

    def center_the_image_cv2(image_cv2, save_folder_path, visualisation=False):
        x_move, y_move = Adjustment.find_circles_in_image_cv2(image_cv2, visualisation=visualisation)
        image_PIL = Maintenance.cv2_to_PIL(image_cv2)
        image_PIL = Adjustment.move_image_PIL(image_PIL, x_move, y_move)

        #save_path = os.path.join(save_folder_path, f"moved_{picture_date}.jpg")
        #image_PIL.save(save_path)

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
        visualisation_image = image.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                x_list.append(x)
                y_list.append(y)    
                r_list.append(r)
                if visualisation == True:
                    cv2.circle(visualisation_image, (x, y), r, (189, 135, 0), 1)
                    cv2.circle(visualisation_image, (x, y), radius=2, color=(48, 0, 197), thickness=-1)
        if visualisation == True:
            Maintenance.show_image(picture=visualisation_image, name='2_big_circles')


        # find small circle
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=300, maxRadius=400)
        visualisation_image = image.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                x_list.append(x)
                y_list.append(y)    
                if visualisation == True:
                    cv2.circle(visualisation_image, (x, y), r, (189, 135, 0), 1)
                    cv2.circle(visualisation_image, (x, y), radius=2, color=(48, 0, 197), thickness=-1)  
        if visualisation == True:
            Maintenance.show_image(picture=visualisation_image, name='3_small_circles')

        # get center of all circles
        x_med,y_med = (int(statistics.median(x_list)), int(statistics.median(y_list)))
        move_x = int(width/2-x_med)
        move_y = int(height/2-y_med)
        r_min = int(min(r_list))
        r_max = int(max(r_list))
        visualisation_image = image.copy()
        if visualisation == False:    
            pass
            #cv2.circle(image, (x_med,y_med), int(statistics.median(r_list)), (255, 255, 255), int((r_max-r_min)/2)+15)
        else:
            cv2.circle(visualisation_image, (x_med,y_med), 6, (48, 0, 197), -1)
            cv2.circle(visualisation_image, (x_med,y_med), int(statistics.median(r_list)), (104, 164, 0), int((r_max-r_min)/2))
            Maintenance.show_image(picture=visualisation_image, name='4_final_circle')


        return move_x, move_y, 

    def move_image_PIL(image_PIL, move_x, move_y):
        new_image = Image.new('RGB', image_PIL.size)

        # paste the original image with offset
        new_image.paste(image_PIL, (move_x, move_y))

        return new_image

    def remove_tables_PIL(image_PIL, mask = Image.open("maska_2000_1800.png"), visualisation=False):
        mask = mask.convert("RGBA")

        # make sure that mask and image are same size
        width, height = image_PIL.size
        mask = mask.resize((width, height))

        # combine
        result = Image.new("RGBA", image_PIL.size)
        result.paste(image_PIL.convert("RGBA"), (0,0))
        result.paste((255,255,255,255), (0, 0), mask)

        if visualisation == True:
            Maintenance.show_image(picture=Maintenance.PIL_to_cv2(result), name='5_applied_mask')   
        return result

    def enhance_image_cv2(img, visualisation=False):
        # if not too dark then make it white
        k=220
        low = (0,0,0)
        high = (k,k,k)

        mask = cv2.inRange(img, low, high)
        mask = 255 - mask
        if visualisation == True:
            Maintenance.show_image(picture=mask, name='6_grayscale1')

        blured = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)
        if visualisation == True:
            Maintenance.show_image(picture=blured, name='7_blured')

        mask2 = cv2.inRange(blured, 0, 200)

        mask2 = 255 - mask2
        if visualisation == True:
            Maintenance.show_image(picture=mask2, name='8_grayscale2')



        return mask2

class Calculations:
    def find_rectangles(enhanced, base, visualisation=False):
        # some more adjustments
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        visualisation_image = base.copy()
        # scanning contours
        for i, cnt in enumerate(contours):
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if visualisation == True:
                cv2.drawContours(visualisation_image, [cnt], -1, (0, 0, 122), 1)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 10 < w < 1000 and 10 < h < 1000:
                    # Extract the region of interest
                    contour_length = cv2.arcLength(cnt, True)
                    contour_area = cv2.contourArea(cnt)
                    pravdivovost, uhel = Calculations.is_parallel(approx)
                    if pravdivovost == True:
                        pravdivost, value = Calculations.is_thin(approx)
                        if pravdivost == False:
                            filename = sunspot_path +'/'+ picture_date +'_'+ str(x)+','+str(y)+','+str(w)+','+str(h)+ '__'+f'cLenght={contour_length}_cArea={contour_area}__tLoustka={value}_uhel={uhel}'+ '.png'
                            if visualisation == True:
                                cv2.drawContours(visualisation_image, [cnt], -1, (0, 122, 0), 2)
                        else:
                            save_path = sunspot_path +'\deleted\TH'
                            Maintenance.make_dir(save_path)
                            filename = save_path+'/'+ picture_date +'_'+ str(x)+','+str(y)+','+str(w)+','+str(h)+ '__'+f'cLenght={contour_length}_cArea={contour_area}'+ '.png'
                    else:
                        save_path = sunspot_path +'\deleted\SHAPE'
                        Maintenance.make_dir(save_path)
                        filename = save_path+'/'+ picture_date +'_'+ str(x)+','+str(y)+','+str(w)+','+str(h)+ '__'+f'cLenght={contour_length}_cArea={contour_area}'+ '.png'
                else:
                    save_path = sunspot_path +'\deleted\WH'
                    Maintenance.make_dir(save_path)
                    filename = save_path+'/'+ picture_date+'.png'

                #techtle mechtle s maskou
                x_middle_of_roi = int(x+w/2)
                y_middle_of_roi = int(y+h/2)

                with open(csv_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([picture_date + '_' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h),
                                        str([[int(approx[0][0][0]) - x_middle_of_roi +150, int(approx[0][0][1]) - y_middle_of_roi +150],
                                        [int(approx[1][0][0]) - x_middle_of_roi +150, int(approx[1][0][1]) - y_middle_of_roi +150],
                                        [int(approx[2][0][0]) - x_middle_of_roi +150, int(approx[2][0][1]) - y_middle_of_roi +150],
                                        [int(approx[3][0][0]) - x_middle_of_roi +150, int(approx[3][0][1]) - y_middle_of_roi +150]])])

                '''mask = np.zeros_like(base, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, color=(255,255,255), thickness=cv2.FILLED)                    
                mask_inverted = ~mask
                roi_whole_image = cv2.bitwise_or(base, mask_inverted)
                '''
                
                # Extract the region of interest
                roi_small = base[y_middle_of_roi-150:y_middle_of_roi+150, x_middle_of_roi-150:x_middle_of_roi+150]

                cv2.imwrite(filename, roi_small)

        if visualisation == True:
            Maintenance.show_image(picture=visualisation_image, name='9_detected_features')


    def is_thin(approx, boundary=28):
        try:
            appr = sorted(approx, key=lambda c: c[0][0])
            pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
            pc, pd = sorted(appr[2:], key=lambda c: c[0][1])

            provided_points = np.array([pa, pb, pc, pd])

            # Calculate pairwise distances between all provided points
            distances = np.sqrt(np.sum((provided_points[:, None] - provided_points) ** 2, axis=-1))
            min_distance_list = []
            for i in range(len(provided_points)):
                for j in range(i + 1, len(provided_points)):
                    distance = distances[i, j]
                    min_distance_list.append(distance)
                    if distance < boundary:
                        print('HIT')
                        return True, min(min_distance_list)
            return False, min(min_distance_list)
        except:
            print('------ERROR------')

    def is_parallel(approx, boundary=8):
        # Ensure there are exactly 4 points
        if len(approx) != 4:
            return False

        # Sort the points
        sorted_approx = sorted(approx, key=lambda c: (c[0][0], c[0][1]))
        pa, pb = sorted(sorted_approx[:2], key=lambda c: c[0][1])
        pc, pd = sorted(sorted_approx[2:], key=lambda c: c[0][1])

        # Calculate angles between lines formed by the points
        angle1 = math.atan2(pa[0][1] - pb[0][1], pa[0][0] - pb[0][0]) - math.atan2(pc[0][1] - pd[0][1], pc[0][0] - pd[0][0])
        angle2 = math.atan2(pa[0][1] - pc[0][1], pa[0][0] - pc[0][0]) - math.atan2(pb[0][1] - pd[0][1], pb[0][0] - pd[0][0])

        # Check if the absolute degrees of the angles are within the boundary
        if abs(math.degrees(angle1)) < boundary and abs(math.degrees(angle2)) < boundary:
            return True, (math.degrees(angle1),math.degrees(angle2))
        else:
            return False, (math.degrees(angle1),math.degrees(angle2))

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
    visualisation = True

    folder_path = r'C:\Users\PlicEduard\SOC\img'
    sunspot_path = r'C:\Users\PlicEduard\SOC\run1'
    csv_path = os.path.join(sunspot_path, 'csv.csv')
    log_path = os.path.join(sunspot_path, 'log.txt')

    Maintenance.remove_dir(os.path.join(sunspot_path))
    Maintenance.make_dir(sunspot_path)
    Maintenance.erase_log(log_path)
    Maintenance.erase_csv(csv_path)
    pictures = [pic for pic in os.listdir(folder_path) if pic.endswith(".jpg")]
    x=0

    for pic in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
        # process only every ...th picture
        #if x==7000:
        if True:
            # repeat code for every image in folder
            try:
                picture_day = Reading.get_day_from_image(pic) #yyyymmdd
                
                picture_time = Reading.get_time_from_csv(picture_day, 'Ondrejov_data_kresba.CSV') #hhmmss
                global picture_date
                picture_date = picture_day+ picture_time #yyyymmmddhhmmss
                picture_full_path=folder_path+'/'+picture_day[2:]+'dr.jpg' #picture_day[2:] for format yymmdd
                
                picture = Image.open(picture_full_path)
                picture = Adjustment.resize_PIL(picture, sunspot_path)
                picture = Maintenance.PIL_to_cv2(picture)
                picture = Adjustment.center_the_image_cv2(picture, sunspot_path,visualisation=visualisation)
                picture = Maintenance.cv2_to_PIL(picture)
                picture = Adjustment.remove_tables_PIL(picture,visualisation=visualisation)
                picture = Maintenance.PIL_to_cv2(picture)
                #cv2.imwrite(os.path.join(mask_dir, pic), picture)
                enhanced_picture = Adjustment.enhance_image_cv2(picture,visualisation=visualisation)
                Calculations.find_rectangles(enhanced_picture,picture,visualisation=visualisation)
                #saveing every sunspot groop              
            except Exception as e:
                log_file = open(log_path, 'a', encoding='utf-8')
                log_file.write(f'Obrázek {pic} nemohl být zpracován protože vyhodil chybu: {e}\n')
                log_file.close()
            x=0
        else:
            x+=1
