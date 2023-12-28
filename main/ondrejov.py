#start of extract sunspots

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
    def make_dir(path):
        if not os.path.exists(path):
            print(path)
            # If it doesn't exist, create the directory
            os.makedirs(path)

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
    def resize_PIL(img):
        # Resize image to width 2000 while maintaining the aspect ratio
        basewidth = 2000
        width_scale = basewidth / float(img.size[0])
        hsize = int((float(img.size[1]) * float(width_scale)))
        img = img.resize((basewidth, hsize), Image.LANCZOS)

        # Create a white canvas of size 2000x1800
        background = Image.new('RGB', (2000, 1800), (0, 0, 255))

        # Calculate position to paste the resized image at the center
        paste_x = (2000 - img.size[0]) // 2
        paste_y = (1800 - img.size[1]) // 2

        # Paste the resized image onto the white canvas
        background.paste(img, (paste_x, paste_y))

        # Save the new image
        #save_path = os.path.join(save_folder_path, f"resized_{picture_date}.jpg")
        #background.save(save_path)
        return background

    def center_the_image_cv2(image_cv2, visualisation=False):
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
        base_for_show = base.copy()
        # some more adjustments
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # scanning contours
        for i, cnt in enumerate(contours):
            cv2.drawContours(base_for_show, [cnt], -1, (0, 122, 122), 1)

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
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
                            #draw contours to display
                            cv2.drawContours(base_for_show, [cnt], -1, (0, 0, 122), 3)


                            filename = os.path.join(image_directory, f'_skvrna_{i+100}.png')
                            print(filename)
                            #techtle mechtle s maskou
                            x_middle_of_roi = int(x+w/2)
                            y_middle_of_roi = int(y+h/2)

                            mask = np.zeros_like(base, dtype=np.uint8)
                            cv2.drawContours(mask, [approx], -1, color=(255,255,255), thickness=cv2.FILLED)                    
                            mask_inverted = ~mask
                            roi_whole_image = cv2.bitwise_or(base, mask_inverted)
                            
                            # Extract the region of interest
                            roi_small = roi_whole_image[y_middle_of_roi-150:y_middle_of_roi+150, x_middle_of_roi-150:x_middle_of_roi+150]

                            cv2.imwrite(filename, roi_small)
        
        base_for_show = cv2.resize(base_for_show, (1000,900), interpolation = cv2.INTER_AREA)
        cv2.imshow('Detekované kontury',base_for_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
        

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
    


if __name__=='__main__':
    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_directory = r'C:\Users\PlicEduard\program'
    try:
        shutil.rmtree(image_directory)
    except:
        pass
    os.makedirs(image_directory)
    # List all files in the script's directory
    files = os.listdir(script_directory)

    # Find the first file with a common image extension (e.g., jpg, png)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more if needed

    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_file_path = os.path.join(script_directory, file)
            break
        else:
            print('no image')
    # Open the image using PIL
    picture = Image.open(image_file_path)

    picture = Adjustment.resize_PIL(picture)
    picture = Maintenance.PIL_to_cv2(picture)
    picture = Adjustment.center_the_image_cv2(picture)
    picture = Maintenance.cv2_to_PIL(picture)
    picture = Adjustment.remove_tables_PIL(picture)
    picture = Maintenance.PIL_to_cv2(picture)

    enhanced_picture = Adjustment.enhance_image_cv2(picture)
    Calculations.find_rectangles(enhanced_picture,picture)

    #end of extract sunspots

    #postprocessing image

    ##remove orange
    LOWER = np.array([0, 0, 200])
    UPPER = np.array([255, 255, 255])  # Adjust upper range to cover more shades of orange
    for file in os.listdir(image_directory):
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
            img_path = os.path.join(image_directory, file)
            im = cv2.imread(img_path)
            im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(im_hsv, LOWER, UPPER)
            # Replace orange regions with white
            im_filtered = im.copy()
            im_filtered[mask > 0] = [255, 255, 255]
            im_filtered = 255 - im_filtered
            cv2.imwrite(img_path, im_filtered)














    #end of postprocessing image


    #starting model
    import os

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    import re


    # Load the model
    samples_dir = image_directory
    model_dir = script_directory
    #classes = ['c','i','o','x']
    #classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
    #classes = ['a', 'h', 'k', 'r', 's', 'x']
    #classes = ['A', 'B', 'C', 'DEF', 'H']
    #classes = ['Axx','Bxi','Bxo','Cai','Cao','Chi','Cho','Cki','Cko','Cri','Cro','Csi','Cso','Dac','Dai','Dao','Dhi','Dkc','Dki','Dko','Dri','Dro','Dsc','Dsi','Dso','Eac','Eai','Ekc','Eki','Eko','Esc','Esi','Fac','Fkc','Fki','Hax','Hhx','Hkx','Hrx','Hsx']

    model_files = [model for model in os.listdir(model_dir) if model.endswith('.h5')]
    #model_file = model_files[0]
    #print(f'Nalezeno celkem {len(model_files)} modelů ve složce, bude používán model: {model_file}')
    for model_file in model_files:
        print(f'\n\n\nbude používán model: {model_file}')
        model = load_model(os.path.join(model_dir, model_file))
        # get classes
        match = re.search(r"\['(.*?)'\]", model_file)
        letter_list_str = match.group(1)
        letter_list = [letter.strip() for letter in letter_list_str.split(',')]
        classes = [s.strip("'") for s in letter_list]

        print(classes)

        # Folder containing the images
        image_folder = samples_dir

        # Get a list of all files in the folder
        image_paths = [os.path.join(root, file) for root, dirs, files in os.walk(image_folder) for file in files if file.endswith(('png', 'jpg', 'jpeg'))]

        # Load and preprocess the images, converting to grayscale
        images = [Image.open(path).convert('L').resize((300, 300)) for path in image_paths]
        image_arrays = [np.array(img) / 255.0 for img in images]
        image_arrays = np.array(image_arrays)

        # Add a channel dimension if the model expects input shape (None, 300, 300, 1)
        if model.input_shape[-1] == 1:
            image_arrays = np.expand_dims(image_arrays, axis=-1)

        # Make batch predictions
        predictions_batch = model.predict(image_arrays)
        print(predictions_batch)


        for i, (path, predictions) in enumerate(zip(image_paths, predictions_batch)):
            class_index = np.argmax(predictions)
            predicted_class = classes[class_index]
            confidence = predictions[class_index]

            # process path
            base, filename = os.path.split(path)
            print(f'Přejmenování souboru {filename} na predikovanou třídu {predicted_class} s přesností {confidence}.')
            os.rename(os.path.join(base, filename), os.path.join(base, f'{predicted_class}{filename}'))

