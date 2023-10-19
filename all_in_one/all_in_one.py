from PIL import Image
import cv2
import numpy as np
import get_angle_for_rotation

def remove_tables(image, mask = Image.open("mask_tables.png")):
    result = Image.new("RGBA", image.size)
    result.paste(image.convert("RGBA"), (0,0))
    result.paste((255,255,255), (0, 0), mask)
    return result

def move_circle_to_middle():
    #zatím ne
    return 0

def remove_circles(image, mask = Image.open("mask_circles.png")):
    result = Image.new("RGBA", image.size)
    result.paste(image.convert("RGBA"), (0,0))
    result.paste((255,255,255), (0, 0), mask)
    return result

def enhance_image(img):
    k=220
    low = (0,0,0)
    high = (k,k,k)

    mask = cv2.inRange(img, low, high)
    mask = 255 - mask

    blured = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)
    mask2 = cv2.inRange(blured, 0, 200)

    mask2 = 255 - mask2

    return mask2

def find_rectangles(img, colored, base):

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    for i, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #colored = cv2.drawContours(colored, [cnt], -1, (0, 255, 0), 1)

        if len(approx) == 4:
            #colored = cv2.drawContours(colored, [cnt], -1, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(approx)
            if w+h>20:
                roi = base[y:y+h, x:x+w]
                
                # Save the extracted rectangle
                filename = f"all_in_one\extracted_rectangles/rectangle_{i}.jpg"
                cv2.imwrite(filename, roi)

    
    cv2.imshow("cv2_image", colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return colored

def rotate_drawing(image_cv2, image_PIL):

    #už nevím jak to funguje tak to dám jako odkaz
    angle = get_angle_for_rotation.main(image_cv2)    
    rotated_image = image_PIL.rotate(angle)
    return rotated_image

def rotate_rectangles():
    #zatím ne
    return 0

def cut_rectangle():
    return 0

def cut_area_around_rectangle():
    return 0

if __name__ == '__main__':
    image_path = '230926dr.jpg'
    image_PIL = Image.open(image_path)
    image_cv2 = cv2.imread(image_path)
    pic1=remove_tables(image_PIL)
    pic2=remove_circles(pic1)
    pic2_cv2 = cv2.cvtColor(np.array(pic2), cv2.COLOR_RGBA2BGR)
    enhanced = enhance_image(pic2_cv2)
    w_rectangles = find_rectangles(enhanced, pic2_cv2, image_cv2)
    cv2.imwrite("all_in_one/final_1.png",w_rectangles)
    rotated = rotate_drawing(pic2_cv2, pic2)
    rotated_cv2 = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGBA2BGR)
    enhanced_2_cv2 = enhance_image(rotated_cv2) 
    w_contours_cv2 = find_rectangles(enhanced_2_cv2,rotated_cv2, rotated_cv2)
    cv2.imshow("cv2_image", w_contours_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("all_in_one/final_2.png",w_contours_cv2)