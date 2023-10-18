import cv2
import numpy as np

def show_image(picture, screen=1):
    # screen 0 notebook, screen 1 monitor
    height, width = picture.shape[:2]
    if screen == 0:
        final_height = 750
    else:
        final_height = 1000
    final_dimension = (round((final_height / height) * width), final_height)
    resized_picture = cv2.resize(picture, dsize=final_dimension)
    cv2.imshow('resized_picture', resized_picture)
    cv2.waitKey()
    cv2.destroyWindow('resized_picture')

def main(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #show_image(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    # Create a copy of the original image to draw contours on
    img_with_contours = img.copy()

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        img_with_contours = cv2.drawContours(img_with_contours, [cnt], -1, (0, 255, 0), 1)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            img_with_contours = cv2.drawContours(img_with_contours, [cnt], -1, (0, 0, 255), 3)

    show_image(img_with_contours)
    cv2.imwrite(f'rectangles/mode/mode.png', img_with_contours)

if __name__ == "__main__":
    image_path = r'del.png'
    main(image_path)