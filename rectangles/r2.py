import cv2

def show_image(picture, screen=1):
    #screen 0 notebook, screen 1 monitor
    height, width = picture.shape[:2]
    if screen == 0:
        final_height=750
    else:
        final_height=1000
    final_dimension=(round((final_height/height)*width),final_height)
    resized_picture = cv2.resize(picture, dsize=final_dimension)
    cv2.imshow('resized_picture', resized_picture)
    cv2.waitKey()
    cv2.destroyWindow('resized_picture')

image = cv2.imread('230926dr.jpg')
blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
show_image(blur)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
show_image(gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
show_image(thresh)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], 0, (0,255,0), 3)
    show_image(image)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(approx)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,36,12),5)




show_image(image)