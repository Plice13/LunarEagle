import numpy as np
import cv2 as cv
img = cv.imread('IMG_1124(2).JPG', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,10,
 param1=60,param2=1,minRadius=1,maxRadius=2)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
 # draw the outer circle
 cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
 # draw the center of the circle
 cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.imwrite('idk.png', cimg)
cv.waitKey(0)
cv.destroyAllWindows()