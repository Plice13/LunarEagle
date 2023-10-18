import cv2
import numpy as np
import show

# read input
img = cv2.imread("230926dr.jpg")

k=225

low = (0,0,0)
high = (k,k,k)

mask = cv2.inRange(img, low, high)
mask = 255 - mask
show.show_image(mask, 0)

# save output
blured = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)
show.show_image(blured,0)
mask2 = cv2.inRange(blured, 0, 200)
show.show_image(mask2, 0)

mask2 = 255 - mask2

cv2.imwrite(f'zkous{k}.png', mask2)
show.show_image(mask2, 0)
