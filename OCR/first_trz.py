from PIL import Image
import pytesseract
import numpy as np
import cv2

filename = 'OCR/cislo.jpg'
img = cv2.imread(filename)

print('using greyscale and threshold to sharpen.')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv2.filter2D(img, -1, sharpen_kernel)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


data = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
print(data)
cv2.imshow('image', img)
cv2.waitKey()