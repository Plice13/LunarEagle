import cv2
import numpy as np

def rotate_image(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the angle in degrees) and the sine and cosine of the angle
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account the translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the rotated image
    return cv2.warpAffine(image, M, (nW, nH))

# load the image
image = cv2.imread("230926dr.jpg")

# rotate the image by 45 degrees
rotated_image = rotate_image(image, 15.7)

# save the rotated image
cv2.imwrite("profi.jpg", rotated_image)