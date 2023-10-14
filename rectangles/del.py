import cv2
import numpy as np

# Načtení obrázku
image_path = r'rotate\rotated_images\rotated_image.png'
img = cv2.imread(image_path, 0)  # Načítáme černobílý obrázek

# Definice jádra pro morfologický uzávěr (zde používáme obdélník)
kernel = np.ones((1,1), np.uint8)  # Velikost jádra můžete upravit

# Aplikace morfologického uzávěru
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Obrázek s uzávěrem", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
