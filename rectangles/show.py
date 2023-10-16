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
