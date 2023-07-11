import cv2
import numpy as np

def average_image_color(img):
    # Resize image to 1x1 pixel
    img = cv2.resize(img, (1, 1), interpolation=cv2.INTER_AREA)
    # Return the average c