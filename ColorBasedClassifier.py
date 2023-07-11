import cv2
import numpy as np

def average_image_color(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Hue, Saturation, Value
    H_sum = 0
    S_sum = 0
    V_sum = 0
    H_avg = 0
    S_avg = 0
    V_avg = 0
    pixel_count = 0

    height, width, channels = hsv_img.shape

    for y in range(height):
        for x in range(width):
            H, S, V = hsv_img[y, x]

            if H!=0 or S!=0 or V!=0:
                H_sum += H
                S_sum += S
                V_sum += V
                pixel_count += 1

    if pixel_count == 0:
        return 0, 0, 0
    else:
        H_avg = H_sum / pixel_count
        S_avg = S_sum / pixel_count
        V_avg = V_sum / pixel_count
        return H_avg, S_avg, V_avg
    
def evaluate_color_on_spectrum(H,S,V):
    ''' 
    Evalutes the given color on the spectrum of colors for the associated fruit
    to determine ripeness
    '''
    if H<20 and S>50 and V>50:      # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 1
    elif H<20 and S>50 and V<50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 2
    elif H<20 and S<50 and V>50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 3
    elif H<20 and S<50 and V<50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 4