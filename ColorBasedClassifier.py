import cv2
import sys
import numpy as np

RIPENESS_LABELS = ["Unripe", "Semi-Ripe", "Ripe", "Overripe"]
CLASSIFIER_HSV_CONSTANTS = [
    {"fruit": "mango", "Unripe": [0, 0, 0], "Semi-Ripe": [0,0,0], "Ripe": [0, 0, 0], "Overripe": [0, 0, 0]},
    {"fruit": "banana", "Unripe": [0, 0, 0], "Semi-Ripe": [0,0,0], "Ripe": [0, 0, 0], "Overripe": [0, 0, 0]},
    {"fruit": "tomato", "Unripe": [0, 0, 0], "Semi-Ripe": [0,0,0], "Ripe": [0, 0, 0], "Overripe": [0, 0, 0]},
]

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
    
def evaluate_color_on_spectrum(H,S,V, fruit_name):
    ''' 
    Evalutes the given color on the spectrum of colors for the associated fruit
    to determine ripeness
    '''
    if H<20 and S>50 and V>50:      # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 0
    elif H<20 and S>50 and V<50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 1
    elif H<20 and S<50 and V>50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 2
    elif H<20 and S<50 and V<50:    # ARBITRARY VALUES MUST BE UPDATED TO REFLECT ACTUAL VALUES
        return 3
    
def evaluate_ripeness(img_path, fruit_name):
    H, S, V = average_image_color(img_path)
    ripeness = evaluate_color_on_spectrum(H,S,V, fruit_name)
    return RIPENESS_LABELS[ripeness]



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ColorBasedClassifier.py <image_path> <fruit_name>")
        sys.exit()
    else:
        img_path = sys.argv[1]
        fruit_name = sys.argv[2]
        ripeness = evaluate_ripeness(img_path, fruit_name)
        print(ripeness)