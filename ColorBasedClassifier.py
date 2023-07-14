import cv2
import sys
import numpy as np
import itertools

FRUIT_LABELS = {"mango": 0, "banana": 1, "tomato": 2}
RIPENESS_LABELS = ["Unripe", "Semi-Ripe", "Ripe", "Overripe"]
CLASSIFIER_RGB_CONSTANTS = [
    {"fruit": "mango", 
     "Unripe":    [96, 124, 83], 
     "Semi-Ripe": [210, 174, 77], 
     "Ripe":      [232, 183, 88], 
     "Overripe":  [102, 54, 9]},
    
    {"fruit": "banana", 
     "Unripe":    [131, 178, 74], 
     "Semi-Ripe": [153, 149, 68], 
     "Ripe":      [248, 204, 55], 
     "Overripe":  [89, 54, 48]},
    
    {"fruit": "tomato", 
     "Unripe":    [157, 164, 112], 
     "Semi-Ripe": [193, 84, 7], 
     "Ripe":      [209, 53, 12], 
     "Overripe":  [133, 24, 13]}
]

def average_image_color(img_path):
    img = cv2.imread(img_path)
    average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
        
    rounded_arr = np.round(average_color).astype(int)
    print(f"RGB: {rounded_arr[0]}, {rounded_arr[1]}, {rounded_arr[2]}")
    
    return average_color
    
def evaluate_color_on_spectrum(average_color, fruit_name):
    ''' 
    Evalutes the given color on the spectrum of colors for the associated fruit
    to determine ripeness
    '''
    
    closest_color = None
    closest_distance = float('inf')
    
    print(CLASSIFIER_RGB_CONSTANTS[FRUIT_LABELS[fruit_name]])
    
    for color_name, rgb_values in itertools.islice(CLASSIFIER_RGB_CONSTANTS[FRUIT_LABELS[fruit_name]].items(), 1, None):
        curr_color = np.array(rgb_values, dtype=np.uint8)
        
        distance = np.linalg.norm(average_color - curr_color)
        
        if distance < closest_distance:
            closest_color = color_name
            closest_distance = distance
    
    return closest_color
        
def evaluate_ripeness(img_path, fruit_name):
    average_color = average_image_color(img_path)
    ripeness = evaluate_color_on_spectrum(average_color, fruit_name)
    return ripeness

if __name__ == "__main__":
    img_path = sys.argv[1]
    fruit_name = sys.argv[2]
    print(f"img_path: {img_path}")
    print(f"fruit_name: {fruit_name}")
    ripeness = evaluate_ripeness(img_path, fruit_name)
    print(ripeness)