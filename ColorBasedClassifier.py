import cv2
import sys
import numpy as np
import itertools
import os

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

def crop_image(img_path, save_cropped_img=False, log=False):
    img = cv2.imread(img_path)
    grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # get binary mask
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # get contours
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    croppedImg = img
    
    # Look for the outer bounding boxes:
    for i, c in enumerate(contours):

        # restrict search to only parent contours
        # (ignores inner contours; e.g. holes)
        if hierarchy[0][i][3] == -1:

            # Get contour area:
            contourArea = cv2.contourArea(c)
            # Set minimum area threshold:
            minArea = 0.025 * img.size
            if log and contourArea > 100: print(f"minArea: {minArea}")
            if log and contourArea > 100: print(f"contourArea: {contourArea}")

            # Look for the largest contour:
            if contourArea > minArea:
                
                # Approximate the contour to a polygon:
                contoursPoly = cv2.approxPolyDP(c, 3, True)
                # Convert the polygon to a bounding rectangle:
                boundRect = cv2.boundingRect(contoursPoly)

                # Set the rectangle dimensions:
                rectangleX = boundRect[0]
                rectangleY = boundRect[1]
                rectangleWidth = boundRect[0] + boundRect[2]
                rectangleHeight = boundRect[1] + boundRect[3]

                # Draw the rectangle:
                croppedImg = img[rectangleY:rectangleHeight, rectangleX:rectangleWidth]
                if save_cropped_img: cv2.imwrite("croppedImg.png", croppedImg)
            
    return croppedImg
    
def average_image_color(img_path=None, read_from_path=False, img=None, log=False):
    if read_from_path: img = cv2.imread(img_path)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # exclude pixels with low saturation or low value
    mask = cv2.inRange(hsv, (0, 50, 50), (255, 255, 255))
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # compute average color of remaining pixels
    average_color_hsv = cv2.mean(masked_hsv)
    average_color = np.array(average_color_hsv[:3], dtype=np.uint8)
    
    rgb_color = cv2.cvtColor(np.array([[average_color]]), cv2.COLOR_HSV2RGB)[0][0]
    rounded_arr = np.round(rgb_color).astype(int)
    
    if log: print(f"RGB: {rounded_arr[0]}, {rounded_arr[1]}, {rounded_arr[2]}")        
    
    return rgb_color
    
def evaluate_color_on_spectrum(average_color, fruit_name, log=False):
    ''' 
    Evalutes the given color on the spectrum of colors for the associated fruit
    to determine ripeness
    '''
    
    closest_color = None
    closest_distance = float('inf')
    
    # if log: print(CLASSIFIER_RGB_CONSTANTS[FRUIT_LABELS[fruit_name]])        
    if log: print(fruit_name)        
    
    for color_name, rgb_values in itertools.islice(CLASSIFIER_RGB_CONSTANTS[FRUIT_LABELS[fruit_name]].items(), 1, None):
        curr_color = np.array(rgb_values, dtype=np.uint8)
        
        distance = np.linalg.norm(average_color - curr_color)
        
        if distance < closest_distance:
            closest_color = color_name
            closest_distance = distance
    
    return closest_color
        
def evaluate_ripeness(img_path, fruit_name):
    img = crop_image(img_path, save_cropped_img=True, log=False)
    average_color = average_image_color(img=img, log=True)
    ripeness = evaluate_color_on_spectrum(average_color, fruit_name, log=True)
    return ripeness

if __name__ == "__main__":
    img_path = sys.argv[1]
    fruit_name = sys.argv[2]
    # print(f"img_path: {img_path}")
    # print(f"fruit_name: {fruit_name}")
    ripeness = evaluate_ripeness(img_path, fruit_name)
    print(ripeness)