import cv2
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture(1)  # Webcam Capture

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Gray Image

    template = cv2.imread('Lab2Template/trackdot1.bmp', 0)
    w, h = template.shape[::-1]  # Used to invert array

    # Template Matching using Normalized Cross Correlation
    # Function cv2.matchTemplate( Original Image, Template, and Method of Matching)

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED) # Try changing methods

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Find center point
    # Midpoint_x = (top_left[0]+bottom_right[0])/2,
    # Midpoint_y = (top_left[1]+bottom_right[1])/2
    # center = (Midpoint_x, Midpoint_y)
    # radius = 10

    cv2.rectangle(frame, top_left, bottom_right, 255, 2)
    # cv2.circle(frame, center, radius, 255, 2)
    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
