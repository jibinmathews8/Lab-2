import cv2
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
cv2.imshow('Original', frame)

while True:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    # Bring in Template (Scaled Properly with 1920 x 1080 with the camera 6 feet from ground)

    template = cv2.imread('Lab2Template/trackdot1.bmp', 0)

    w, h = template.shape[::-1]  # Width and Height of Template

    res = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)

    threshold = .9

    loc = np.where(res >= threshold)

    for box in zip(*loc[::-1]):
        cv2.rectangle(frame, box, (box[0]+w, box[1]+h), (0, 0, 255), 2)
    cv2.imshow('Applied Match Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

