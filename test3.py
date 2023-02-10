import cv2
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture(1)


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


make_1080p()
change_res(1920, 1080)



while (True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, 3):

        template = cv2.imread('Lab2Template/trackdot' + str(i) + '.bmp', 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)

        threshold = 0.80

        loc = np.where(res >= threshold)
        for box in zip(*loc[::-1]):
            cv2.rectangle(frame, box, (box[0] + w, box[1] + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()