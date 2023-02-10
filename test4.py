import cv2
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
import numpy as np

# np.set_printoptions(threshold=np.nan)

# Steps for Tracking the Rover
# 1. Capturing Webcam Data
# 2. Identify template being used for matching (Find its dimensions)
# 3. Set Threshold for which the matching is compared to in order to identify a match
# 4. Apply Template Matching (cv2.matchTemplate)
# 5. Once match is found mark its location (*loc[::-1])
# 6. Apply Non-Max Suppression (Used to get exact match rather than getting several matches for a single match)
# 7. Get Location of the Non Max Suppression templates
# 8. Figure out which points are missing (ie. Rover covering multiple dots on the track)
# 9. Compare Current frame with previous frame and determine the rate at which the dots are
#       disappearing and reappearing
# 10. Since there are 64 dots the rover should be able to cover some of them in one second and uncover them at a
#       similar rate

# Notes: Location points are available before Non Max Suppression however there are a large amount of matches
#           being detected which measures an even larger amount of data points
#        After Applying the Non-Maximum suppression, the list of numbers only represented one dot and would not
#           give the locations of all the dot


cap = cv2.VideoCapture(1)

print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("Frame resolution set to: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")

while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Gray Image

    template = cv2.imread('Lab2Template/test1080.bmp', 0)
    w, h = template.shape[::-1]  # Used to invert array

    # Template Matching using Normalized Cross Correlation
    # Function cv2.matchTemplate( Original Image, Template, and Method of Matching)

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)  # Try changing methods
    threshold = 0.87
    loc = np.where(res >= threshold)

    circ = []
    count = 0

    for box in zip(*loc[::-1]):
        # cv2.rectangle(frame, box, (box[0] + w, box[1] + h), (0, 0, 255), 1)
        # circ.append((box[0], box[1], (box[0] + w), (box[1] + h)))
        # print(*loc[::-1])
        c1 = ((box[0]) + (box[0] + w)) / 2  # X coordinate for center of circle
        c2 = ((box[1]) + (box[1] + h)) / 2  # Y coordinate for center of circle
        # radius = 3.5
        #   cv2.circle(frame, (int(c1), int(c2)), radius, (0, 0, 255), 1) # Before Non Max Suppression
        count += 1

    # Apply Non Max Suppression

    circ = []

    for box in zip(*loc[::-1]):
        circ.append((box[0], box[1], box[0] + w, box[1] + h))
    pick = non_max_suppression(np.array(circ))
    print(len(pick))

    for (startX, startY, endX, endY) in pick:
        NMS_x = (startX + endX) / 2
        NMS_y = (startY + endY) / 2
        value = NMS_x, NMS_y
        # draw the bounding box on the image
        cv2.circle(frame, (int(NMS_x), int(NMS_y)), 11, (0, 0, 255), 2)  # After Non Max Suppression
        #print(NMS_x)

    cv2.imshow('MatchedTemplate', frame)

    # Now we need to compare frames to find where the rover is with respect to the course

    # count_frame = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     cv2.imshow('window-name', frame)
    #     cv2.imwrite("frame%d.jpg" % count_frame, frame)
    #     count_frame = count_frame + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
