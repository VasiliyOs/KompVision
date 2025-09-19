import cv2
import numpy as np

cap = cv2.VideoCapture(0)

core = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break

    h1 = np.array((0, 90, 90))
    h2 = np.array((5, 255, 255))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    treshold = cv2.inRange(frame, h1, h2)

    cv2.imshow("HSV", frame)
    cv2.imshow("HSV only Red", treshold)

    white  = cv2.erode(treshold, core)
    niggas = cv2.dilate(treshold, core)

    cv2.imshow("White", white)
    cv2.imshow("Black", niggas)

    wh_and_bl = cv2.dilate(white, core)

    cv2.imshow("All", wh_and_bl)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()