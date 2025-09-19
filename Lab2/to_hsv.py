import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()