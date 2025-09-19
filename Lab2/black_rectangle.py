import cv2
import numpy as np

cap = cv2.VideoCapture(0)

core = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break
    orig_frame = frame

    h1 = np.array((0, 90, 90))
    h2 = np.array((5, 255, 255))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    treshold = cv2.inRange(frame, h1, h2)

    cv2.imshow("HSV", frame)

    white  = cv2.erode(treshold, core)
    wh_and_bl = cv2.dilate(white, core)

    cv2.imshow("Red after morph", wh_and_bl)

    moments = cv2.moments(treshold)

    S = moments['m00']
    M01 = moments['m01']
    M10 = moments['m10']
    if S>0:
        x_center = int(M10 / S)
        y_center = int(M01 / S)

        cv2.circle(orig_frame, center=(x_center, y_center), radius=4, color=(70, 255, 255), thickness=-1)

        contours, _ = cv2.findContours(wh_and_bl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    cv2.imshow("Red after morph", wh_and_bl)
    cv2.imshow("Original",orig_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()