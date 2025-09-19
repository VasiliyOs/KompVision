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

    white  = cv2.erode(treshold, core)
    wh_and_bl = cv2.dilate(white, core)

    cv2.imshow("Red after morph", wh_and_bl)

    moments = cv2.moments(treshold)

    S = moments['m00']
    M01 = moments['m01']
    M10 = moments['m10']

    info_display = np.zeros((200, 400, 3), dtype=np.uint8)

    cv2.putText(info_display, f"S: {S}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_display, f"M01: {M01}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_display, f"M10: {M10}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Red after morph", wh_and_bl)
    cv2.imshow("Info", info_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()