from random import randint

import cv2
import numpy as np


cap = cv2.VideoCapture(r'video.mp4',cv2.CAP_ANY)
while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break

    frame = cv2.resize(frame,(randint(500,600), randint(500,600))) #Видева жёстко флексит
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()