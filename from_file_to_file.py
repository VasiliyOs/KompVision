from random import randint

import cv2
import numpy as np


cap = cv2.VideoCapture(r'video.mp4',cv2.CAP_ANY)
path = "video2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vido = cv2.VideoWriter(path, fourcc, 25, (width,height))
while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break

    frame = frame
    vido.write(frame)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


vido.release()
cap.release()
cv2.destroyAllWindows()