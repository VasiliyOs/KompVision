from random import randint

import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.0.216:8080/video")
path = "video_telephone.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_width = width // 2
center_height = height // 2
vido = cv2.VideoWriter(path, fourcc, 25, (width,height))
while cap.isOpened():
    ret, frame = cap.read()
    if not(ret):
        break

    b, g, r = frame[center_height, center_width]
    if r > g and r > b:
        color = (0, 0, 255)
    elif g > r and g > b:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    cv2.rectangle(frame,
                  (center_width - 10, center_height - 50),
                  (center_width + 10, center_height + 50),
                  (0, 0, 255),
                  5)
    cv2.rectangle(frame,
                  (center_width + 50, center_height - 10),
                  (center_width - 50, center_height + 10),
                  (0, 0, 255),
                  5)

    cv2.rectangle(frame,
                  (center_width - 9, center_height - 49),
                  (center_width + 9, center_height + 49),
                  color,
                  -1)
    cv2.rectangle(frame,
                  (center_width + 49, center_height - 9),
                  (center_width - 49, center_height + 9),
                  color,
                  -1)



    frame = frame
    vido.write(frame)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


vido.release()
cap.release()
cv2.destroyAllWindows()
