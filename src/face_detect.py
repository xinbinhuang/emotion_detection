#! /usr/bin/env python

import os
import cv2
import numpy as np


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                     minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    # get (x1, y1), (x2, y2) = (x1, y1), (x1 + w, y1 + h)
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color=(255, 0, 0)):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def draw_str(img, str, position, color=(255, 255, 0)):
    cv2.putText(img, str, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


if __name__ == "__main__":

    haarcascade_path = os.path.join("..", "haarcascade_files")
    face_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_path,
                                         'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_path,
                                        'haarcascade_eye.xml'))

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        while True:
            status, frame = cap.read()

            if status:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                vis = frame.copy()
                faces = detect(gray, face_cascade)
                draw_rects(vis, faces, (0, 255, 0))
                for x1, y1, x2, y2 in faces:
                    roi_gray = gray[y1:y2, x1:x2]
                    roi_vis = vis[y1:y2, x1:x2]
                    eyes = detect(roi_gray, eye_cascade)
                    draw_rects(roi_vis, eyes, (255, 0, 0))

            draw_str(vis, "Press 'q' to exit", (10, 20), (255, 255, 0))
            cv2.imshow("Face Detection", vis)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
