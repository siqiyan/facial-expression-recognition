import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_source = cv2.imread('demo.jpg')
    img_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img_source, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # plt.imshow(img_source)
    # plt.show()
    cv2.imshow('img_source', img_source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
