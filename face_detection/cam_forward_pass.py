import cv2
import caffe
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

data_img_height = 135
data_img_width = 100
model_def = None
model_weights = None

# Change the following parameters as needed:
mode = 'GPU'
# mode = 'CPU'

def normalize_image(img):
    return img.astype(float) / 255.0

if __name__ == '__main__':
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for x, y, w, h in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            data_img = frame[..., ::-1] # convert BGR to RGB
            data_img = data_img[y:y+h, x:x+w, :]
            data_img = imresize(data_img, [data_img_height, data_img_width])
            data_img = normalize_image(data_img) 
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    # cv2.destroyAllWindows()
