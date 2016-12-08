import cv2
# import caffe
import numpy as np
from scipy.misc import imresize
import scipy.misc
import matplotlib.pyplot as plt

import subprocess
from subprocess import PIPE, Popen
import time

data_img_height = 100
data_img_width = 100
model_def = None
model_weights = None

# Change the following parameters as needed:
#mode = 'GPU'
mode = 'CPU'

def normalize_image(img):
    return img.astype(float) / 255.0

if __name__ == '__main__':

    if mode == 'GPU':
        pass
        #caffe.set_device(0)
        #caffe.set_mode_gpu()
    else:
        pass
        # caffe.set_mode_cpu()

    # Change process call here
    path = "/Users/kevingordon/darknet/"
    commands = [path+"./darknet", "classifier", "predict", path+"cfg/faces.data", path+"cfg/dec4_conv5_bn.cfg", path+"backup/dec4_conv5_bn_5.weights"]
    # Darknet Init
    p = subprocess.Popen(commands, stdin=PIPE)    
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_gray_eq = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        count = 0
        for x, y, w, h in faces:
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            data_img = frame            
            data_img = data_img[y:y+h, x:x+w, :]
            data_img = imresize(data_img, [data_img_height, data_img_width])

            cv2.imshow('face'+str(count), data_img)
            # scipy.misc.imsave('face.JPG', out[...,::-1])
            cv2.imwrite('face.JPG',data_img)

            print >> p.stdin, "face.JPG"
            time.sleep(0.08)

        frame_small = imresize(frame, [300, 300])

        cv2.imshow('frame', frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    # cv2.destroyAllWindows()
