import caffe
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import os
import shutil
from random import shuffle

model_def = 'conv5_bn_test_v0.1.prototxt'
model_weights = 'snapshot_iter_15000.caffemodel'
label_ref = ['NE', 'HA', 'AN', 'AF', 'DI', 'SA', 'SU']

def load_image(test_image):
    image = caffe.io.load_image(test_image)
    image = image.astype(np.float32) / 255.0
    image = imresize(image, [135, 100])
    image = np.transpose(image, [2, 0, 1])
    image = image[np.newaxis, ...]
    return image

def parse_label(filename):
    label = filename[4:6]
    algle = filename[6:8]
    if not label in label_ref:
        # print filename, label, 'unknown label'
        return None
    return label_ref.index(label)

def load_all_images_and_shuffle():
    root = 'KDEF'
    img_src = []
    for folder in os.listdir(root):
        folder = os.path.join(root, folder)
        for src in os.listdir(folder):
            ext = src.split('.')[-1].lower()
            if ext != 'jpg':
                continue
            src = os.path.join(folder, src)
            img_src.append(src)
    shuffle(img_src)
    return img_src

if __name__ == '__main__':
    img_src = load_all_images_and_shuffle()
    print '%d images loaded' %(len(img_src))
    img_src = img_src[:1000]
    print '%d images selected' %(len(img_src))
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    count = 0
    right = 0
    for src in img_src:
        fname = src.split('/')[-1]
        idx = parse_label(fname)
        net.blobs['data'].data[...] = load_image(src)
        output = net.forward()
        prob = output['prob'][0]
        prediction = np.argmax(prob)
        count += 1
        right += (prediction == idx)
        print 'count = %d, right = %d, accuracy = %f' \
            %(count, right, float(right) / float(count))
