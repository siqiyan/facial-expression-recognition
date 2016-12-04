import os
import numpy as np
import caffe
import lmdb
from random import shuffle
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt

"""
This script will create a training lmdb and a test lmdb for KDEF database, and
the lmdb is ready for training in Caffe.
"""
train_map_size  = 8000000000 # the max size of the database, in bytes
test_map_size   = 600000000 # the max size of the database, in bytes

img_height  = 100 # output image height
img_width   = 100 # output image width

# label_ref = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA']
label_ref = ['NE', 'HA', 'AN', 'AF', 'DI', 'SA', 'SU']

def parse_label(filename):
    label = filename[4:6]
    angle = filename[6:8]
    if angle[0] != 'S':
        # Only keep frontal face
        return None
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

def crop_image(img, crop_ratio):
    h, w, _ = img.shape
    crop_len = int(np.round(w / 2 * crop_ratio))
    mid_h = int(np.round(h / 2))
    mid_w = int(np.round(w / 2))
    img = img[mid_h - crop_len:mid_h + crop_len + 1,
            mid_w - crop_len:mid_w + crop_len + 1, :]
    return img
    

def create_lmdb(img_src, db_name, map_size):
    count = 0
    env = lmdb.open(db_name, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in xrange(len(img_src)):
            src = img_src[i]
            filename = src.split('/')[-1]
            label = parse_label(filename)
            if label == None:
                continue
            img = np.array(Image.open(src))
            img = crop_image(img, 0.8)
            img = imresize(img, [img_height, img_width])
            img = np.transpose(img, [2, 0, 1])
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img.shape[0]
            datum.height = img.shape[1]
            datum.width = img.shape[2]
            datum.data = img.tostring()
            datum.label = label
            str_id = '{:08}'.format(i)
            txn.put(str_id, datum.SerializeToString())
            count += 1
    print '%s created, total images = %d' %(db_name, count)

if __name__ == '__main__':
    img_src = load_all_images_and_shuffle()
    print '%d images loaded' %(len(img_src))
    train_img_src = img_src[:4400]
    test_img_src = img_src[4400:]
    print 'Creating training database, this will take a few minutes, please be patient!'
    print 'If the program crashed, try increase train_map_size'
    create_lmdb(train_img_src, 'KDEF_train_lmdb', train_map_size)
    print 'Creating testing database, this will take less than a minute'
    print 'If the program crashed, try increase test_map_size'
    create_lmdb(test_img_src, 'KDEF_test_lmdb', test_map_size)
