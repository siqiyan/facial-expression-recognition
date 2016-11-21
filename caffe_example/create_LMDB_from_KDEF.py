import os
import shutil
import numpy as np
import caffe
import lmdb
from random import shuffle
from PIL import Image

"""
This script will create a training lmdb and a test lmdb for KDEF database, and
the lmdb is ready for training in Caffe.
"""
train_map_size = 8000000000 # the max size of the database, in bytes
test_map_size = 60000000 # the max size of the database, in bytes

label_ref = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA']

def parse_label(filename):
    label = filename[4:6]
    algle = filename[6:8]
    if not label in label_ref:
        print filename, label, 'unknown label'
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

def create_lmdb(img_src, db_name, map_size):
    env = lmdb.open(db_name, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in xrange(len(img_src)):
            src = img_src[i]
            filename = src.split('/')[-1]
            label = parse_label(filename)
            if label == None:
                continue
            img = np.array(Image.open(src))
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img.shape[2]
            datum.height = img.shape[0]
            datum.width = img.shape[1]
            datum.data = img.tostring()
            datum.label = label
            str_id = '{:08}'.format(i)
            txn.put(str_id, datum.SerializeToString())
            print '%d image added' %(i)
    print db_name, 'created.'

if __name__ == '__main__':
    img_src = load_all_images_and_shuffle()
    print '%d images loaded' %(len(img_src))
    train_img_src = img_src[:4400]
    test_img_src = img_src[4400:]
    create_lmdb(train_img_src, 'KDEF_train_lmdb', train_map_size)
    create_lmdb(test_img_src, 'KDEF_test_lmdb', test_map_size)
