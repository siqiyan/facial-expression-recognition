"""
This script generates a list for a dataset, in the following format:

path/to/imageA.jpg 0
path/to/imageB.jpg 5
path/to/imageC.jpg 2
path/to/imageD.jpg 1
...

This file list will be passed to $CAFFE_PATH/tools/convert_imageset.cpp to
generate a lmdb database, and then run $CAFFE_PATH/tools/compute_image_mean.cpp
on the lmdb database to compute the mean file.

Instead of running this script directly, the entire process is implemented in a
bash script in ./generate_lmdb.sh
"""
import os
from include import *

split_percentage = 0.8

# def load_all_images(root):
    # img_src = []
    # for folder in os.listdir(root):
        # folder = os.path.join(root, folder)
        # for src in os.listdir(folder):
            # ext = src.split('.')[-1].lower()
            # if ext != 'jpg':
                # continue
            # label = parse_label(src.split('/')[-1])
            # if label == None:
                # continue
            # src = os.path.join(folder, src)
            # img_src.append((src, label))
    # return img_src


def load_all_images(root):
    img_src = []
    for src in os.listdir(root):
        ext = src.split('.')[-1].lower()
        if ext != 'jpg':
            continue
        label = special_parse(src.split('/')[-1])
        if label == None:
            continue
        src = os.path.join(root, src)
        img_src.append((src, label))
    return img_src


def generate_list(listfile, img_src):
    with open(listfile, 'w') as writer:
        for src, label in img_src:
            src = '/'.join(src.split('/')[1:])
            writer.write(src + ' ' + str(label) + '\n')


if __name__ == '__main__':
    # root = 'dataset/KDEF'
    root = 'dataset/aug'
    img_src = load_all_images(root)
    sp = int(round(len(img_src) * split_percentage))
    # train_list = 'dataset/kdef_train.list'
    # test_list = 'dataset/kdef_test.list'
    train_list = 'dataset/kdef_aug_train.list'
    test_list = 'dataset/kdef_aug_test.list'
    generate_list(train_list, img_src[:sp])
    generate_list(test_list, img_src[sp:])
