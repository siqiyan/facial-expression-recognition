import os
import shutil

"""
I am going to use LMDB, therefore, this script is no longer useful.

This script should be put into the same directory as KDEF.
It will generate a new directory called data_dir, and should be ready for this
script:
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
which will generate training data for the Inception model.

The structure is like:
reformat_KDEF.py
KDEF
|-  AF01
|-  AF02
.
.
.
data_dir
|-  afraid
    |-  oneimage.jpg
    |-  anotherimage.jpg
    .
    .
    .
|-  angry
|-  disgusted
.
.
.
"""
label_ref = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA']
label_path_list = []

def parse_label(filename):
    label = filename[4:6]
    algle = filename[6:8]
    if not label in label_ref:
        print filename, label, 'unknown label'
        return None
    return label_ref.index(label)

def make_dir(label):
    _path = os.path.join('data_dir', label)
    if not os.path.exists(_path):
        os.makedirs(_path)
    return _path

def make_all_dir():
    global label_path_list
    for label in label_ref:
        label_path_list.append(make_dir(label))

def copy_image_from_folder(path):
    image_list = os.listdir(path)
    for img in image_list:
        ext = img.split('.')[-1]
        if ext.lower() != 'jpg':
            continue
        i = parse_label(img)
        if i == None:
            continue
        img = os.path.join(path, img)
        shutil.copy(img, label_path_list[i])

def copy_all_images():
    all_src_folder = os.listdir('KDEF')
    for src in all_src_folder:
        src = os.path.join('KDEF', src)
        copy_image_from_folder(src)

if __name__ == '__main__':
    make_all_dir()
    copy_all_images()
