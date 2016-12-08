#!/bin/bash

# This script generate lmdb database directly in one step.

#rm -rf dataset/kdef_train
#rm -rf dataset/kdef_test
#rm -f dataset/kdef_train.list
#rm -f dataset/kdef_test.list
#python make_list.py
#convert_imageset -resize_height 100 -resize_width 100 dataset/ dataset/kdef_train.list dataset/kdef_train
#convert_imageset -resize_height 100 -resize_width 100 dataset/ dataset/kdef_test.list dataset/kdef_test
#compute_image_mean dataset/kdef_train dataset/kdef_train_mean.binaryproto
#compute_image_mean dataset/kdef_test dataset/kdef_test_mean.binaryproto


rm -rf dataset/kdef_aug_train
rm -rf dataset/kdef_aug_test
rm -f dataset/kdef_aug_train.list
rm -f dataset/kdef_aug_test.list
python make_list.py
convert_imageset dataset/ dataset/kdef_aug_train.list dataset/kdef_aug_train
convert_imageset dataset/ dataset/kdef_aug_test.list dataset/kdef_aug_test
compute_image_mean dataset/kdef_aug_train dataset/kdef_aug_train_mean.binaryproto
compute_image_mean dataset/kdef_aug_test dataset/kdef_aug_test_mean.binaryproto
