#!/bin/bash

# I use this script to load the correct driver for my webcam, but normally
# you be able to run cam_detection.py directly.
bash -c 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libv4l/v4l1compat.so python cam_detection.py'
