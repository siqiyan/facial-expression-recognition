#!/bin/bash
caffe train \
    --solver=solver/full_solver.prototxt \
    --snapshot=model_weights/data_aug_mix_iter_10000.solverstate
