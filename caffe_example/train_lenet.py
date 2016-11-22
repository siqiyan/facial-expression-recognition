import numpy as np
import caffe
'''
Simply train the network defined in lenet_solver.prototxt
and lenet_train_test.prototxt
'''

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('lenet_solver.prototxt')
    solver.solve()
