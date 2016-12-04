import caffe
import os
import matplotlib.pyplot as plt

# Change the following parameters as needed:
mode = 'GPU'
# mode = 'CPU'
max_iter = 5000
test_interval = 500
test_iter = 100

if __name__ == '__main__':
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    solver = caffe.get_solver('solver_adam.prototxt')

    train_loss = zeros(max_iter)
    test_acc = zeros(int(np.ceil(max_iter / test_interval)))

    for i in xrange(max_iter):
        solver.step(1)
        train_loss = solver.net.blobs['loss'].data
        if i % test_interval == 0:
            correct = 0
            for test_i in xrange(
