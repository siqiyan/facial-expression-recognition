import caffe
import os

# Change the following parameters as needed:
mode = 'GPU'
# mode = 'CPU'

if __name__ == '__main__':
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    solver = caffe.get_solver('solver_adam.prototxt')
    solver.solve()
