import caffe
import os

# Change the following parameters as needed:
mode = 'GPU'
# mode = 'CPU'
state_file = 'snapshot_iter_8000.solverstate'

if __name__ == '__main__':
    assert os.path.isfile(state_file)
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    solver = caffe.get_solver('solver_adam.prototxt')
    solver.restore(state_file)
    solver.solve()
