import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype='int')
    y = data[:,0]
    x = data[:,1:].astype('float32') / 255

    return x, y