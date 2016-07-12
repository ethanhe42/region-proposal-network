#-.-encoding=utf-8-.-``
# Yihui He, https://yihui-he.github.io
import sys
import pandas as pd
sys.path.append("/home/yihuihe/rpn_drn_new")
sys.path.insert(0, "/home/yihuihe/miscellaneous/caffe/python")
print sys.path
import caffe
print caffe.__file__
import numpy as np
import cv2
from utils import NetHelper, CaffeSolver
import os
import matplotlib.pyplot as plt
from ultrasound import rawData
debug=True

# init
caffe.set_device(1)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

class solverWrapper(object):
    def __init__(self):
        self.solver=caffe.SGDSolver("solver.prototxt")
        # self.solver.net.layers[0].set_data_queue(rawData())
        # self.solver.net.copy_from("init.caffemodel")
        self.solver.net.copy_from("rpn_drn_iter_20000.caffemodel")
    
    def train_model(self):
        for iter in range(500*2000):
            if debug:
                if iter % 100 == 0 and iter !=0:
                    nethelper=NetHelper(self.solver.net)
                    # nethelper.hist('label')
                    # nethelper.hist('prob', filters=2,attr="blob")
                    # nethelper.hist('data', filters=2,attr="blob")

                    if False:
                        for i in range(nethelper.net.blobs['data'].data.shape[0]):
                            plt.subplot(221)
                            plt.imshow(nethelper.net.blobs['data'].data[i,0])
                            plt.subplot(222)
                            plt.imshow(nethelper.net.blobs['prob'].data[i,0])
                            plt.subplot(223)
                            plt.imshow(nethelper.net.blobs['label'].data[i,0])
                            plt.show()
                        
            self.solver.step(1)




if __name__=="__main__":
    solve=solverWrapper()
    solve.train_model()


