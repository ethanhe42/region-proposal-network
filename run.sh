export PYTHONPATH=`pwd`:/home/huyangyang/caffe-master/python:$PYTHONPATH
caffe train -gpu $1 -solver="/home/huyangyang/caffe-master/models/fasterRCNN/rpn_drn/solver.prototxt"
