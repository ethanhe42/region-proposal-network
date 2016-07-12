import caffe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

def transform_bb(bb0,w,h):
    bb = np.array([0,0,0,0])
    bb[2]=np.exp(bb0[2])*w
    bb[3]=np.exp(bb0[3])*h
    bb[0]=bb0[0]*w+w/2 - 0.5 * bb[2]
    bb[1]=bb0[1]*h+h/2 - 0.5 * bb[3]
    return bb

def non_max_suppression_slow(boxes,probs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0]+boxes[:,2]-1
    y2 = boxes[:,1]+boxes[:,3]-1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


class FasterRCNN_Loss(caffe.Layer):
    def setup(self, bottom, top):
        self.reg_loss_weight = 0.1
        self.phase = eval(self.param_str)['phase']  
        self.iter = 0
        self.py_fn = __file__.split('/')[-1]
        self.py_dir = os.path.dirname(__file__)
        
    def reshape(self, bottom, top):
        self._name_to_bottom_map={'conv8_cls':0,'conv8_reg':1,'label':2,'sampling_param':3,'data':4}
        top[0].reshape(1)

    def forward(self, bottom, top):
        sampling_param = bottom[3].data
        tags = bottom[2].data
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        
        batch_size = len(sampling_param)
        cls_loss = 0.0
        correct_count = np.array([0,0],dtype=np.float32)
        probs = np.zeros(batch_size,dtype=np.float32)     
        for i in range(batch_size):
            x_index = sampling_param[i][4]
            y_index = sampling_param[i][5]
            size_index = sampling_param[i][6]
            
            cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
            cls_pred = cls_conv[0,2*size_index : 2*size_index+2 ,y_index,x_index]#real number
            cls_pred_max = np.max(cls_pred)
            cls_loss += cls_pred[int(cls_label)] - cls_pred_max - np.log(np.sum(np.exp(cls_pred - cls_pred_max))) 
            
            probs[i] = 1.0 / np.sum(np.exp(cls_pred - cls_pred[1]))
            if int(cls_pred[1] > cls_pred[0]) == int(cls_label):
                correct_count[int(cls_label)] += 1
                 
        cls_loss = - cls_loss/batch_size
        cls_acc = (correct_count[0]+correct_count[1])/batch_size
               
        reg_loss = 0
        pos_count = 0
        for i in range(batch_size):
            x_index = sampling_param[i][4]
            y_index = sampling_param[i][5]
            size_index = sampling_param[i][6]
            
            cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
            if cls_label < 0.5:
                continue
            
            pos_count += 1
            reg_label = tags[0,5*size_index+1 : 5*size_index+5 ,y_index,x_index]
            reg_pred = reg_conv[0, 4*size_index : 4*size_index+4 ,y_index,x_index]
            reg_loss += np.sum((reg_label - reg_pred)**2)
        
        if pos_count > 0:
            reg_loss = reg_loss/2.0/pos_count            
        
        top[0].data[...] = cls_loss + self.reg_loss_weight * reg_loss
        
	if correct_count[1] + batch_size - pos_count - correct_count[0] == 0:#no positive prediction
		cls_precision = np.float32(correct_count[1] == 0)
	else:
	        cls_precision = correct_count[1] / (correct_count[1] + batch_size - pos_count - correct_count[0])

        cls_recall = correct_count[1] / pos_count
                
        if self.phase == 'TRAIN':
            print '[%s] Train net output #1: acc = %f' % (self.py_fn,cls_acc)
            print '[%s] Train net output #2: cls_loss = %f' % (self.py_fn,cls_loss)
            print '[%s] Train net output #3: reg_loss = %f' % (self.py_fn,reg_loss)
            print '[%s] Train net output #4: precision = %f' % (self.py_fn,cls_precision)
            print '[%s] Train net output #5: recall = %f' % (self.py_fn,cls_recall)
        else:
            print '[%s] Test net output #1: acc = %f' % (self.py_fn,cls_acc)
            print '[%s] Test net output #2: cls_loss = %f' % (self.py_fn,cls_loss)
            print '[%s] Test net output #3: reg_loss = %f' % (self.py_fn,reg_loss)
            print '[%s] Test net output #4: precision = %f' % (self.py_fn,cls_precision)
            print '[%s] Test net output #5: recall = %f' % (self.py_fn,cls_recall)
        sys.stdout.flush()
  
        if 0:#self.phase == 'TEST' or self.iter % 10 == 0:
            img0 = bottom[4].data[0,:,:,:]
            img = np.transpose(img0, (1,2,0)) + 0.5
            
            plt.clf()
            ax = plt.gca()
            plt.imshow(img)        
            resize_width = 320
            resize_height = 240
            sliding_window_width =  [20,20, 30,30, 40,40, 50,50]
            sliding_window_height = [30,40, 45,60, 60,80, 75,100] 
            sliding_window_stride = 8 
            cand_bbs = list()
            cand_probs = list()
            for size_index in range(len(sliding_window_height)):
                for y_index in range(resize_height/sliding_window_stride):
                    for x_index in range(resize_width/sliding_window_stride):
                        h = sliding_window_height[size_index]
                        w = sliding_window_width[size_index]
                        x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
                        y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2

                        prob = np.exp(cls_conv[0, 2*size_index+1, y_index, x_index])/np.sum(
                                np.exp(cls_conv[0, 2*size_index : 2*size_index+2, y_index, x_index]))
                        #print prob

                        if prob>0.50:   
                            bb = np.zeros(4,dtype=np.float32)          
                            for i in range(4):
                                bb[i] = reg_conv[0, 4*size_index+i, y_index, x_index]
                            bb=transform_bb(bb ,w,h) + np.array([x,y,0,0])
                            cand_bbs.append(bb)
                            cand_probs.append(prob)

            cand_bbs=np.array(cand_bbs)
            cand_probs = np.array(cand_probs)  
                  
#             ind = np.argsort(cand_probs)
#             ind = ind[::-1]
#             bb_num_show = np.min([32, len(cand_bbs)])
#             cand_bbs = cand_bbs[ind[:bb_num_show]]
#             cand_probs = cand_probs[ind[:bb_num_show]]
#             for i in range(bb_num_show):
#                 bb = cand_bbs[i]
#                 prob = cand_probs[i]
#                 ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3],facecolor='none',edgecolor=(1,1-prob,1-prob)))          
#                 plt.text(bb[0],bb[1],'%.3f' % prob,color='b')
            
            
            nms_bbs = non_max_suppression_slow(cand_bbs,cand_probs,0.3)
            for bb in nms_bbs:
                ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3],facecolor='none',edgecolor=(0,1,0)))        
            plt.text(10,10,'iter=%06d, precision=%.3f, recall=%.3f' % (self.iter,cls_precision,cls_recall),color='r')
            
            plt.savefig('%s/snapshot/%s_%06d.jpg' % (self.py_dir,self.phase,self.iter),dpi=100)            
#             plt.show()
        self.iter += 1        
        
    def backward(self, top, propagate_down, bottom):
        sampling_param = bottom[3].data
        tags = bottom[2].data
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        
        batch_size = len(sampling_param)
        
        if propagate_down[0]:
            cls_diff = np.zeros_like(cls_conv, dtype=np.float32)        
            for i in range(batch_size):
                x_index = sampling_param[i][4]
                y_index = sampling_param[i][5]
                size_index = sampling_param[i][6]
            
                cls_label = tags[0,5*size_index,y_index,x_index]    
                cls_pred = cls_conv[0,2*size_index : 2*size_index+2 ,y_index,x_index]#real number                        
                
                if cls_pred[1] - cls_pred[0] < 20:
                    cls_diff[0,  2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[0])) - np.float32(cls_label<0.5)
                else:
                    cls_diff[0,  2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label<0.5)
                
                if cls_pred[0] - cls_pred[1] < 20:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[1])) - np.float32(cls_label>0.5)
                else:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label>0.5)
                
            bottom[0].diff[...] = cls_diff/batch_size
        
        if propagate_down[1]:
            reg_diff = np.zeros_like(reg_conv, dtype=np.float32)
            pos_count = 0
            for i in range(batch_size):
                x_index = sampling_param[i][4]
                y_index = sampling_param[i][5]
                size_index = sampling_param[i][6]
            
                cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
                if cls_label < 0.5:
                    continue
                pos_count += 1
                
                reg_label = tags[0,5*size_index+1 : 5*size_index+5, y_index, x_index]
                reg_pred = reg_conv[0,4*size_index : 4*size_index+4, y_index, x_index]
                reg_diff[0,4*size_index : 4*size_index+4, y_index, x_index] += reg_pred - reg_label
            
            if pos_count > 0:
                reg_diff = reg_diff/pos_count
            
            bottom[1].diff[...] = self.reg_loss_weight * reg_diff
        
