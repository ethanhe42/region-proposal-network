import numpy as np
from scipy import misc
import h5py 
import sys
import caffe
import os
import cython_bbox
from ultrasound import rawData
import cfgs

def random_zoomout(img0):
    zoom_ratio = 1.0 #np.random.rand()*0.8 + 0.2
    sz = np.shape(img0)
    h = np.int32(zoom_ratio*sz[0])
    w = np.int32(zoom_ratio*sz[1])
    p = [np.random.randint(np.max([1,sz[1] - w])) , np.random.randint(np.max([1,sz[0] - h]))]
    img1 = misc.imresize(img0, (h,w))
    img = np.zeros_like(img0)
    # print img0.shape
    if len(img0.shape)==2:
        img[p[1] : p[1] + h, p[0]:p[0] + w] = img1
    else:
        img[p[1] : p[1] + h, p[0]:p[0] + w, :] = img1

    return (img,p,zoom_ratio)

class FasterRCNN_Data(caffe.Layer):       
    def setup(self,bottom,top):   
        self._name_to_top_map={'data':0,'label':1,'sampling_param':2}
        self.resize_width = cfgs.resize_width
        self.resize_height = cfgs.resize_height
        self.batch_size = cfgs.batch_size
        self.sliding_window_width = cfgs.sliding_window_width
        self.sliding_window_height = cfgs.sliding_window_height
        self.sliding_window_stride = cfgs.sliding_window_stride
        self.iou_positive_thres = cfgs.iou_positive_thres
        self.iou_negative_thres = cfgs.iou_negative_thres 
        self.phase = eval(self.param_str)['phase']
        self.dat_dir = eval(self.param_str)['dat_dir']     
        self.iter = 0
        self.py_fn = __file__.split('/')[-1]
        self.py_dir = os.path.dirname(__file__)

        print "setup"
        self.data_queue=rawData()
        
    def reshape(self,bottom,top):
        top[0].reshape(1, 3, self.resize_height, self.resize_width)
        feature_map_height = self.resize_height / self.sliding_window_stride
        feature_map_width = self.resize_width / self.sliding_window_stride
        top[1].reshape(1, 5*len(self.sliding_window_width), feature_map_height, feature_map_width)
        top[2].reshape(self.batch_size, 7)

        top[3].reshape(1, 4)

    def get_next_image(self):
        while True:
            blobs=self.data_queue.nextBatch()
            img=blobs['data'][0,0]
            bbs,_=np.hsplit(blobs['gt_boxes'],[-1])
            # print bbs.shape
            if len(bbs)==0:
                continue

            bbs[:,2]-=bbs[:,0]
            bbs[:,3]-=bbs[:,1]

            return img, bbs
        

        # dat_index = -1
        # img_index = -1
        # while True:
        #     if self.phase == 'TRAIN': 
        #         candidate_index = range(300) + range(400,582)               
        #         dat_index = candidate_index[np.random.randint(len(candidate_index))]
        #         img_index = np.random.randint(512)
        #     else:
        #         dat_index = np.random.randint(300,400)
        #         img_index = np.random.randint(512)   
                             
        #     img_fn = '%s/dat-%06d/img-%06d.jpg' % (self.dat_dir, dat_index, img_index)
        #     tag_fn = '%s/dat-%06d/img-%06d.h5' % (self.dat_dir, dat_index, img_index)
        #     if not( os.path.exists(img_fn) and os.path.exists(tag_fn)):
        #         continue
        #     with h5py.File(tag_fn,'r') as h5f:
        #         bbs = np.float32(h5f['upper_label'][:])
        #         if bbs.shape[0]==0:
        #             continue
        #     if os.path.exists(img_fn) and os.path.exists(tag_fn):
        #         break
        # return (img_fn, tag_fn)

            
    def forward(self,bottom,top):
        #load image
        # (img_fn, tag_fn) = self.get_next_image()
        (img, bbs) = self.get_next_image()
        
        #print img_fn
        # img = misc.imread(img_fn)
        (img,pos,zoom_ratio) = random_zoomout(img)
        
        img_height = np.shape(img)[0]
        img_width = np.shape(img)[1] 
        img = misc.imresize(img,(self.resize_height, self.resize_width))
        minv = np.min(img)
        maxv = np.max(img)
        if minv == maxv:
            norm_img = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.float32)
        else:
            norm_img = (np.float32(img) - minv) / (maxv - minv) - 0.5
        
        if len(norm_img.shape)==2:
            top[0].data[0,0,:,:]=norm_img
        else:
            top[0].data[0,:,:,:]=np.transpose(norm_img, (2,0,1))
        # 0 xmin 1 ymin 2 w 3 h 

        #load tag
        # with h5py.File(tag_fn,'r') as h5f:
        #     bbs = np.float32(h5f['upper_label'][:])
        #     bbs[:,0] = bbs[:,0]*zoom_ratio + pos[0]
        #     bbs[:,1] = bbs[:,1]*zoom_ratio + pos[1]
        #     bbs[:,2] = bbs[:,2]*zoom_ratio
        #     bbs[:,3] = bbs[:,3]*zoom_ratio
                    
        bbs[:,0] = bbs[:,0]*zoom_ratio + pos[0]
        bbs[:,1] = bbs[:,1]*zoom_ratio + pos[1]
        bbs[:,2] = bbs[:,2]*zoom_ratio
        bbs[:,3] = bbs[:,3]*zoom_ratio        

        bbs[:,0] = bbs[:,0]*self.resize_width/img_width
        bbs[:,2] = bbs[:,2]*self.resize_width/img_width
        bbs[:,1] = bbs[:,1]*self.resize_height/img_height
        bbs[:,3] = bbs[:,3]*self.resize_height/img_height
        
        #compute all ious  
        feature_map_height = self.resize_height / self.sliding_window_stride
        feature_map_width = self.resize_width / self.sliding_window_stride
        size_num = len(self.sliding_window_height)
                      
        anchor_bbs = np.zeros((size_num*feature_map_height*feature_map_width,4),dtype = np.float64)
        for size_index in range(size_num):
            h=self.sliding_window_height[size_index]
            w=self.sliding_window_width[size_index]
            xs = np.arange(feature_map_width) * self.sliding_window_stride + self.sliding_window_stride/2-1 - w/2
            for y_index in range(feature_map_height):                  
                y=y_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - h/2
                ind = size_index*feature_map_height*feature_map_width + y_index*feature_map_width
                anchor_bbs[ind : ind + feature_map_width,0] = xs
                anchor_bbs[ind : ind + feature_map_width,2] = xs + w
                anchor_bbs[ind : ind + feature_map_width,1] = y
                anchor_bbs[ind : ind + feature_map_width,3] = y + h
        
        bbs2 =  np.zeros((len(bbs),4), dtype = np.float64)
        bbs2[:,0:2] = bbs[:,0:2]
        bbs2[:,2:4] = bbs[:,0:2] + bbs[:,2:4]    
        iou = cython_bbox.bbox_overlaps(anchor_bbs,bbs2)                
        
  
        #anchor box and gt box assignment
        pos_anchor=list()
        anchor_fired_bbs = list()        
        neg_anchor=list()
        bbs_fire_list = np.zeros(len(bbs),dtype=np.int8)
        for size_index in range(size_num):
            h=self.sliding_window_height[size_index]
            w=self.sliding_window_width[size_index] 
            for y_index in range(feature_map_height):
                y=y_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - h/2
                for x_index in range(feature_map_width):
                    x=x_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - w/2
                    anchor_box = [x,y,w,h, x_index, y_index, size_index]                      
                    anchor_index = size_index*feature_map_height*feature_map_width + y_index*feature_map_width + x_index
                    fired_bb = np.where(iou[anchor_index,:] > self.iou_positive_thres)[0]
                    max_iou = np.max(iou[anchor_index,:])                                                
                    if max_iou < self.iou_negative_thres:
                        neg_anchor.append(anchor_box)
                    elif max_iou > self.iou_positive_thres:
                        pos_anchor.append(anchor_box)
                        bb_ind = int(fired_bb[np.random.randint(len(fired_bb))])
                        anchor_fired_bbs.append(bb_ind)
                        bbs_fire_list[bb_ind] = 1
                                
        
        for j in range(len(bbs)):
            if bbs_fire_list[j] > 0:
                continue #this gt bb has been assigned an anchor box
#             print 'bbs[%d] is un-assigned' % j
            max_iou_anchor_ind = np.argmax(iou[:,j])
            size_index = max_iou_anchor_ind / (feature_map_height*feature_map_width)            
            y_index = (max_iou_anchor_ind % (feature_map_height*feature_map_width) ) / feature_map_width
            x_index = max_iou_anchor_ind % feature_map_width
            h=self.sliding_window_height[size_index]
            w=self.sliding_window_width[size_index]
            x=x_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - w/2
            y=y_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - h/2
            anchor_box = [x,y,w,h, x_index, y_index, size_index]
            pos_anchor.append(anchor_box)
            anchor_fired_bbs.append(j)

        pos_anchor = np.array(pos_anchor)
        anchor_fired_bbs = np.array(anchor_fired_bbs)
        neg_anchor = np.array(neg_anchor)
        
        
        #sampling from pos_anchor and neg_anchor
        sampling_param = np.zeros([self.batch_size, 7], dtype=np.float32)
        tags = np.zeros([1, 5*len(self.sliding_window_width),feature_map_height,feature_map_width],dtype=np.float32)
        
        rnd_perm = np.random.permutation(len(pos_anchor))        
        pos_anchor = pos_anchor[rnd_perm]
        anchor_fired_bbs = anchor_fired_bbs[rnd_perm]
        neg_anchor = np.random.permutation(neg_anchor)
         
        pos_num_in_batch = min([self.batch_size,len(pos_anchor)])
        
        for i in range(pos_num_in_batch):
            x = pos_anchor[i][0]
            y = pos_anchor[i][1]
            w = pos_anchor[i][2]
            h = pos_anchor[i][3]
            x_index = pos_anchor[i][4]
            y_index = pos_anchor[i][5]
            size_index = pos_anchor[i][6]
            tags[0,0+5*size_index,y_index,x_index]=1.0
            gt = bbs[anchor_fired_bbs[i]]
            tags[0,1+5*size_index,y_index,x_index]=(gt[0] + 0.5*gt[2] - x - 0.5*w) / w
            tags[0,2+5*size_index,y_index,x_index]=(gt[1] + 0.5*gt[3] - y - 0.5*h) / h
            tags[0,3+5*size_index,y_index,x_index]=np.log(np.float32(gt[2])/w)
            tags[0,4+5*size_index,y_index,x_index]=np.log(np.float32(gt[3])/h)
            sampling_param[i,:] = pos_anchor[i]
        
        
        if pos_num_in_batch < self.batch_size:
            neg_anchor_num = len(neg_anchor)
            for i in range(pos_num_in_batch,self.batch_size):
                sampling_param[i,:] = neg_anchor[(i - pos_num_in_batch) % neg_anchor_num]
        
        if np.random.randint(50)==0:
            print '[%s] pos_anchor: %d, neg_anchor:%d' % (self.py_fn, len(pos_anchor), len(neg_anchor))
        top[1].data[...]=tags    
        top[2].data[...]=sampling_param   
        
        top[3].data[...]=bbs
        self.iter += 1  

    def backward(self,top,propagate_Down,bottom):
        pass

