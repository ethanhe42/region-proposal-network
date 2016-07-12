import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_path='/mnt/data1/yihuihe/mnc/'

class rawData:
    def __init__(self):
        self.imdb=np.load(data_path+'data.npy')
        self.maskdb=np.load(data_path+'maskdb.npy')
        self.roidb=np.load(data_path+'roidb.npy')        
        self.mask=np.load(data_path+'mask.npy')
        self.total=self.imdb.shape[0]


    def nextBatch(self, d=False):
        while True:
            idx=np.random.randint(self.total)
            if len(self.roidb[idx])!=0:
                break
        
        data=self.imdb[idx][np.newaxis,:]
        gt_boxes=np.array(self.roidb[idx])
        
        maskdb=self.maskdb[idx]
        mask_max_x=0
        mask_max_y=0
        for ins in maskdb:
            if ins.shape[0]>mask_max_y:
                mask_max_y=ins.shape[0]
            if ins.shape[1]>mask_max_x:
                mask_max_x=ins.shape[1]

        gt_masks=np.zeros((len(maskdb),mask_max_y,mask_max_x))
        mask_info=np.zeros((len(maskdb),2))
        for j in range(len(maskdb)):
            mask=maskdb[j]
            mask_x=mask.shape[1]
            mask_y=mask.shape[0]
            gt_masks[j,0:mask_y,0:mask_x]=mask
            mask_info[j,0]=mask_y
            mask_info[j,1]=mask_x

        blobs={
            'data': data,
            'gt_boxes': gt_boxes,
            'im_info': np.array([[data.shape[2],data.shape[3],1]], dtype=np.float32),
            'gt_masks':gt_masks,
            'mask_info':mask_info
        }
        if d: 
            # i is always 1
            for i in range(blobs['data'].shape[0]):
                print blobs['im_info']
                print blobs['mask_info']
                print blobs['gt_boxes']
                img=blobs['data'][0,0]
                print img.shape
                fig=plt.figure()
                ax=fig.add_subplot(111)
                plt.imshow(img)
                for j,bbox in enumerate(gt_boxes):
                    blank=np.zeros_like(img)
                    print blank.shape,maskdb[j].shape,bbox
                    blank[bbox[1]:maskdb[j].shape[0]+bbox[1],bbox[0]:maskdb[j].shape[1]+bbox[0]]=maskdb[j]
                    blank[blank>0]=1
                    plt.imshow(blank,alpha=.9)
                    ax.add_patch(patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False))
                    plt.text(bbox[0],bbox[1],bbox[-1],bbox=dict(facecolor='blue',alpha=0.5),fontsize=14, color='white')
                plt.show()
            for i in blobs:
                print i,blobs[i].shape
            print ''
        return blobs

if __name__=='__main__':
    raw=rawData()
    while True:
        raw.nextBatch(True)
