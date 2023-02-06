import numpy as np
import cv2
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def iou(box1, box2, box_mode = "xy"):
    '''
    Compute IoU between box1 and box2
    box1, box2 ndarray with shape (num_boxes, 4)
    box_mode - box coordinates format:  xywh - (x, y, w, h), xy - (x1, y1, x2, y2)
    '''
    if box_mode == "xywh":
        box1_maxes = box1[..., :2] + box1[..., 2:] // 2
        box1_mins = box1[..., :2] - box1[..., 2:] // 2
        box2_maxes = box2[..., :2] + box2[..., 2:] // 2
        box2_mins = box2[..., :2] - box2[..., 2:] // 2
    if box_mode == "xy":
        box1_maxes = box1[..., 2:]
        box1_mins = box1[..., :2]
        box2_maxes = box2[..., 2:]
        box2_mins = box2[..., :2]
    intersect_mins = np.maximum(box1_mins, box2_mins)
    intersect_maxes = np.minimum(box1_maxes, box2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]
    # IoU shape (num_boxes)
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou
  
  
  def draw_bbox(image, labels):
    """
    Draw bounding boxes on image according to labels and display plot
    label format : ndarray with shape [num_boxes, box_coord]
    box coordinates format : (x1, y1, x2, y2)
    """
    for i in range(labels.shape[0]):
        x1 = labels[i][0]   # rectangle left up corner
        y1 = labels[i][1]
        x2 = labels[i][2]  # rectangle right down corner
        y2 = labels[i][3]
        image1=cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # draw rectangle 
    plt.figure()
    plt.imshow(image1)
    
    
    class MetaLoader:
    
    
    def __init__(self, folder):
        """
        Create meta pd file from dataset directory
        """
        img_list = []
        label_list = []
        images = pathlib.Path(folder).glob('*.png') #get list of png images
        labels = pathlib.Path(folder).glob('*.txt')
        for img, label in zip(images, labels):
            img_list.append(img)
            label_list.append(label)
        data = {"image_path" : img_list, "label_path" : label_list}
        self.meta = pd.DataFrame(data)
    
    
    def train_val_split(self, val_size = 0.1):
        """
        Split filenames in train/validation data
        """
        images = self.meta["image_path"].unique()
        train_images, val_images = train_test_split(images, test_size = val_size)
        trainset = self.meta[self.meta["image_path"].isin(train_images)].copy()
        valset = self.meta[self.meta["image_path"].isin(val_images)].copy()
        return trainset, valset
    
    
class Evaluate:
    
    
    def __init__(self, meta):
        """
        Collect files according to meta in images and boxes ndarrays
        """
        self.meta = meta.set_index("image_path") # meta data of dataset for testing
        self.image_list = meta.image_path.unique() # list of image pathes
        images = []
        boxes = []
        for img in self.image_list:
            image = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2GRAY)
            labels = open(str(self.meta.loc[img].values[0]), "r").readlines()
            label = [[float(i) for i in 
                     labels[j].split()] 
                     for j in range(len(labels))]
            label = np.array(label).astype('int32')
            images.append(image)
            boxes.append(label)
        self.images = np.array(images)
        self.boxes = np.array(boxes)
        
        
    def get_filtering_metrics(self, alg):
        '''
        alg - filtering algorithm
        '''
        BSF = []
        SNRg = []
        num_obj = self.boxes.shape[0]
        for i in range(num_obj):
            image = self.images[i,:,:]
            label = self.boxes[i, :].reshape(1,4)
            fmap = alg(image)
            if np.isnan(fmap).max() == True:
                print("NaN detected")
            background_bef = image.copy()
            background_aft = fmap.copy()
            I_obj_bef = []
            I_obj_aft = []
            for i in range(label.shape[0]):
                x1, y1, x2, y2 = label[i, :]
                # delete object from image
                background_bef[y1 : y2, x1 : x2] -= image[y1 : y2, x1 : x2]
                # delete object from feature map
                background_aft[y1 : y2, x1 : x2] -= fmap[y1 : y2, x1 : x2]
                # find object signal before and after filtering
                I_obj_bef.append(image[y1 : y2, x1 : x2])
                I_obj_aft.append(fmap[y1 : y2, x1 : x2])
                
            BSF.append(np.std(background_bef) / (np.std(background_aft) + 0.0001))
            I_obj_bef = np.array(I_obj_bef).max()
            I_obj_aft = np.array(I_obj_aft).max()
            SNR_bef = (I_obj_bef - background_bef.mean()) / (np.std(background_bef)
                                                             + 0.0001)
            SNR_aft = (I_obj_aft - background_aft.mean()) / (np.std(background_aft)
                                                             + 0.0001)
            SNRg.append(SNR_aft / SNR_bef)
        
        # draw image and filtering result for random image
        num = np.random.randint(0, high = len(self.image_list))
        img = cv2.cvtColor(cv2.imread(str(self.image_list[num])), 
                                cv2.COLOR_BGR2GRAY)
        plt.figure(figsize = (10, 5))
        plt.subplot(1,2,1)
        plt.imshow(img, cmap = "gray")
        plt.subplot(1,2,2)
        plt.imshow(alg(img), cmap = "gray")
        return {"BSF" : np.array(BSF).mean(), "SNRg" : np.array(SNRg).mean()}
    
    def get_detection_metrics(self, alg, clf_thresh, thresh):
        '''
        alg - complete algorithm
        clf_thresh - threshold for classification / segmentation
        thresh - IoU threshold
        '''
        num_obj = self.boxes.shape[0] # total number of objects on image
        TP = 0 # number of true positive detections
        FP = 0 # number of false positive detections
        FN = 0 # number of false negative detections
        for i in range(num_obj):
            detections = alg(self.images[i,:,:], clf_thresh)
            print(detections)
            if detections.size != 0:
                iou_gt = iou(detections, np.repeat(self.boxes[i, :].reshape(1,4),
                                                       detections.shape[0], axis = 0))
                
                if (iou_gt > thresh).sum() >= 1:
                    TP += 1
                FP += (iou_gt <= thresh).sum()
                if (iou_gt > thresh).sum() == 0:
                    FN += 1
            else:
                FN += 1
        if TP==FP==0:
            return {
                    "True Detection Rate" : TP / num_obj, 
                    "False Alarm Rate" : 1,
                    "Precision" : 0,
                    "Recall" : TP / (TP + FN),
                    "F-score" : 0
                    }

        return {
                "True Detection Rate" : TP / num_obj, 
                "False Alarm Rate" : FP / (FP + TP),
                "Precision" : TP / (TP + FP),
                "Recall" : TP / (TP + FN),
                "F-score" : TP / (TP + 0.5 * (FP + FN))
                }
    
        
    def get_ROC(self, alg, bin_thresh):
        """
        Compute points in ROC-curve
        alg - complete algorithm
        bin_thresh - tuple of thresh segmentation values
        """
        c = 0 # counter
        iters = len(bin_thresh)
        ROC = np.zeros((iters, 2))
        for i in range(iters):
            c += 1
            metr = self.get_detection_metrics(alg, bin_thresh[i], 0)
            ROC[i, 0] = metr["True Detection Rate"]
            ROC[i, 1] = metr["False Alarm Rate"]
            print("Processing...  ", (c / iters) * 100, "% done")
        return ROC
