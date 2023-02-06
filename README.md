# IR_small_target_detection
IR small target detection in cluttered background algorithms implementations based on actual researches

Link to used datasets:
https://www.kaggle.com/datasets/llpukojluct/ir-small-target?select=dataset_IR_1

IR_detection.py contains algorithms with feature map extraction, segmentation and classification
If name of function contain "clf" - it combined with segmentation and classification, else it is only feature extraction

utils.py contains algorithms for collecting meta data from dataset folder and testing detection methodes to get metrics

Example:

import utils as u
# load dataset and create meta DataFrame
dsfolder = "your_dataset_folder"
loader = u.MetaLoader(dsfolder)
data = loader.train_val_split(0.1)[0]

ev = u.Evaluate(data) # create Evaluate object


get_filtering_metrics - requires only feature extraction algorithm such as LCM or top_hat
get_detection_metrics and get_ROC - requires complete algorithm such as LCM_clf or top_hat_clf
