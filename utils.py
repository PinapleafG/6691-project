import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
import random
import json
import logging
import clip


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    

def make_desc_features(desc_path, clip_model, device):
    desc_feat = {}
    with open(desc_path, 'r') as f:
        desc_dict = json.load(f)
    class_idx = 0
    with torch.no_grad():
        for k, v in desc_dict.items():
            description = v["features"]
            token = clip.tokenize(description).to(device)
            feat = clip_model.encode_text(token).cpu().numpy()
            desc_feat[k] = {}
            desc_feat[k]["features"] = feat
            desc_feat[k]["class"] = class_idx
            class_idx += 1

    return desc_feat

def setup_seed(seed):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def accuracy(output, gt):
    output = output.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    
    ## calculate Dice-Similarity coefficie and HD95
    dice = []
    hd95 = []
    for i in range(output.shape[0]):
        d, h = calculate_metric_percase(output[i], gt[i])
        dice.append(d)
        hd95.append(h)

    ## calculate miou
    iou = []
    for i in range(output.shape[0]):
        intersection = np.logical_and(output[i], gt[i])
        union = np.logical_or(output[i], gt[i])
        iou.append((np.sum(intersection)+1e-8)/ (np.sum(union)+1e-8))
    
    ## calculate F1 score
    f1_score = []
    for i in range(output.shape[0]):
        precision = metric.binary.precision(output[i], gt[i])
        recall = metric.binary.recall(output[i], gt[i])
        f1 = (2*(precision*recall)+1e-8) / (precision+recall+1e-8)
        f1_score.append(f1)

    return dice, hd95, iou, f1_score
    # return iou, f1_score
