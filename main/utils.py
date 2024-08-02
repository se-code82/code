import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import copy
from tkinter import *
from PIL import ImageTk
from PIL import Image

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

def Miou_score(pred, gt, num_classes):

    smooth = 1e-5
    iou_list = []

    for class_id in range(1, num_classes):
        pred_class = pred == class_id
        gt_class = gt == class_id
        intersection = (pred_class & gt_class).sum()
        union = (pred_class | gt_class).sum()
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / len(iou_list)
    return mean_iou


def metrics_all(pred, gt):
    smooth = 1e-5
    tp = pred & gt
    fn = ((pred == 0) & (gt == 1)).astype('int')
    fp = ((pred == 1) & (gt == 0)).astype('int')
    tn = ((pred == 0) & (gt == 0)).astype('int')

    precision = (tp.sum()+smooth) / (tp.sum() + fp.sum()+smooth)
    Acc = (tp.sum()+tn.sum()+smooth) / (tp.sum()+fn.sum()+fp.sum()+tn.sum()+smooth)
    recall = (tp.sum()+ smooth)/ (tp.sum() + fn.sum()+smooth)
    f1 = 2 * precision * recall / (precision + recall)

    return float(f1), float(precision), float(recall), float(Acc)


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _, x, y = image.shape


    if x != patch_size[0] or y != patch_size[1]:

        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)

    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()

    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=3)
        else:
            prediction = out

    metric_list = []
    metric_list1 = []
    metric_list2 = []


    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))


    for i in range(1, classes):
        metric_list1.append(Miou_score(prediction == i, label == i, classes))
        metric_list2.append(metrics_all(prediction == i, label == i))



    if test_save_path is not None:
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)

        a1[a1 == 1] = 255
        a1[a1 == 2] = 207
        a1[a1 == 3] = 255
        a1[a1 == 4] = 20
        #
        a2[a2 == 1] = 255
        a2[a2 == 2] = 53
        a2[a2 == 3] = 0
        a2[a2 == 4] = 10
        #
        a3[a3 == 1] = 255
        a3[a3 == 2] = 46
        a3[a3 == 3] = 0
        a3[a3 == 4] = 120


        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path+'/'+case+'.png')

    return metric_list, metric_list1, metric_list2

def get_num_parameters(net):
    encoder_p = sum([p.numel() for p in net.encoder.parameters()]) / 10**6
    aspp_p = sum([p.numel() for p in net.aspp.parameters()]) / 10**6
    decoder_p = sum([p.numel() for p in net.decoder.parameters()]) / 10**6
    return encoder_p, aspp_p, decoder_p
