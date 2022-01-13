import torch
import torch.nn as nn

from utils import IOU

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss(reduction = "sum")
        self.BCE = nn.BCEWithLogitsLoss()
        self.Entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid
        self.lambda_noobj = 10
        self.lambda_class = 1
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self,prediction,target, anchors):
        exist_obj = target[..., 0] == 1
        not_exists_obj = target[...,0] == 0

        #No obj loss
        no_obj_loss = self.BCE(prediction[...,0:1], (target[...,0:1][not_exists_obj]))

        #Obj loss
        anchors = anchors.reshape(1,3,1,1,2) #3x2 reshape for p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(prediction[...,1:3]),torch.exp(prediction[...,3:5]) * anchors],dim = -1)
        ious = IOU(box_preds[exist_obj], target[...,1:5][exist_obj])

        #Coords loss

        #Class loss