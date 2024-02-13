import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet_extractor(nn.Module):
    def __init__(self):
        super(Resnet_extractor, self).__init__()
        resent_model = models.resnet18()

        self.conv1 = resent_model.conv1
        self.bn1 = resent_model.bn1
        self.relu = resent_model.relu
        self.maxpool = resent_model.maxpool
        self.layer1 = resent_model.layer1
        self.layer2 = resent_model.layer2
        self.layer3 = resent_model.layer3
        self.layer4 = resent_model.layer4
        self.avg_pool = resent_model.avgpool
        
    def forward(self, t):
        t= self.conv1(t)
        t= self.bn1(t)
        t= self.relu(t)
        t= self.maxpool(t)
        t= self.layer1(t)
        t= self.layer2(t)
        t= self.layer3(t)
        t= self.layer4(t)
        without_polling = t
        t = self.avg_pool(t)
        t = t.view(t.size(0), -1)
        return t , without_polling





class Obj_classifier(nn.Module):
    def __init__(self, input_channel=1024, output_class=47, drop_out=0.5):
        super(Obj_classifier, self).__init__()
        self.fc1 = nn.Sequential(  # it is just a way to avoid defining the forward function for each layer
            nn.Linear(input_channel, 1000),
            nn.BatchNorm1d(1000, affine=True),     # what is affine????
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out)
        )
        self.fc2 = nn.Linear(1000, output_class)
        
    def forward(self, t):
        t = self.fc1(t)
        t = self.fc2(t)
        return t





class Rot_classifier(nn.Module):
    def __init__(self, input_channels=1024, output_class=4, projection_dim = 100):
        super(Rot_classifier, self).__init__()
        self.projection_dim = projection_dim
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(input_channels, projection_dim, [1,1], stride=[1,1]),   # 1x1 as fileter is [1,1]
            nn.BatchNorm2d(projection_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(projection_dim, projection_dim, [3,3], stride=[2,2]),
            nn.BatchNorm2d(projection_dim),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(projection_dim*3*3, projection_dim),    # as output of conv_3x3 has hight and width of 3
            nn.BatchNorm1d(projection_dim, affine=True),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Linear(projection_dim, output_class)


    def forward(self, t):
        t = self.conv_1x1(t)
        t = self.conv_3x3(t)
        t = t.view(t.size(0), -1)
        t = self.fc1(t)
        t = self.fc2(t)
        return t

        
class Rot_regressor(nn.Module):
    def __init__(self, input_channels=1024, output_len=1, drop_out=0.5, projection_dim = 100):
        super(Rot_regressor, self).__init__()
        self.p_cos = Regressor_sub()
        self.p_sin = Regressor_sub()

    def forward(self, t):
        cos_out = self.p_cos(t)
        sin_out = self.p_sin(t)
        return cos_out , sin_out


class Regressor_sub(nn.Module):
    def __init__(self, input_channels=1024, output_len=1, drop_out=0.5, projection_dim = 100):
        super(Regressor_sub, self).__init__()                
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(input_channels, projection_dim, [1,1], stride=[1,1]),   # 1x1 as fileter is [1,1]
            nn.BatchNorm2d(projection_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(projection_dim, projection_dim, [3,3], stride=[2,2]),
            nn.BatchNorm2d(projection_dim),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(projection_dim*3*3, projection_dim),    # as output of conv_3x3 has hight and width of 3
            nn.BatchNorm1d(projection_dim, affine=True)
        )
        self.drop_out = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(projection_dim, output_len)

    def forward(self, t):
        t = self.conv_1x1(t)
        t = self.conv_3x3(t)
        t = t.view(t.size(0), -1)
        t = self.fc1(t)
        t = self.drop_out(t)
        t = self.fc2(t)
        return t