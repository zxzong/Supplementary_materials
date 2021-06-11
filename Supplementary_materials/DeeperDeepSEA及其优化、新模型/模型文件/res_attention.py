
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#res模块
class Bottleneck(nn.Module):
    expansion = 1 #
 
    def __init__(self, inplanes, planes, stride=1, downsample=None,use_1x1conv=False):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm1d(planes)
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn_2 = nn.BatchNorm1d(planes)
        self.conv_3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if use_1x1conv:
            self.conv_4 = nn.Conv1d(inplanes, planes,kernel_size = 1, stride=stride)
        else:
            self.conv_4 = False
        self.bn_res = nn.BatchNorm1d(planes)

 
    def forward(self, x):
        if self.conv_4:
            residual = self.conv_4(x)
            residual = self.bn_res(residual)
        else:
            residual = x
            residual = self.bn_res(residual)
 
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
 
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)
 
        out = self.conv_3(out)
        out = self.bn_3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
 
        self.fc1   = nn.Conv1d(in_channel, in_channel // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_channel // 16, in_channel, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
 
#空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
 
        self.conv1 = nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DeeperDeepSEA(nn.Module):
 

    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        
        self.conv_h1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size= 7),
            nn.ReLU(inplace=True)
            )
        self.conv_h2 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9,padding=1),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h3 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=11,padding=2),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h4 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=13,padding=3),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h5 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=15,padding=4),
            nn.ReLU(inplace=True)
            )
        
        #接收n个通道的输入
        self.ca = ChannelAttention(160)
        self.sa = SpatialAttention()


        self.conv =  nn.Sequential(
            Bottleneck(160,160),
            Bottleneck(160,320,use_1x1conv=True),

            #降低通道数
            nn.Conv1d( 320,160, kernel_size=1),
            
            nn.Conv1d( 160,24, kernel_size=1),
            nn.ReLU(inplace=True),
            
            #
            nn.Conv1d(24,24,498),
            nn.ReLU(inplace=True),
            nn.Conv1d(24,24,497),

        )

        self.classifier = nn.Sequential(
            nn.Sigmoid()
            )
            


    def forward(self, x):
       

        #直接reshape,全局不用全连接层
        out1 = self.conv_h1(x)
        out2 = self.conv_h2(x)
        out3 = self.conv_h3(x)
        out4 = self.conv_h4(x)
        out5 = self.conv_h5(x)
        out_merge = torch.cat((out1,out2,out3,out4,out5),dim=1)

        out_merge_ca = self.ca(out_merge) * out_merge
        out_merge_sa = self.sa(out_merge_ca) * out_merge_ca
        
        out_ = self.conv(out_merge_sa)

        reshape_out = out_.view(out_.size(0), 24 )
        predict = self.classifier(reshape_out)

        return predict
        
def criterion():

    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.SGD,{"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
