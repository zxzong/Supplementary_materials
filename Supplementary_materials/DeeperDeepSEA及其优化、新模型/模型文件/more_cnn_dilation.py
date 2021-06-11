import numpy as np
import torch
import torch.nn as nn
class DeeperDeepSEA(nn.Module):
 

    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        
        self.conv_h1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size= 7),
            nn.ReLU(inplace=True)
            )
        self.conv_h2 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9,padding=1),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h3 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11,padding=2),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h4 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=13,padding=3),
            nn.ReLU(inplace=True)
            )
        
        self.conv_h5 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15,padding=4),
            nn.ReLU(inplace=True)
            )
        

        self.conv =  nn.Sequential(
            nn.Conv1d(320, 320, kernel_size=7,stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(320),
            nn.Conv1d(320, 480, kernel_size=1),
            nn.Conv1d(480, 480, kernel_size=3,dilation=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=3,dilation=4,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=3,dilation=8,padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=7,stride =4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

           
            nn.Conv1d(480, 960, kernel_size=1),
            nn.Conv1d(960, 960, kernel_size=3,dilation=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=3,dilation=4,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=3,dilation=8,padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=3,dilation=16,padding=15),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=3,dilation=32,padding=31),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=3,dilation=64,padding=63),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
           
            #降低通道数
            nn.Conv1d( 960,480, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d( 480,120, kernel_size=1),
            
            nn.Conv1d( 120,24, kernel_size=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(24,24,47)
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
        out_ = self.conv(out_merge)
        reshape_out = out_.view(out_.size(0), 24 )
        predict = self.classifier(reshape_out)
        return predict
        
def criterion():

    return nn.BCELoss()

def get_optimizer(lr):

    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6,"momentum": 0.9})
