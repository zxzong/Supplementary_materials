#自定义loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperDeepSEA(nn.Module):
    """
    A deeper DeepSEA model architecture.
    Parameters
    ----------
    sequence_length : int
        The length of the sequences on which the model trains and and makes
        predictions.
    n_targets : int
        The number of targets (classes) to predict.
    Attributes
    ----------
    conv_net : torch.nn.Sequential
        The convolutional neural network component of the model.
    classifier : torch.nn.Sequential
        The linear classifier and sigmoid transformation components of the
        model.
    """

    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2) 
            
            )

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._n_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_targets),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())


    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

class My_loss(nn.Module):
    def __init__(self,pos_weight=1):
        super().__init__()
        self.pos_weight = pos_weight
        #self.record_y = []

        
    def forward(self, x, y):
        loss = - self.pos_weight * y * torch.log(x+1e-5) - (1 - y) * torch.log((1 - x)+1e-5) 
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_sum = torch.tensor(np.array(y.tolist()).sum(axis =1)).to(device).float()
        batch_tar =  ((np.array(y.tolist()).sum(axis =1))/12)**2
        batch_tar = torch.tensor(batch_tar).to(device).float()

        weight__ = torch.sqrt(torch.tensor(len(y)*[1.]).to(device)+batch_sum*torch.pow((torch.tensor([1.]*len(y)).to(device)-torch.min(torch.tensor(len(y)*[1.]).to(device),batch_tar)),2))
        
        '''
        for j in y:
                self.record_y.append(torch.sum(j))

        print(self.record_y)
       '''
         
        loss  = weight__*torch.mean(loss,dim=1,keepdim=False)
        return loss.mean()


def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.
    Returns
    -------
    torch.nn._Loss
    """
    return My_loss()

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.
    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.
    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
