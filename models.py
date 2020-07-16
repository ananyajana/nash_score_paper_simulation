import torch
from torch import nn
from torchvision import models as torch_models
import torch.nn.functional as F

class Inception_v3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.incnet = torch_models.inception_v3(pretrained=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

        for param in self.incnet.parameters():
            param.requires_grad = True

    def forward(self, x, is_inception = False):
        #print('in forward, x shape before: ', x.shape)
        x = self.incnet.Conv2d_1a_3x3(x)
        #print('in forward, x shape after: ', x.shape)
        x = self.incnet.Conv2d_2a_3x3(x)
        x = self.incnet.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        
        x = self.incnet.Conv2d_3b_1x1(x)
        x = self.incnet.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        x = self.incnet.Mixed_5b(x)
        x = self.incnet.Mixed_5c(x)
        x = self.incnet.Mixed_5d(x)

        x = self.incnet.Mixed_6a(x)
        x = self.incnet.Mixed_6b(x)
        x = self.incnet.Mixed_6c(x)
        x = self.incnet.Mixed_6d(x)
        x = self.incnet.Mixed_6e(x)
        
        if is_inception:
            #print('in forward, x shape before: ', x.shape)
            aux = self.incnet.AuxLogits(x)
            #print('in forward, x shape after: ', x.shape)
        else:
            aux = None

        x = self.incnet.Mixed_7a(x)
        x = self.incnet.Mixed_7b(x)
        x = self.incnet.Mixed_7c(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x, aux

