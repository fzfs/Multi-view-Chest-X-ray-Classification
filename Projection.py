import torch
from torch import nn
import torch.nn.functional as F
    
class position_pair(nn.Module):
    def __init__(self,channel=64, spatial=112):
        super(position_pair, self).__init__()
        self.conv_1 = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(1)        
        self.conv_2 = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(1)        
        self.conv_3 = nn.Conv2d(spatial, channel, kernel_size=3, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, input1,input2):
        b, c, h, w = input1.size()
        x1 = self.conv_1(input1).view(b, 1, h, w)
        x1 = self.bn_1(x1)
        x1 = self.relu(x1)
        x1 = x1.expand(b,h,h,w)
        
        x2 = self.conv_2(input2).view(b, 1, h, w)
        x2 = self.bn_2(x2)
        x2 = self.relu(x2)
        x2 = x2.expand(b,h,h,w)
        x2 = x2.permute(0,3,2,1)            
        x3 = x1+x2
        x3 = torch.div(x3, 2.0)
        x3 = self.conv_3(x3)
        x3 = self.bn_3(x3)
        x3 = self.relu(x3)

        return x3