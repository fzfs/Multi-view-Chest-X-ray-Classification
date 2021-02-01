import torch
from Xception import Block, SeparableConv2d
import torch.nn as nn
import torch.nn.functional as F

class Xception2(nn.Module):

    def __init__(self, view, num_classes=13):
        
        super(Xception2, self).__init__()
        self.num_classes = num_classes
        
        model1 = [nn.Conv2d(1, 32, 3, 2, 0, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32,64,3,bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]
        
        model1 += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        model1 += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        model1 += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]
        
        for i in range(8):
            model1 += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
        self.model1 = nn.Sequential(*model1)
        
        model2 = [nn.Conv2d(1, 32, 3, 2, 0, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32,64,3,bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]
        
        model2 += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        model2 += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        model2 += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]
        
        for i in range(8):
            model2 += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
        self.model2 = nn.Sequential(*model2)
		
        
        model3 = [Block(728,1024,2,2,start_with_relu=True,grow_first=False)]
        
        model3 += [SeparableConv2d(1024,1536,3,1,1)]
        model3 += [nn.BatchNorm2d(1536)]
        model3 += [nn.ReLU(inplace=True)]
        
        model3 += [SeparableConv2d(1536,2048,3,1,1)]
        model3 += [nn.BatchNorm2d(2048)]
        model3 += [nn.ReLU(inplace=True)]
        
        self.model3 = nn.Sequential(*model3)
        
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self,input1,input2):
        x1 = self.model1(input1)
        x2 = self.model2(input2)
        x = x1 + x2
        x = torch.div(x,2.0)
        x = self.model3(x)
        x = self.logits(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
		
        return x

    def logits(self, input):
        x = F.adaptive_avg_pool2d(input, (1, 1))
        x = x.view(x.size(0), -1)
        return x