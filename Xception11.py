import torch
from Projection import position_pair
import torch.nn as nn
from Xception import Block, SeparableConv2d

class Xception11(nn.Module):

    def __init__(self, view, num_classes=13):        
        super(Xception11, self).__init__()
        self.view = view
        self.num_classes = num_classes     
        
        begin1 = [nn.Conv2d(1, 32, 3, 2, 0, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32,64,3,bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]        
        self.begin1 = nn.Sequential(*begin1)
        
        Entry1 = []
        Entry1 += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        Entry1 += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        Entry1 += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]        
        self.Entry1 = nn.Sequential(*Entry1)
        
        middle1 = []
        for i in range(8):
            middle1 += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
            
        self.middle1 = nn.Sequential(*middle1)
        
        self.Exit1 = Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        
        end1 = []
        end1 += [SeparableConv2d(1024,1536,3,1,1)]
        end1 += [nn.BatchNorm2d(1536)]
        end1 += [nn.ReLU(inplace=True)]        
        end1 += [SeparableConv2d(1536,2048,3,1,1)]
        end1 += [nn.BatchNorm2d(2048)]
        end1 += [nn.ReLU(inplace=True)]
        self.end1 = nn.Sequential(*end1)
        
        begin2 = [nn.Conv2d(1, 32, 3, 2, 0, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32,64,3,bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]        
        self.begin2 = nn.Sequential(*begin2)
        
        Entry2 = []
        Entry2 += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        Entry2 += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        Entry2 += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]        
        self.Entry2 = nn.Sequential(*Entry2)
        
        middle2 = []
        for i in range(8):
            middle2 += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
            
        self.middle2 = nn.Sequential(*middle2)
        
        self.Exit2 = Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        
        end2 = []
        end2 += [SeparableConv2d(1024,1536,3,1,1)]
        end2 += [nn.BatchNorm2d(1536)]
        end2 += [nn.ReLU(inplace=True)]        
        end2 += [SeparableConv2d(1536,2048,3,1,1)]
        end2 += [nn.BatchNorm2d(2048)]
        end2 += [nn.ReLU(inplace=True)]
        self.end2 = nn.Sequential(*end2)
        
        Entry3 = []
        Entry3 += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        Entry3 += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        Entry3 += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]        
        self.Entry3 = nn.Sequential(*Entry3)
        
        middle3 = []
        for i in range(8):
            middle3 += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
            
        self.middle3 = nn.Sequential(*middle3)
        
        self.Exit3 = Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        
        end3 = []
        end3 += [SeparableConv2d(1024,1536,3,1,1)]
        end3 += [nn.BatchNorm2d(1536)]
        end3 += [nn.ReLU(inplace=True)]        
        end3 += [SeparableConv2d(1536,2048,3,1,1)]
        end3 += [nn.BatchNorm2d(2048)]
        end3 += [nn.ReLU(inplace=True)]
        self.end3 = nn.Sequential(*end3)
        
        
        self.position1 = position_pair(64,109)
        self.position2 = position_pair(728,14)
        self.position3 = position_pair(728,14)
        self.position4 = position_pair(1024,7)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxfusion = nn.MaxPool2d(kernel_size=(3,1), stride=1)
        
        
        if self.view == 6:
            self.fc1 = nn.Linear(2048, self.num_classes)
            self.fc2 = nn.Linear(2048, self.num_classes)
        self.fc3 = nn.Linear(2048, self.num_classes)

    def forward(self, input1,input2):
        
        x1 = self.begin1(input1)
        x2 = self.begin2(input2)
        
        x3 = self.position1(x1,x2)
        
        x1,x2,x3 = self.fusion(x1,x2,x3,self.Entry1,self.Entry2,self.Entry3,self.position2)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.middle1,self.middle2,self.middle3,self.position3)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.Exit1,self.Exit2,self.Exit3,self.position4)
        
        x1 = self.end1(x1)                
        x2 = self.end2(x2)
        x3 = self.end3(x3)
        
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)               
        x3 = self.avgpool(x3)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1) 
        x3 = x3.view(x3.size(0), -1)
#        
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        
        x = torch.cat((x1,x2,x3),1)
        x = self.maxfusion(x)
        
        x = x.view(x.size(0),-1)        
        x = self.fc3(x)
        
        if self.view == 5:
            return x
        else:           
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1) 
            x1 = self.fc1(x1)
            x2 = self.fc2(x2)            
            return x, x1, x2
        
    def fusion(self, input1,input2,input3,layer1,layer2,layer3,layer4):
        x1 = layer1(input1)
        x2 = layer2(input2)
        x3 = layer3(input3)
        x33 = layer4(x1,x2)
        x3 = x3 + x33
        return x1,x2,x3