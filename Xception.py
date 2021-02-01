import torch.nn as nn
import torch.nn.functional as F

class Xception(nn.Module):

    def __init__(self, view=4, num_classes=13):
        
        super(Xception, self).__init__()
        self.num_classes = num_classes
        c = 1
        if view == 4:
            c = 2
        model = [nn.Conv2d(c, 32, 3, 2, 0, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32,64,3,bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]
        
        model += [Block(64,128,2,2,start_with_relu=False, grow_first=True)]
        model += [Block(128,256,2,2,start_with_relu=True, grow_first=True)]
        model += [Block(256,728,2,2,start_with_relu=True, grow_first=True)]
        
        for i in range(8):
            model += [Block(728,728,3,1,start_with_relu=True,grow_first=True)]
            
        model += [Block(728,1024,2,2,start_with_relu=True,grow_first=False)]
        
        model += [SeparableConv2d(1024,1536,3,1,1)]
        model += [nn.BatchNorm2d(1536)]
        model += [nn.ReLU(inplace=True)]
        
        model += [SeparableConv2d(1536,2048,3,1,1)]
        model += [nn.BatchNorm2d(2048)]
        model += [nn.ReLU(inplace=True)]
        
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self,input):
        x = self.model(input)
        x = self.logits(x)
        return x

    def logits(self, input):
        x = F.adaptive_avg_pool2d(input, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()
        
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
            
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x