import torch
from Resnet import Bottleneck as block
from Projection import position_pair
import torch.nn as nn

class ResNet11(nn.Module):

    def __init__(self, layers, view, num_classes=13, img_size = 224):        
        super(ResNet11, self).__init__()
        self.view = view
        self.num_classes = num_classes     
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.inplanes = 64
        self.conv11 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn11 = nn.BatchNorm2d(64)
        self.layer11 = self._make_layer(block, 64, layers[0])
        self.layer21 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer31 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer41 = self._make_layer(block, 512, layers[3], stride=2)
                      
        self.inplanes = 64
        self.layer13 = self._make_layer(block, 64, layers[0])
        self.layer23 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer33 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer43 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.position1 = position_pair(64,112)
        self.position2 = position_pair(256,56)
        self.position3 = position_pair(512,28)
        self.position4 = position_pair(1024,14)
        self.position5 = position_pair(2048,7)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxfusion = nn.MaxPool2d(kernel_size=(3,1), stride=1)
        
        if self.view == 6:
            self.fc1 = nn.Linear(2048, self.num_classes)
            self.fc2 = nn.Linear(2048, self.num_classes)
        self.fc3 = nn.Linear(2048, self.num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input1,input2):    
        x1 = self.conv1(input1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)                
        x2 = self.conv11(input2)
        x2 = self.bn11(x2)
        x2 = self.relu(x2)
        
        x31 = self.position1(x1,x2)
        x1 = self.maxpool(x1)        
        x2 = self.maxpool(x2)
        x3 = self.maxpool(x31)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.layer1,self.layer11,self.layer13,self.position2)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.layer2,self.layer21,self.layer23,self.position3)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.layer3,self.layer31,self.layer33,self.position4)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.layer4,self.layer41,self.layer43,self.position5)
        
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)               
        x3 = self.avgpool(x3)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1) 
        x3 = x3.view(x3.size(0), -1)

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
        
    def fusion(self, input1,input2,input3,layer1,layer2,layer3,layer4=None, times=1):
        x1 = layer1(input1)
        x2 = layer2(input2)
        x3 = layer3(input3)
        x33 = layer4(x1,x2)
        x3 = x3 + x33
        return x1,x2,x3