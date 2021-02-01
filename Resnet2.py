import torch
import torch.nn as nn
from Resnet import Bottleneck as block

class ResNet2(nn.Module):

    def __init__(self, layers, view, num_classes=13, img_size = 224):

        super(ResNet2, self).__init__()      
        self.view = view
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
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
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
            
        x1 = self.maxpool(x1) 
        x2 = self.maxpool(x2)  
        
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
                
        x2 = self.layer11(x2)
        x2 = self.layer21(x2)
        x2 = self.layer31(x2)
        
        x = x1 + x2
        x = torch.div(x,2.0)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)        
        
        return x
    