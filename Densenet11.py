import torch
import torch.nn as nn
import torch.nn.functional as F
from Projection import position_pair
from Densenet import _DenseBlock, _DenseLayer, _Transition

class Densenet11(nn.Module):
    def __init__(self, block_config, view, growth_rate=32, num_init_features=64, bn_size=4, num_classes=13):
 
        super(Densenet11, self).__init__()
        self.view = view
        
        self.conv1 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = _DenseBlock(num_layers=block_config[0], num_input_features=num_init_features, bn_size=bn_size)
        num_features = num_init_features + block_config[0] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block2 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[1] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block3 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[2] * growth_rate
        self.trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block4 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[3] * growth_rate        
        self.bn2 = nn.BatchNorm2d(num_features)
        
        self.conv11 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn11 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        
        self.block11 = _DenseBlock(num_layers=block_config[0], num_input_features=num_init_features, bn_size=bn_size)
        num_features = num_init_features + block_config[0] * growth_rate
        self.trans11 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block21 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[1] * growth_rate
        self.trans21 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block31 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[2] * growth_rate
        self.trans31 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block41 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[3] * growth_rate        
        self.bn21 = nn.BatchNorm2d(num_features)
        
        self.block13 = _DenseBlock(num_layers=block_config[0], num_input_features=num_init_features, bn_size=bn_size)
        num_features = num_init_features + block_config[0] * growth_rate
        self.trans13 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block23 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[1] * growth_rate
        self.trans23 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block33 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[2] * growth_rate
        self.trans33 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        self.block43 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size)
        num_features += block_config[3] * growth_rate        
        self.bn23 = nn.BatchNorm2d(num_features)
        
        self.position1 = position_pair(64,112)
        self.position2 = position_pair(128,28)
        self.position3 = position_pair(256,14)
        self.position4 = position_pair(512,7)
        self.position5 = position_pair(1024,7)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxfusion = nn.MaxPool2d(kernel_size=(3,1), stride=1)
        
        if self.view == 6:
            self.fc1 = nn.Linear(num_features, num_classes)
            self.fc2 = nn.Linear(num_features, num_classes)
        self.fc3 = nn.Linear(num_features, num_classes)
        
    def forward(self, x1,x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv11(x2)
        x2 = self.bn11(x2)
        x2 = self.relu(x2) 
        
        x3 = self.position1(x1,x2)
        
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)         
        x3 = self.maxpool(x3)              
        
        x1,x2,x3 = self.fusion(x1,x2,x3,self.block1,self.trans1,self.block11,self.trans11,self.block13,self.trans13, self.position2)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.block2,self.trans2,self.block21,self.trans21,self.block23,self.trans23, self.position3)
        x1,x2,x3 = self.fusion(x1,x2,x3,self.block3,self.trans3,self.block31,self.trans31,self.block33,self.trans33, self.position4)        
        x1,x2,x3 = self.fusion(x1,x2,x3,self.block4,self.bn2,self.block41,self.bn21,self.block43,self.bn23, self.position5)
        
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

    def fusion(self, input1,input2,input3,layer1,layer11,layer2,layer22,layer3,layer33,layer4=None):
        x1 = layer1(input1)        
        x1 = layer11(x1)
        x2 = layer2(input2)
        x2 = layer22(x2)
        x3 = layer3(input3)
        x3 = layer33(x3)
        x33 = layer4(x1,x2)
        x3 = x3 + x33
        return x1,x2,x3