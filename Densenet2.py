import torch
import torch.nn as nn
from Densenet import _DenseBlock, _DenseLayer, _Transition

class Densenet2(nn.Module):
    def __init__(self, block_config, view, growth_rate=32, num_init_features=64, bn_size=4, num_classes=13):
 
        super(Densenet2, self).__init__()
        
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
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x1,x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv11(x2)
        x2 = self.bn11(x2)
        x2 = self.relu(x2)
        
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)
        
        x1 = self.block1(x1)
        x2 = self.block11(x2)        
        x1 = self.trans1(x1)
        x2 = self.trans11(x2)
        
        x1 = self.block2(x1)
        x2 = self.block21(x2)        
        x1 = self.trans2(x1)
        x2 = self.trans21(x2)
        
        x1 = self.block3(x1)
        x2 = self.block31(x2)        
        x1 = self.trans3(x1)
        x2 = self.trans31(x2)
        
        x = x1 + x2
        x = torch.div(x,2.0)
        
        x = self.block41(x)
        x = self.bn21(x)        
        x = self.avgpool(x)        
        x = x.view(x1.size(0), -1)
        x = self.fc(x)
        
        return x