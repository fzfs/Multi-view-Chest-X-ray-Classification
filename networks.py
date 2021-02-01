import uitl
import Resnet
import Resnet1
import Resnet2
import Resnet11

import Xception
import Xception1
import Xception2
import Xception11

import Densenet
import Densenet1
import Densenet2
import Densenet11

def define_X(init_type, gpu_ids,view,network):
    if network == 1:
        net = ResNet50(view)
    elif network == 2:
        net = Xception20(view)
    elif network == 3:
        net = Densenet121(view)
    return uitl.init_net(net,init_type, gpu_ids)

#{1:Frontal, 2:Lateral, 3:DualNet, 4:Stacted, 7:HeMIS,
# 5:Our method without Auloss and mimicry loss, 
# 6:Our method with Auloss and mimicry loss}
    
def ResNet50(view):
    layers = [3, 4, 6, 3]
    if view == 1 or view == 2 or view == 4:
        model = Resnet.ResNet(Resnet.Bottleneck, layers, view=view)
    elif view == 3:
        model = Resnet1.ResNet1(layers, view=view)
    elif view == 7:
        model = Resnet2.ResNet2(layers, view=view)
    elif view == 5 or view == 6:
        model = Resnet11.ResNet11(layers, view=view)
    return model

def Xception20(view):
    if view == 1 or view == 2 or view == 4:
        model = Xception.Xception(view=view)
    elif view == 3:
        model = Xception1.Xception1(view=view)
    elif view == 7:
        model = Xception2.Xception2(view=view)  
    elif view == 5 or view == 6:
        model = Xception11.Xception11(view=view)
    return model

def Densenet121(view):
    block_config=[6, 12, 24, 16]
    if view == 1 or view == 2 or view == 4:
        model = Densenet.Densenet(block_config, view=view)
    elif view == 3:
        model = Densenet1.Densenet1(block_config, view=view)
    elif view == 7:
        model = Densenet2.Densenet2(block_config, view=view) 
    elif view == 5 or view == 6:
        model = Densenet11.Densenet11(block_config, view=view)
    return model

