import cv2
import torch
import numpy as np
import argparse
from torch.nn import init
from sklearn import metrics
from MyDataset import MyDataset
from torchvision import transforms

def init_net(net, init_type, gpu_ids, init_gain=0.02):
    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])        
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def print_metrics(labels_all, predict_all,val_loss,mode,name,view=0, class_name=''):
    aucs = []
    message = []
    for i in range(len(class_name)):
        auc = metrics.roc_auc_score(labels_all[:,i],predict_all[:,i])   
        aucs.append(auc)
        temp = class_name[i] + ':'+str(auc)
        print(temp)
        message.append(temp)
    mean = 'Mean:' + str(np.mean(np.array(aucs)))
    message.append(mean)
    print(mean)
    if mode == 'test' :
        file_name= name+'/result1.txt'
        with open(file_name, 'a+') as opt_file:
            opt_file.write('\n'.join(message)+'\n')
    
    return np.mean(np.array(aucs))

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--view', type=int, default= -1)    
    parser.add_argument('--network', type=int, default= -1)
    parser.add_argument('--epochs', type=int, default= 10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--lr', type=str, default='1e-5')
    parser.add_argument('--weight_decay', type=str, default='1e-3')
    parser.add_argument('--alpha', type=str, default= '1.0')
    parser.add_argument('--tm', type=str, default='337843')
    parser.add_argument('--class_weight', type=str, default='10234, 9991 ,1811 ,4612 ,1406 ,1944 , 2801, 13739, 43928, 11896, 868, 5485, 2043')
    parser.add_argument('--class_name', type=str, default='Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax')
    
    
    opt = parser.parse_args()
    str_ids = opt.gpu_ids.split(',')    
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        opt.gpu_ids.append(id)
        
    str_class_weight = opt.class_weight.split(',')
    opt.class_weight = []
    for str_class_weight in str_class_weight:
        class_weight = (89694.0 - float(str_class_weight))/float(str_class_weight)
        opt.class_weight.append(class_weight)
    
    opt.class_name = opt.class_name.split(',')
    
    opt.alpha = float(opt.alpha)
    opt.lr = float(opt.lr)
    opt.weight_decay = float(opt.weight_decay)
    
    return opt

def load_data(path, mode,batch_size, name, drop_last=False):
    data  = MyDataset(path,mode,name)
    loader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=drop_last)
    dataset_size = len(loader)
    print('The number of '+mode+' batches = %d' % dataset_size)
    return loader, dataset_size


