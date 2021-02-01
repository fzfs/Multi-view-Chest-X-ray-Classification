import torch
import networks
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
class Net(nn.Module):
    def __init__(self, lr, weight_decay, class_weight, init_type, gpu_ids,dataset_size,view,alpha,network):
        super(Net, self).__init__()
        self.view = view
        self.gpu_ids = gpu_ids
        self.alpha = alpha
        self.network = network
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 

        self.model = networks.define_X(init_type, gpu_ids,view,network)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor(class_weight).cuda(gpu_ids[0]))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.CyclicLR(self.optimizer,base_lr=lr, max_lr=10*lr, step_size_up=dataset_size//2, step_size_down=dataset_size-dataset_size//2,cycle_momentum=False, mode='triangular2')
        self.softmax = nn.Softmax()
        
    def forward(self):
        #{1:Frontal, 2:Lateral, 3:DualNet, 4:Stacted, 7:HeMIS,
        # 5:Our method without Auloss and mimicry loss, 
        # 6:Our method with Auloss and mimicry loss}        
        
        if self.view == 1:
            self.predict = self.model(self.frontal)
            self.loss = self.criterion(self.predict.double(), self.label.double())
        elif self.view == 2:
            self.predict = self.model(self.lateral)
            self.loss = self.criterion(self.predict.double(), self.label.double())
        elif self.view == 4:
            self.predict = self.model(torch.cat((self.frontal, self.lateral), dim=1))
            self.loss = self.criterion(self.predict.double(), self.label.double())
        elif self.view == 3 or self.view == 5 or self.view == 7:
            self.predict = self.model(self.frontal, self.lateral)
            self.loss = self.criterion(self.predict.double(), self.label.double())                                  
        elif self.view == 6 :
            self.predict, predict1, predict2 = self.model(self.frontal, self.lateral)
            self.predict1 = torch.sigmoid(predict1)
            self.predict2 = torch.sigmoid(predict2)
            temp = torch.sigmoid(self.predict)
            self.loss = self.alpha * (F.mse_loss(self.predict1,temp,reduction='none').sum(1).mean().float().cuda(self.gpu_ids[0]) + \
                        F.mse_loss(self.predict2,temp,reduction='none').sum(1).mean().float().cuda(self.gpu_ids[0]) + \
                        F.mse_loss(self.predict1,self.predict2,reduction='none').sum(1).mean().float().cuda(self.gpu_ids[0]))+ \
                        self.criterion(self.predict.double(), self.label.double()).float()+ \
                        self.criterion(predict1.double(), self.label.double()).float()+ \
                        self.criterion(predict2.double(), self.label.double()).float()
        self.predicted = torch.sigmoid(self.predict)
    
    def set_input(self, input):
        self.frontal = input['frontal'].to(self.device)
        self.lateral = input['lateral'].to(self.device)
        self.label = input['label'].to(self.device)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
    def predicted_label(self):
        a = self.predicted.cpu().numpy()
        b = self.label.cpu().numpy()
        return a,b
    
    def single_predicted_label(self):
        a = self.predict1.cpu().numpy()
        b = self.predict2.cpu().numpy()
        return a,b
        
    def update_learning_rate(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']
    
    def print_networks(self):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))