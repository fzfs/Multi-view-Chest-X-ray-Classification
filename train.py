import uitl
import time
import torch
import numpy as np
from model import Net
from tensorboardX import SummaryWriter

if __name__ == '__main__':    
        
    opt = uitl.parser()    
    lr = opt.lr
    view = opt.view
    alpha = opt.alpha
    epochs = opt.epochs
    network = opt.network
    gpu_ids = opt.gpu_ids
    init_type = opt.init_type
    batch_size = opt.batch_size
    class_name = opt.class_name
    class_weight = opt.class_weight
    weight_decay = opt.weight_decay
    fn = 'runs/' + str(network)+'_'+str(view)+'_'+str(weight_decay)+'_'+str(alpha)+'_'+str(time.time())[-6:]
    writer = SummaryWriter(fn)

    train_loader, train_dataset_size = uitl.load_data('data/mimic-cxr-2.0.0-pair-train.csv','train',batch_size,class_name,True)
    val_loader, val_dataset_size = uitl.load_data('data/mimic-cxr-2.0.0-pair-val.csv','val',batch_size,class_name)

    model = Net(lr, weight_decay, class_weight,init_type, gpu_ids,train_dataset_size,view,alpha,network)
    model.print_networks()
    count = 0
    auc_temp = 0
    loss_temp = 100
    tamp = train_dataset_size//5
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, train_data in enumerate(train_loader):
            count += 1
            model.set_input(train_data)
            model.optimize_parameters()           
            train_loss += float(model.loss)
            model.update_learning_rate() 
            writer.add_scalar('lr', float(model.lr), count)
            writer.add_scalar('train_loss', float(model.loss), count)
            
            
            if i % tamp == 0:
                print("Training: Epoch[{:0>2}/{:0>2}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                epoch + 1, epochs, i + 1, train_dataset_size, train_loss / (i+1)))
     
                with torch.no_grad():
                    val_loss = 0.0
                    predict_all = np.array([], dtype=int)
                    labels_all = np.array([], dtype=int)
                    model.eval()
                    for j, val_data in enumerate(val_loader):
                        model.set_input(val_data)
                        model()
                        val_loss += float(model.loss)
                        predicted, labels = model.predicted_label()
                        if j == 0:
                            p_all = predicted
                            l_all = labels
                        else:
                            p_all = np.vstack((p_all, predicted))
                            l_all = np.vstack((l_all, labels))                            
                    
                    mean = uitl.print_metrics(l_all,p_all,val_loss/(j+1),mode='val',name=fn,view=view, class_name=class_name)
                    if i > 0:
                        writer.add_scalars('loss', {'train_loss_1':train_loss / tamp, 'val_loss':val_loss/val_dataset_size}, count/tamp)                   
                        writer.add_scalars('auc_loss', {'val_mean':mean, 'val_loss':val_loss/val_dataset_size}, count/tamp)
                    model.train()
                    train_loss = 0.0
        
            if mean > auc_temp:
                auc_temp = mean
                torch.save(model, fn+'/model_auc.pkl')
            
            if val_loss/val_dataset_size < loss_temp:
                loss_temp = val_loss/val_dataset_size
                torch.save(model, fn+'/model_loss.pkl')
    writer.close()