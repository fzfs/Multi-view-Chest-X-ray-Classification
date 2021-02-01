import torch
import uitl
import numpy as np

if __name__ == '__main__':    
        
    opt = uitl.parser()
    
    lr = opt.lr
    tm = opt.tm
    view = opt.view
    alpha = opt.alpha
    network = opt.network
    gpu_ids = opt.gpu_ids
    batch_size = opt.batch_size    
    class_name = opt.class_name
    class_weight = opt.class_weight
    weight_decay = opt.weight_decay
    fn = 'runs/' + str(network)+'_'+str(view)+'_'+str(weight_decay)+'_'+str(alpha)+'_'+str(tm)
    test_loader, test_dataset_size = uitl.load_data('data/mimic-cxr-2.0.0-pair-test.csv','test',batch_size,class_name)
    
    model_path = fn+'/model_auc.pkl'
        
    model = torch.load(model_path)
    
    with torch.no_grad():
        test_loss = 0.0
        predict_all = np.array([], dtype=int)        
        predict_all1 = np.array([], dtype=int)
        predict_all2 = np.array([], dtype=int)
        model.eval()
        for k, test_data in enumerate(test_loader):
            model.set_input(test_data)
            model()
            
            test_loss += float(model.loss)
            predicted, labels = model.predicted_label()
            if k == 0:
                p_all = predicted
                l_all = labels
            else:
                p_all = np.vstack((p_all, predicted))
                l_all = np.vstack((l_all, labels))
            
            if view == 6:            
                predicted1, predicted2 = model.single_predicted_label()            
                if k == 0:
                    p_all1 = predicted1
                    p_all2 = predicted2
                else:
                    p_all1 = np.vstack((p_all1, predicted1))
                    p_all2 = np.vstack((p_all2, predicted2))
            
        _ = uitl.print_metrics(l_all,p_all,test_loss/(k+1),mode='test',name=fn,view=view, class_name=class_name)
        if view == 6:
            _ = uitl.print_metrics(l_all,p_all1,test_loss/(k+1),mode='test', name=fn,view=view, class_name=class_name)
            _ = uitl.print_metrics(l_all,p_all2,test_loss/(k+1),mode='test', name=fn,view=view, class_name=class_name)