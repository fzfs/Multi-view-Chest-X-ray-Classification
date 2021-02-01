import Cutout
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, csv_path,mode,class_name):
        self.mode = mode
        data = pd.read_csv(csv_path)
        labels = data[class_name]
        labels.fillna(0, inplace=True)
        labels.replace([-1],[0], inplace=True)
        labels = labels.values
        imgs = []
        for index, row in data.iterrows():
            lateral_path = row['path_l']
            frontal_path = row['path_f']
            label = labels[index,:]  
            imgs.append((lateral_path, frontal_path, label))
            
        self.imgs = imgs
        
    def __getitem__(self, index):
        lateral_path, frontal_path, label = self.imgs[index]
        lateral = Image.open(lateral_path).convert('L')
        frontal = Image.open(frontal_path).convert('L')
        
        F_transform = get_transform(self.mode)
        l = F_transform(lateral)
        f = F_transform(frontal)
        
        return {'lateral':l, 'frontal':f, 'label':label}

    def __len__(self):
        return len(self.imgs)
    
def get_transform(mode='train', load_size=224, n_holes=1, length=56):
    transform_list = []
    
    if mode == 'train':
        transform_list += [transforms.RandomHorizontalFlip()]
        transform_list += [transforms.RandomRotation(15)]
        
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5,), (0.5,))]
    if mode == 'train':        
        transform_list += [Cutout.Cutout(n_holes=n_holes, length=length)]    
        
    return transforms.Compose(transform_list)