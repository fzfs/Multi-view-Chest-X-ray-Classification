# Multi-view-Chest-X-ray-Classification

## train
train.py --epochs 1 --weight_decay 1e-3 --lr 1e-5 --batch_size 64 --network 1 --view 6 --alpha 1.0

## test
test.py  --weight_decay 1e-3 --batch_size 64 --network 3 --view 6 --tm 336385

## view
1:Frontal
2:Lateral
3:DualNet
4:Stacted
5:Our method without Auloss and mimicry loss
6:Our method with Auloss and mimicry loss
7:HeMIS

## network：backbone
1:Resnet
2:Xception
3:Densenet

## alpha：weight coefficient

## tm: a part of the file name

## lr: learning rate
