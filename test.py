
import torch.nn as nn
import torch
from utils import show_keypoints, AverageMeter
import pandas as pd
import cv2

raw_data = pd.read_csv('data_under_scene.csv',header=0)
dataset = raw_data.values
X = dataset[:, 0:36].astype(float)
Y = dataset[:, 36:].astype(float)
Y = (Y==3)*1 #fall_down作为正样本，其余数据都作为负样本  #TODO
ckpt = torch.load('./best.pt')
net = ckpt['model'].float().eval()
for x, y in zip(X,Y):
    img = show_keypoints(x)
    inp = torch.tensor(x)
    with torch.no_grad():
            predict = net(inp.float().cuda()).sigmoid()
    action = (predict.detach().cpu().numpy()[0] > 0.5)*1
    s = f'p: {action}, t: {y[0]}'
    cv2.putText(img, s, (100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.imshow('result',img)
    key = cv2.waitKey(0)
    print(ckpt['epoch'])