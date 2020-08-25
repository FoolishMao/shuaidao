
import torch.nn as nn
import torch
from utils import show_keypoints, AverageMeter, compute_accuracy
import pandas as pd
import cv2
import numpy as np

raw_data = pd.read_csv('data_under_scene.csv',header=0)
dataset = raw_data.values
X = dataset[:, 0:36].astype(float)
Y = dataset[:, 36:].astype(float)
Y = (Y==3)*1 #fall_down作为正样本，其余数据都作为负样本  #TODO
ckpt = torch.load('./best.pt')
net = ckpt['model'].float().eval()
results = []
labels = []
for x, y in zip(X,Y):
    img = show_keypoints(x)
    inp = torch.tensor(x)
    with torch.no_grad():
        predict = net(inp.float().cuda()).sigmoid()
    action = (predict.detach().cpu().numpy()[0] > 0.5)*1
    predict = predict.detach().cpu().numpy().tolist()
    results += predict
    y = y.tolist()
    labels += y
    s = f'p: {action}, t: {y[0]}'
    cv2.putText(img, s, (100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.imshow('result',img)
    key = cv2.waitKey(1)
acc = compute_accuracy(np.array(labels).flatten(), np.array(results).flatten()) 
print(acc)