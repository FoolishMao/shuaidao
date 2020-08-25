import pandas as pd
from utils import show_keypoints, AverageMeter, compute_accuracy
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np


'''
    stand = 0
    walk = 1
    operate = 2
    fall_down = 3
    run = 4
'''
# 超参数
batch_size = 64
learning_rate = 0.1
weight_decay = 0.0005
num_epochs = 20
gpus = '0'

best_auc = 0
last = './last.pt'
best = './best.pt'

#分配GPU
os.environ["CUDA_VISIBLE_DEVICES"]=gpus
# load data
raw_data = pd.read_csv('data_under_scene.csv',header=0)
dataset = raw_data.values
X = torch.tensor(dataset[:, 0:36].astype(float))
Y = torch.tensor(dataset[:, 36:].astype(float))
Y = (Y==3)*1 #fall_down作为正样本，其余数据都作为负样本  #TODO 制定合理的正负样本比例
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9) #数据集拆分为训练集和测试集 #TODO 使用K折交叉验证
net = nn.Sequential(
    nn.Linear(36, 128),
    nn.Dropout(),
    nn.ReLU(inplace=True),
    nn.Linear(128,64),
    nn.Dropout(),
    nn.ReLU(inplace=True),
    nn.Linear(64,1),
)
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size)
# 这里使用了Adam优化算法
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
loss = nn.BCELoss()
net.float().cuda()
# 训练
for epoch in range(num_epochs):
    net.train()
    loss_data = AverageMeter()
    pbar = enumerate(train_iter)
    nb = len(train_iter)
    pbar = tqdm(train_iter, total=nb)
    results = []
    labels = []
    for inp, out in pbar:
        predict = net(inp.float().cuda()).sigmoid()
        l = loss(predict, out.float().cuda())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        #show
        predict = predict.detach().cpu().numpy().tolist()
        results += predict
        out = out.numpy().tolist()
        labels += out
        loss_data.update(l.item())
        s = ('%10s' * 2 + '%10.4g' * 1) % (epoch+1, num_epochs, loss_data.avg)
        pbar.set_description(s)
    acc = compute_accuracy(np.array(labels).flatten(), np.array(results).flatten()) #TODO 尝试除了准确率之外其他的评价指标来判断模型的好坏
    s = f'train_acc: {acc}'
    print(s)
    # 测试
    net.eval()
    results = []
    labels = []
    for inp, out in val_iter:
        with torch.no_grad():
            predict = net(inp.float().cuda()).sigmoid()
        predict = predict.detach().cpu().numpy().tolist()
        results += predict
        out = out.numpy().tolist()
        labels += out
    acc = compute_accuracy(np.array(labels).flatten(), np.array(results).flatten())
    s = f'test_acc: {acc}'
    print(s)
    ckpt = {'epoch': epoch,
            'model': net.module if hasattr(net, 'module') else net,}

    # Save last, best and delete
    if acc >= best_auc:
        best_auc = acc
        torch.save(ckpt, best)
    del ckpt