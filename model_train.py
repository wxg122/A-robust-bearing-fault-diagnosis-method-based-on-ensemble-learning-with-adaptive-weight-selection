from torch.utils.data import DataLoader
from util import netDataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import torch.optim as optim
import matplotlib.ticker as mtick

import random

def eval(model,test_loader):
    eval_loss=0.0
    total_acc=0.0
    model.eval()

    for i,batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits= model(x)
            batch_loss=loss_function(logits,y)
            eval_loss+=batch_loss.item()
            _,preds= logits.max(1)
            num_correct=(preds==y).sum().item()
            total_acc+=num_correct

    loss=eval_loss/len(test_loader)
    acc=total_acc/(len(test_loader)*eval_batch_size)
    return loss,acc


if __name__ == "__main__":
    device = torch.device("cuda")
    print("load dataset.........................")
    trainpath='image/train/'
    valpath='image/val/'

    img_size=64
    train_batch_size=8
    eval_batch_size=8
    learning_rate = 1e-3
    weight_decay=1e-5
    total_epoch = 50

    train_lines = os.listdir(trainpath)
    val_lines = os.listdir(valpath)
    model = Model(num_classes=10).to(device)
    train_dataset = netDataset(trainpath,train_lines,img_size)
    val_dataset = netDataset(valpath,val_lines,img_size)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size,
                              num_workers=0, pin_memory=True,drop_last=True)
    test_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size,
                             num_workers=0,pin_memory=True,drop_last=True)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)

    print("training.........................")
    val_loss_list=[]
    val_acc_list=[]
    train_loss_list=[]
    train_acc_list=[]
    max_acc = 0
    for i in range(total_epoch):
        model.train()
        train_loss=0
        for step, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss=loss_function(logits,y)
            train_loss +=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss=train_loss/len(train_loader)
        train_loss_list.append(train_loss)
        _, train_acc = eval(model, train_loader)
        train_acc_list.append(train_acc)

        print("train Epoch:{},loss:{},train_acc:{}".format(i,train_loss,train_acc))

        eval_loss, eval_acc = eval( model, test_loader)
        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)

        print("val Epoch:{},eval_loss:{},eval_acc:{}".format(i, eval_loss, eval_acc))
        if eval_acc > max_acc:
            max_acc=eval_acc
            torch.save(model, 'output/best.pt')

        for name, p in model.named_parameters():
            if name == 'W':
                print("特征权重: ", name)
                w1 = (torch.exp(p[0]) / torch.sum(torch.exp(p))).item()
                w2 = (torch.exp(p[1]) / torch.sum(torch.exp(p))).item()
                w3 = (torch.exp(p[2]) / torch.sum(torch.exp(p))).item()
                print("w1={} w2={} w3={}".format(w1, w2, w3))
                print("")

    torch.save(model, 'output/last.pt')
    np.savetxt("output/train_loss_list1.txt", train_loss_list)
    np.savetxt("output/train_acc_list1.txt", train_acc_list)
    np.savetxt("output/val_loss_list1.txt",val_loss_list)
    np.savetxt("output/val_acc_list1.txt",val_acc_list)


    with open('output/train_loss_list1.txt','r') as f:
        train_loss_list=f.readlines()
    train_loss=[float(i.strip()) for i in train_loss_list]
    with open('output/val_loss_list1.txt','r') as f:
        val_loss_list=f.readlines()
    val_loss=[float(i.strip()) for i in val_loss_list]

    with open('output/train_acc_list1.txt','r') as f:
        train_acc_list=f.readlines()
    train_acc=[float(i.strip())*100 for i in train_acc_list]
    with open('output/val_acc_list1.txt','r') as f:
        val_acc_list=f.readlines()
    val_acc=[float(i.strip())*100 for i in val_acc_list]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, color='blue', linestyle='-.', marker='o', label='Training Loss')
    plt.plot(val_loss, color='red', linestyle='-.', marker='x', label='Validation Loss')
    plt.legend(fontsize='18')
    plt.xlabel('Epoch', fontsize='18')
    plt.ylabel('AvgLoss', fontsize='18')
    plt.xticks(fontsize='16')
    plt.yticks(fontsize='16')
    plt.savefig('result/Training and validation loss.jpg', dpi=600)

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, color='blue', linestyle='-.', marker='o', label='Training Accuracy')
    plt.plot(val_acc, color='red', linestyle='-.', marker='x', label='Validation Accuracy')
    plt.legend(fontsize='18')
    plt.xlabel('Epoch', fontsize='18')
    plt.ylabel('AvgAcc (%)', fontsize='18')
    plt.xticks(fontsize='16')
    plt.yticks(fontsize='16')
    plt.savefig('result/Training and validation accuracy.jpg', dpi=600)

    plt.show()