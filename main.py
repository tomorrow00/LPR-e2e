from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from VGG_pr import Net
from data import get_training_set, get_val_set
from torchvision import models
from torchsummary import summary

import os
import datetime
import logging

labeldict_Chi = dict()
labeldict_Letter = dict()
labeldict_NL = dict()


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--valBatchSize', type=int, default=10, help='validating batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default=0.9')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpu', type=str, default="0", help='GPU to use. Default=0')
# parser.add_argument('--logDir', type=str, default="Log", help='Log directory.')
opt = parser.parse_args()

print(opt)

device = torch.device("cuda")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device("cuda")

torch.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
val_set = get_val_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
validating_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.valBatchSize, shuffle=False)

print('===> Building model')

# model = models.vgg16(pretrained=True)
# model.classifier = Net().classifier
model = models.resnet50(pretrained=True)
model.fc = Net().classifier

model = model.to(device)
print(model)

summary(model, (3, 72, 272))

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), momentum=opt.momentum, lr=opt.lr)

def initdict():
    labelmap_Chi = open('dict/labelmap_Chinese.txt', "r")
    lines = labelmap_Chi.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_Chi[temp[1]] = temp[0]
    labelmap_Chi.close()

    labelmap_Letter = open('dict/labelmap_Letter.txt', "r")
    lines = labelmap_Letter.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_Letter[temp[1]] = temp[0]
    labelmap_Letter.close()

    labelmap_NL = open('dict/labelmap_Num&Letter.txt', "r")
    lines = labelmap_NL.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_NL[temp[1]] = temp[0]
    labelmap_NL.close()

def train(epoch):
    for batch_idx, batch in enumerate(training_data_loader, 1):
        data = batch[0].to(device)
        target0 =  [int(labeldict_Chi[x[0].strip()]) for x in batch[1]]
        target1 =  [int(labeldict_Letter[x[1].strip()]) for x in batch[1]]
        target2 =  [int(labeldict_NL[x[2].strip()]) for x in batch[1]]
        target3 =  [int(labeldict_NL[x[3].strip()]) for x in batch[1]]
        target4 =  [int(labeldict_NL[x[4].strip()]) for x in batch[1]]
        target5 =  [int(labeldict_NL[x[5].strip()]) for x in batch[1]]
        target6 =  [int(labeldict_NL[x[6].strip()]) for x in batch[1]]
             
        target0 = torch.tensor(target0).to(device)
        target1 = torch.tensor(target1).to(device)
        target2 = torch.tensor(target2).to(device)
        target3 = torch.tensor(target3).to(device)
        target4 = torch.tensor(target4).to(device)
        target5 = torch.tensor(target5).to(device)
        target6 = torch.tensor(target6).to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss0 = criterion(output[0], target0)
        loss1 = criterion(output[1], target1)
        loss2 = criterion(output[2], target2)
        loss3 = criterion(output[3], target3)
        loss4 = criterion(output[4], target4)
        loss5 = criterion(output[5], target5)
        loss6 = criterion(output[6], target6)
        (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6).backward()
        # loss1.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
            for param_lr in optimizer.param_groups: #同样是要加module  
                print('lr_rate: ' + str(param_lr['lr']))

def val():
    val_loss = 0
    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    correct6 = 0
    with torch.no_grad():
        for batch in validating_data_loader:
            input = batch[0].to(device)
            target0 =  [int(labeldict_Chi[x[0].strip()]) for x in batch[1]]
            target1 =  [int(labeldict_Letter[x[1].strip()]) for x in batch[1]]
            target2 =  [int(labeldict_NL[x[2].strip()]) for x in batch[1]]
            target3 =  [int(labeldict_NL[x[3].strip()]) for x in batch[1]]
            target4 =  [int(labeldict_NL[x[4].strip()]) for x in batch[1]]
            target5 =  [int(labeldict_NL[x[5].strip()]) for x in batch[1]]
            target6 =  [int(labeldict_NL[x[6].strip()]) for x in batch[1]]
             
            target0 = torch.tensor(target0).to(device)
            target1 = torch.tensor(target1).to(device)
            target2 = torch.tensor(target2).to(device)
            target3 = torch.tensor(target3).to(device)
            target4 = torch.tensor(target4).to(device)
            target5 = torch.tensor(target5).to(device)
            target6 = torch.tensor(target6).to(device)
 
            prediction = model(input)
            val_loss += criterion(prediction[0], target0).item()
            val_loss += criterion(prediction[1], target1).item()
            val_loss += criterion(prediction[2], target2).item()
            val_loss += criterion(prediction[3], target3).item()
            val_loss += criterion(prediction[4], target4).item()
            val_loss += criterion(prediction[5], target5).item()
            val_loss += criterion(prediction[6], target6).item()

            pred0 = prediction[0].max(1, keepdim=True)[1]
            pred1 = prediction[1].max(1, keepdim=True)[1]
            pred2 = prediction[2].max(1, keepdim=True)[1]
            pred3 = prediction[3].max(1, keepdim=True)[1]
            pred4 = prediction[4].max(1, keepdim=True)[1]
            pred5 = prediction[5].max(1, keepdim=True)[1]
            pred6 = prediction[6].max(1, keepdim=True)[1]
            
            correct0 += pred0.eq(target0.view_as(pred0)).sum().item()
            correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target2.view_as(pred2)).sum().item()
            correct3 += pred3.eq(target3.view_as(pred3)).sum().item()
            correct4 += pred4.eq(target4.view_as(pred4)).sum().item()
            correct5 += pred5.eq(target5.view_as(pred5)).sum().item()
            correct6 += pred6.eq(target6.view_as(pred6)).sum().item()
            
    val_loss /= len(validating_data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%)\n'.format(
        val_loss, correct0, len(validating_data_loader.dataset),
        100. * correct0 / len(validating_data_loader.dataset),
        100. * correct1 / len(validating_data_loader.dataset),
        100. * correct2 / len(validating_data_loader.dataset),
        100. * correct3 / len(validating_data_loader.dataset),
        100. * correct4 / len(validating_data_loader.dataset),
        100. * correct5 / len(validating_data_loader.dataset),
        100. * correct6 / len(validating_data_loader.dataset),
        ))


def checkpoint(epoch):
    model_out_path = "./model/model_pr_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

initdict()

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val()
    if(epoch % 50 == 0):
       checkpoint(epoch)
       for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99
