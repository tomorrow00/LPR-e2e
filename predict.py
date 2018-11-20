from __future__ import print_function

from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from dataset import DatasetFromFolder

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

labeldict_Chi = dict()
labeldict_Letter = dict()
labeldict_NL = dict()

def initdict():
    labelmap_Chi = open('dataset/dict/labelmap_Chinese.txt', "r")
    lines = labelmap_Chi.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_Chi[temp[1]] = temp[0]
    labelmap_Chi.close()

    labelmap_Letter = open('dataset/dict/labelmap_Letter.txt', "r")
    lines = labelmap_Letter.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_Letter[temp[1]] = temp[0]
    labelmap_Letter.close()

    labelmap_NL = open('dataset/dict/labelmap_Num&Letter.txt', "r")
    lines = labelmap_NL.readlines()
    for line in lines:
        line = line.strip("\n")
        temp = line.split(" ")
        labeldict_NL[temp[1]] = temp[0]
    labelmap_NL.close()


def get_test_set():
    root_dir =  'dataset'
    test_dir = join(root_dir, "validation")
    crop_size = (72,272)
    return DatasetFromFolder(test_dir,
                             input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(),]))

def test(model):
    test_loss = 0
    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    correct6 = 0
    with torch.no_grad():
        for batch in testing_data_loader:
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
            test_loss += criterion(prediction[0], target0).item()
            test_loss += criterion(prediction[1], target1).item()
            test_loss += criterion(prediction[2], target0).item()
            test_loss += criterion(prediction[3], target1).item()
            test_loss += criterion(prediction[4], target1).item()
            test_loss += criterion(prediction[5], target0).item()
            test_loss += criterion(prediction[6], target1).item()
            
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

    test_loss /= len(testing_data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%)\n'.format(
        test_loss, correct0, len(testing_data_loader.dataset),
        100. * correct0 / len(testing_data_loader.dataset), 100. * correct1 / len(testing_data_loader.dataset),
        100. * correct2 / len(testing_data_loader.dataset),
        100. * correct3 / len(testing_data_loader.dataset),
        100. * correct4 / len(testing_data_loader.dataset),
        100. * correct5 / len(testing_data_loader.dataset),
        100. * correct6 / len(testing_data_loader.dataset),
        ))

device = torch.device("cuda")
test_set = get_test_set()
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=100, shuffle=False)
criterion = nn.CrossEntropyLoss().to(device)
model = torch.load('model_epoch_100.pth') 

initdict()
test(model)