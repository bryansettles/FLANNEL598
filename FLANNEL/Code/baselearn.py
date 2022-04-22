#!/usr/bin/env python
modelType = 'alexnet' #set as needed

import os
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def loadModData():
    batch_size=32
    data_path = "../../images/"
    train_base_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.ToTensor()
        ])
    test_base_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.ToTensor()
        ])
    train_red_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=.2, contrast=.2),
        ])
    test_red_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    return torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=data_path + "/train/", transform=train_base_trans), batch_size=batch_size, shuffle=True, drop_last=False), torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=data_path  + "/val/", transform=test_base_trans), batch_size=batch_size, shuffle=False, drop_last=False)


trainData, validationData = loadModData() 
imageClass = ('Covid-19', 'Normal', 'Pneumonia')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
my_model = torchvision.models.__dict__[modelType](pretrained=True)
my_model.to(device)
if modelType == 'resnet152':
    my_model.fc = torch.nn.Linear(2048, 3, bias=True)
elif modelType == 'alexnet':
    my_model.classifier[6] = torch.nn.Linear(my_model.classifier[6].in_features, 3, bias=True)
elif modelType == 'densenet161':
    my_model.classifier = torch.nn.Linear(2208, 3, bias=True)
my_model.eval()

if modelType == 'alexnet':
    my_model.features = torch.nn.DataParallel(my_model.features)
    # my_model.cuda()
else:
    my_model = torch.nn.DataParallel(my_model)
    # my_model.cuda()

optimizer = torch.optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)

def train_model(mod, mod_trainData):
    mod.train() 
    n_epoch = 1
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in mod_trainData:
            # data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            loss = criterion(mod(data), target)
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())
    return mod

if os.path.exists(modelType + '.pth'):
    my_model = torch.load(modelType + '.pth')
else:
    my_model = train_model(my_model, trainData)
    torch.save(my_model, modelType + '.pth')

def eval_model(mod, data):
    testY = list()
    predY = list()
    mod.eval()
    for d, t in data:
        # d, t = d.cuda(), t.cuda()
        _, predYicted = torch.max(mod(d).data, 1)
        predY = np.append(predY,predYicted.detach().cpu().numpy())
        testY = np.append(testY,t.detach().cpu().numpy())
    return predY, testY

predY, trueY = eval_model(my_model, validationData)
print("Accuracy: " + str(accuracy_score(trueY, predY)))

print(classification_report(trueY, predY))
print(confusion_matrix(trueY, predY))


cm = confusion_matrix(trueY, predY)
# display = ConfusionMatrixDisplay(cm, imageClass).plot()
# plt.show()
totalcount = 0
correctcount = 0
with torch.no_grad():
    for d in validationData:
        img, lbls = d[0].to(device), d[1].to(device)
        totalcount += lbls.size(0)
        correctcount += (torch.max(my_model(img).data, 1)[1] == lbls).sum().item()

totalc = [0.0,0.0,0.0]
correctc = [0.0,0.0,0.0]

with torch.no_grad():
    for d in validationData:
        img, lbls = d[0].to(device), d[1].to(device)
        c = (torch.max(my_model(img), 1)[1] == lbls).squeeze()
        for i in range(4):
            totalc[lbls[i]] += 1
            correctc[lbls[i]] += c[i].item()

for i in range(3):
    print('Accuracy of %5s : %2d %%' % (
         imageClass[i], 100 * correctc[i] / totalc[i]))

