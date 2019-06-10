
import torch
import numpy as np


from torchvision import datasets
from os import path
import glob

from torch.utils.data.sampler import SubsetRandomSampler

import os
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

gpu_available = torch.cuda.is_available()
print(gpu_available)



transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.Resize(30),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



train_data_path = '/content/Fruit-Images-Dataset/Training'
test_data_path = '/content/Fruit-Images-Dataset/Test'

classes = sorted(os.listdir(train_data_path))
train_data = datasets.ImageFolder(train_data_path,transform=transforms)
test_data = datasets.ImageFolder(test_data_path,transform=transforms)


num_workers = 0
batch_size = 5
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
#max_index = np.max(indices)
#train_idx = indices
split = int(np.floor(valid_size * num_train))
train_idx,valid_idx = indices[split:],indices[:split]

num_test = len(test_data)
test_indices = list(range(num_test))
np.random.shuffle(test_indices)
test_idx = test_indices


train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(train_data,num_workers=num_workers,batch_size=batch_size,sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(test_data,num_workers=num_workers,batch_size=batch_size, sampler = test_sampler)
valid_loader = torch.utils.data.DataLoader(train_data,num_workers=num_workers,batch_size=batch_size,sampler = valid_sampler)


def imshow(img):
  img= img/2 + 0.5
  plt.imshow(np.transpose(img,(1,2,0)))

dataiter = iter(train_loader)
images,labels = dataiter.next()
images = images.numpy()

print(labels)
print(images.shape)
fig = plt.figure(figsize=(25,4))
for idx in np.arange(5):
  print(idx)
  ax = fig.add_subplot(2, 3 ,idx+1 , xticks = [], yticks=[])
  imshow(images[idx])
  try:
    ax.set_title(classes[labels[idx]])

  except:
    ax.set_title("None")

classes[79]

#@title Define the Architecture


class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    #Convolutional Layers
    self.conv1 = nn.Conv2d(3,15,3,padding=1)
    self.conv2 = nn.Conv2d(15,30,3,padding=1)
    self.conv3 = nn.Conv2d(30,60,3,padding=1)

    #Max Pooling Layers
    self.pool = nn.MaxPool2d(2,2)

    #Linear Fully Connected layers
    self.fc1 = nn.Linear(60*3*3,500)
    self.fc2 = nn.Linear(500,103)

    #Dropout
    self.dropout = nn.Dropout(p=0.25)


  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1,60*3*3)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x

model = Net()
print(model)
if gpu_available:
  print(gpu_available)
  model.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)


print("Training the Network")
epochs = 20
valid_loss_min = np.Inf
for epoch in range(1,epochs+1):
  train_loss = 0
  valid_loss = 0
  model.train()

  for data,target in train_loader:
    if gpu_available:
      data,target = data.cuda(),target.cuda()

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*data.size(0)



  #####################
  ##    Validate     ##
  ######################
  model.eval()
  for data,target in valid_loader:
    if gpu_available:
      data,target = data.cuda(),target.cuda()

    output = model(data)
    loss = criterion(output,target)
    valid_loss += loss.item()*data.size(0)


train_loss = train_loss/len(train_loader.dataset)
valid_loss = valid_loss/len(valid_loader.dataset)
print("Epoch {} ... Train Loss: {:.6f}....Valid_loss: {0:.6f}".format(epoch, train_loss,valid_loss))

if valid_loss < valid_loss_min:
  torch.save(model.state_dict(),'model_fruit.pt')
  valid_loss_min = valid_loss
  print('Validation Loss Decreased: {:.6f} >>>> {:.6f}'.format(valid_loss_min,valid_loss))



model.load_state_dict(torch.load('model_fruits.pt'))

#@title Testing the Network

test_loss = 0.0
class_correct = list(0.for i in range(103))
class_total = list(0.for i in range(103))


model.eval()

for data,target in test_loader:
  if gpu_available:
    data,target = data.cuda(),target.cuda(),

  output = model(data)
  loss = criterion(output,target)
  test_loss += loss.item()*data.size(0)
  _,pred = torch.max(output,1)
  correct_tensor = pred.eq(target.data.view_as(pred))
  correct = np.squeeze(correct_tensor.numpy) if not gpu_available else np.squeeze(correct_tensor.cpu().numpy())

  #Calculate test accuracy for each class
  for i in range(batch_size):
    label = target.data[i]
    class_correct[label] += correct[i].item()
    class_total[label] +=1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}'.format(test_loss))

for i in range(103):
  if class_total[i]>0:
    print('Test Accuracy of %5s: %2d%% (%2d/%2d)'% (
    classes[i],100*class_correct[i]/class_total[i],
    np.sum(class_correct[i]),np.sum(class_total[i])))

  else:
    print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\n Test Accuracy(overall): %2d%% (%2d/%2d)' % (100*np.sum(class_correct)/np.sum(class_total),np.sum(class_correct),np.sum(class_total)))

import random

dataiter = iter(test_loader)
images,labels = dataiter.next()
print(labels)
images.numpy()


if gpu_available:
  imagescuda = images.cuda()


output = model(imagescuda)
_,preds_tensor = torch.max(output,1)
pred = np.squeeze(preds_tensor.numpy()) if not gpu_available else np.squeeze(preds_tensor.cpu().numpy())


figure = plt.figure(figsize =(25,4))

for idx in np.arange(5):
  ax = figure.add_subplot(2 , 6/2, idx+1 , xticks = [], yticks=[])
  imshow(images[idx])
  ax.set_title('{} ({})'.format(classes[pred[idx]],classes[labels[idx]]),
              color = 'green' if pred[idx]==labels[idx].item() else 'red')
