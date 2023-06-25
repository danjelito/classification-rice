import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import module
import config


# import dataset
dir= Path.cwd() / 'dataset/Rice_Image_Dataset'
images, labels= module.load_image_dataset(dir, shuffle= True)

# encode label
encoder= LabelEncoder()
labels= encoder.fit_transform(np.array(labels))

# train val test split
test_size= 0.1
val_size= 0.1
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images, labels, 
    test_size= test_size, 
    stratify= labels,
    random_state= config.RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size= val_size, 
    stratify= y_train_val,
    random_state= config.RANDOM_STATE)

# create transformation pipeline
# resize data 
import torchvision.transforms as transforms
img_height, img_width = 64, 64
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width), 
                      antialias= True),
])

# create dataset class
from torch.utils.data import Dataset, DataLoader
class ImageDataset(Dataset):
    
    def __init__(self, file_list, labels, transform= None):
        self.file_list = file_list
        self.labels = labels
        self.transform= transform
    
    def __getitem__(self, index):
        img= Image.open(self.file_list[index])
        if self.transform is not None:
            img= self.transform(img)
        label= self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)
    
# create dataset
batch_size= 32
train_set= ImageDataset(X_train, y_train, transform)
val_set= ImageDataset(X_val, y_val, transform)
train_dl= DataLoader(train_set, batch_size= batch_size)
val_dl= DataLoader(val_set, batch_size= batch_size)

# create model
import torch.nn as nn

class MyCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # conv 1
        conv1= nn.Conv2d(in_channels= 3, 
                         out_channels= 32, 
                         kernel_size= 3, 
                         padding= 1)
        relu1= nn.ReLU()
        pool1= nn.MaxPool2d(kernel_size= 2)
        # conv 2
        conv2= nn.Conv2d(in_channels= 32, 
                         out_channels= 64, 
                         kernel_size= 3, 
                         padding= 1)
        relu2= nn.ReLU()
        pool2= nn.MaxPool2d(kernel_size= 2)
        # linear
        f= nn.Flatten()
        lin1= nn.Linear(16384, 1024)
        relu3= nn.ReLU()
        lin2= nn.Linear(1024, 5) # 5 classes
        activ= nn.Softmax(dim= 1)
        self.module_list= nn.ModuleList([
            conv1, relu1, pool1, conv2, relu2, pool2, 
            f, lin1, relu3, lin2, activ
        ])

    def forward(self, X):
        for f in self.module_list:
            X= f(X)
        return X

# define training function
def train(model, n_epochs, train_dl, val_dl):
    
    n_train_samples= len(train_dl.dataset)
    n_val_samples= len(val_dl.dataset)
    
    loss_hist_train= [0] * n_epochs
    acc_hist_train= [0] * n_epochs
    loss_hist_val= [0] * n_epochs
    acc_hist_val= [0] * n_epochs
    
    for epoch in range(n_epochs):
        
        model.train()
        for X_batch, y_batch in train_dl:
            y_pred_proba= model(X_batch)
            y_pred= torch.argmax(y_pred_proba, dim= 1)
            loss= loss_fn(y_pred_proba, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            is_correct= (y_batch == y_pred).float()
            acc_hist_train[epoch] += is_correct.sum()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
        loss_hist_train[epoch] /= n_train_samples
        acc_hist_train[epoch] /= n_train_samples

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                y_pred_proba= model(X_batch)
                y_pred= torch.argmax(y_pred_proba, dim= 1)
                loss= loss_fn(y_pred_proba, y_batch)
                is_correct= (y_batch == y_pred).float()
                acc_hist_val[epoch] += is_correct.sum()
                loss_hist_val[epoch] += loss.item() * y_batch.size(0)
            loss_hist_val[epoch] /= n_val_samples
            acc_hist_val[epoch] /= n_val_samples
    
        train_acc= acc_hist_train[epoch]
        val_acc= acc_hist_val[epoch]
        print(f'Epoch {epoch + 1}\n'
              f'Train accuracy = {train_acc: .3f}, '
              f'val accuracy = {val_acc: .3f}')
    
    return (loss_hist_train, 
            acc_hist_train, 
            loss_hist_val, 
            acc_hist_val)

model= MyCNN()
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs= 5
hist= train(model, n_epochs, train_dl, val_dl)
loss_hist_train, acc_hist_train, loss_hist_val, acc_hist_val= hist