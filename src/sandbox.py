import torch.nn as nn
import torch


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv 1
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=2)
        # conv 2
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=2)
        # linear
        f = nn.Flatten()
        lin1 = nn.Linear(16384, 1024)
        relu3 = nn.ReLU()
        lin2 = nn.Linear(1024, 5)  # 5 classes
        activ = nn.Softmax(dim=1)
        self.module_list = nn.ModuleList([conv1, relu1, pool1, conv2, relu2, pool2, f])

    def forward(self, X):
        for f in self.module_list:
            X = f(X)
        return X


X = torch.Tensor(32, 3, 48, 64)
model = MyCNN()
print(model(X).shape)
