import torch
import torch.nn as nn
from collections import OrderedDict 

model1 = nn.Sequential(OrderedDict({
        'conv_1': nn.Conv2d(1, 10, kernel_size=5),
        'conv_2': nn.Conv2d(10, 20, kernel_size=5),
        'dropout': nn.Dropout2d(),
        'linear_1': nn.Linear(320, 50),
        'linear_2': nn.Linear(50, 10)
        }))

model2 = nn.Sequential(OrderedDict({
        'conv_1': nn.Conv2d(1, 10, kernel_size=5),
        'conv_2': nn.Conv2d(10, 20, kernel_size=5),
        'dropout': nn.Dropout2d(),
        'linear_1': nn.Linear(320, 50),
        'linear_2': nn.Linear(50, 10)
        }))

model3 = nn.Sequential(OrderedDict({
        'conv_1': nn.Conv2d(1, 10, kernel_size=5),
        'conv_2': nn.Conv2d(10, 20, kernel_size=5),
        'dropout': nn.Dropout2d(),
        'linear_1': nn.Linear(320, 50),
        'linear_2': nn.Linear(50, 10)
        }))

model = nn.Sequential()
model.add_module('model1',model1)
model.add_module('model2', model2)
model.add_module('model3', model3)

layer = model._modules["model1"]
print(layer)