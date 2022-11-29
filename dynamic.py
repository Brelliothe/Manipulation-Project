import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision import models
from torch.utils.data import dataloader
import numpy as np
from PIL import Image

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 10


class Net(nn.Module):
    """
    Use 3 layers of transformer
    take the initial figure and action as input
    output the final figure
    """
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.decoder = models.resnet18(pretrained=True)
        
    def forward(self, pic, u):
        encode = self.encoder(pic) # shape Batch x 1000
        input = torch.cat((encode, u), dim=0) # shape = Batch x 1008
        decode = self.decoder(input)
    

class NetAsMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 480 * 480)
    
    def forward(self, u):
        output = self.layer(u)
        return output.reshape(-1, 480, 480)
        

net = NetAsMatrix()
f = np.load('data_0.npz')
action, before, after = f['action'], f['before'], f['after']
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for act, origin, goal in zip(action, before, after):
    optimizer.zero_grad()
    matrix = net(torch.Tensor(act))
    loss = nn.MSELoss(torch.matmul(matrix, torch.Tensor(before)), torch.Tensor(after))
    loss.backward()
    optimizer.step()
    
# print(torch.Tensor(before.shape))
# net = Net()
# net(torch.Tensor(before).permute(0, 3, 1, 2), action)