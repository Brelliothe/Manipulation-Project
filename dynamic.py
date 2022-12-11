import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.linear1 = nn.Linear(1008, 16384)
        self.linear2 = nn.Linear(16384, 32 * 32) 
        
    def forward(self, pic, u):
        encode = self.encoder(pic) # shape Batch x 1000
        input = torch.cat((encode, u), dim=1) # shape = Batch x 1008
        output = self.linear2(F.relu(self.linear1(input)))
        output = output.unsqueeze(1).expand(-1, 3, 32 * 32)
        return output.reshape(-1, 3, 32, 32)
    

class NetAsMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.actionlayer1 = nn.Linear(8, 1024)
        self.actionlayer2 = nn.Linear(1024, 32768)
        self.actionlayer3 = nn.Linear(32768, 1024 * 1024)
        self.figurelayer1 = nn.Linear(1024, 32768)
        self.figurelayer2 = nn.Linear(327868, 1024 * 1024)
        
    def forward(self, fig, u):
        u = F.relu(self.actionlayer1(u))
        u = F.relu(self.actionlayer2(u))
        u = self.actionlayer3(u)
        fig = F.relu(self.figurelayer1(fig))
        fig = self.actionlayer2(self.figurelayer2(fig))
        output = fig.reshape(-1, 1024, 1024) + fig.reshape(-1, 1024, 1024)
        return output
        

class CustomDataset(Dataset):
    def __init__(self, filepath):
        f = np.load(filepath)
        # self.action = f[0]
        # self.before = f[1]
        # self.after = f[2]
        self.action = torch.Tensor(f['action'])
        self.transforms = transforms.Compose(
            [# transforms.Grayscale(3),
            transforms.ToTensor()]
        )
    
    def __len__(self):
        # return len(self.action)
        return 5000
    
    def __getitem__(self, idx):
        # action = torch.Tensor(self.action[idx])
        # before = torch.Tensor(self.before[idx])
        # after = torch.Tensor(self.after[idx])
        # before = torch.mean(before, dim=-1) / 255
        # after = torch.mean(after, dim=-1) / 255
        action = self.action[idx]
        before = self.transforms(Image.open('envs/data/before/{}.jpg'.format(idx))).squeeze()
        after = self.transforms(Image.open('envs/data/after/{}.jpg'.format(idx+1))).squeeze()
        return action, before, after


class LSTDataset(Dataset):
    def __init__(self, filepath) -> None:
        f = np.load(filepath)
        self.action = torch.Tensor(f['action'])
        self.matrix = torch.Tensor(f['matrix'])
        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
        
    def __len__(self):
        return len(self.action) * 64
        
    def __getitem__(self, index):
        action = self.action[index // 64]
        before = self.transforms(Image.open('envs/data/before/{}.jpg'.format(index))).squeeze().reshape(32 * 32)
        after = self.transforms(Image.open('envs/data/after/{}.jpg'.format(index))).squeeze().reshape(32 * 32)
        matrix = self.matrix[index // 64]
        return action, matrix, before, after
    

model = NetAsMatrix()
# train_loader = DataLoader(CustomDataset('envs/data/action_0.npz'), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(CustomDataset('envs/data/action_0.npz'), batch_size=batch_size, shuffle=False)
train_loader = DataLoader(LSTDataset('envs/data/lst_action_0.npz'), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(LSTDataset('envs/data/lst_action_0.npz'), batch_size=batch_size, shuffle=False)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.MSELoss()


def train(dataloader):
    model.train()
    for _, (act, matrix, origin, goal) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        output = model(origin, act)
        # loss = loss_fn(torch.clamp(torch.matmul(matrix, origin) + origin, 0, 1), goal)
        # loss = loss_fn(torch.matmul(matrix, origin) + origin, goal) 
        loss = loss_fn(output, matrix)
        loss.backward()
        optimizer.step()


def test(model, dataloader):
    model.eval()
    loss = 0
    for _, (act, matrix, origin, goal) in enumerate(dataloader):
        output = model(origin, act)
        # loss += loss_fn(torch.clamp(torch.matmul(matrix, origin) + origin, 0, 1), goal).detach()
        # loss += loss_fn(torch.matmul(matrix, origin) + origin, goal).detach() # loss_fn(matrix + origin, goal).detach()
        loss += loss_fn(output, matrix).detach()
    return loss


def main():
    loss = []
    for idx in range(epoch):
        train(train_loader)
        loss.append(test(model, test_loader))
        # if idx % 10000 == 9999:
    torch.save(model, 'best.pt')
    plt.plot(loss)
    plt.draw()
    plt.savefig('image.png')

# model = torch.load('best.pt')
main()

model = torch.load('best.pt')
for _, (a, m, o, g) in enumerate(test_loader):
    matrix = model(o, a)
    # print(matrix)
    predict = (torch.matmul(matrix, o) + o).reshape(batch_size, 32, 32)
    # print(matrix)
    # print(loss_fn(predict, g))
    transforms.ToPILImage()(predict[0]).save('predict.png')
    transforms.ToPILImage()(g[0].reshape(32, 32)).save('gt.png')
    transforms.ToPILImage()((m @ o[0]).reshape(32, 32)).save('gt_lst.png')
    # transforms.ToPILImage()(torch.clamp(torch.abs(predict[0] - g[0]), 0, 1)).save('diff.png')
    # transforms.ToPILImage()(torch.clamp(torch.abs(g[0] - o[0]), 0, 1)).save('move.png')
    break
        