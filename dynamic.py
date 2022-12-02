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
epoch = 1000


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
        self.layer = nn.Linear(8, 32 * 32)
    
    def forward(self, u):
        output = self.layer(u)
        return output.reshape(-1, 32, 32)
        

class CustomDataset(Dataset):
    def __init__(self, filepath):
        f = np.load(filepath)
        # self.action = f[0]
        # self.before = f[1]
        # self.after = f[2]
        self.action = torch.Tensor(f['action'])
        self.transforms = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()]
        )
    
    def __len__(self):
        return len(self.action)
    
    def __getitem__(self, idx):
        # action = torch.Tensor(self.action[idx])
        # before = torch.Tensor(self.before[idx])
        # after = torch.Tensor(self.after[idx])
        # before = torch.mean(before, dim=-1) / 255
        # after = torch.mean(after, dim=-1) / 255
        action = self.action[idx]
        before = self.transforms(Image.open('envs/data/before/{}.jpg'.format(idx))).squeeze()
        after = self.transforms(Image.open('envs/data/after/{}.jpg'.format(idx))).squeeze()
        return action, before, after
        

model = NetAsMatrix()
train_loader = DataLoader(CustomDataset('envs/data/action_0.npz'), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset('envs/data/action_0.npz'), batch_size=batch_size, shuffle=False)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()


def train(dataloader):
    model.train()
    for _, (act, origin, goal) in enumerate(dataloader):
        optimizer.zero_grad()
        matrix = model(act)
        # print(matrix.shape)
        # print(torch.matmul(matrix, origin).shape)
        # break
        loss = loss_fn(torch.matmul(matrix, origin), goal)
        loss.backward()
        optimizer.step()


def test(model, dataloader):
    model.eval()
    loss = 0
    for _, (act, origin, goal) in enumerate(dataloader):
        matrix = model(act)
        loss += loss_fn(torch.matmul(matrix, origin), goal).detach()
    return loss


def main():
    loss = []
    for _ in tqdm(range(epoch)):
        train(train_loader)
        loss.append(test(model, test_loader))
    torch.save(model, 'best.pt')
    plt.plot(loss)
    plt.draw()
    plt.savefig('image.png')

main()

model = torch.load('best.pt')
for _, (a, o, g) in enumerate(test_loader):
    matrix = model(a)
    predict = torch.matmul(matrix, o)
    transforms.ToPILImage()(predict[0]).save('predict.png')
    transforms.ToPILImage()(g[0]).save('gt.png')
    break
        