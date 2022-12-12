import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import copy
from torch.optim import lr_scheduler
from torchsummary import summary


class CustomDataset(Dataset):
    def __init__(self, mode):
        self.filepath = 'envs/data/{}'.format(mode)
        f = np.load('{}/action.npz'.format(self.filepath))
        self.action = torch.Tensor(f['action'])
        self.transforms = transforms.Compose(
            [transforms.Grayscale(3),
            transforms.ToTensor()]
        )
    
    def __len__(self):
        return len(self.action)
    
    def __getitem__(self, idx):
        before = self.transforms(Image.open('{}/before/{}.jpg'.format(self.filepath, idx)))
        after = self.transforms(Image.open('{}/after/{}.jpg'.format(self.filepath, idx % len(self.action))))
        return before, self.action[idx], after
    

class EncoderDecoderDataset(Dataset):
    def __init__(self, mode):
        self.filepath = 'envs/data/{}'.format('train')
        f = np.load('{}/action.npz'.format(self.filepath))
        self.action = torch.Tensor(f['action'])
        self.transforms = transforms.Compose(
            [transforms.Grayscale(3),
            transforms.ToTensor()]
        )
    
    def __len__(self):
        # return 2 * len(self.action) // 10
        return 2 * len(self.action)
    
    def __getitem__(self, idx):
        # if idx < len(self.action):
        #     image = self.transforms(Image.open('{}/before/{}.jpg'.format(self.filepath, idx)))
        # else:
        #     image = self.transforms(Image.open('{}/after/{}.jpg'.format(self.filepath, idx % len(self.action))))
        # return image
        return self.transforms(Image.open('{}/before/{}.jpg'.format('envs/data/train', 0)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
train_set = EncoderDecoderDataset('train')
valid_set = EncoderDecoderDataset('valid')
test_set = EncoderDecoderDataset('test')
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
}
    
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
    
    
class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        # self.conv_up3 = convrelu(512, 512, 3, 1)
        # self.conv_up2 = convrelu(512, 256, 3, 1)
        # self.conv_up1 = convrelu(256, 256, 3, 1)
        # self.conv_up0 = convrelu(256, 128, 3, 1)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1)
        # self.conv_original_size2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.actionlayer = nn.Linear(520, 512)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        # layer4 = torch.cat((layer4.squeeze(), action), dim=1)
        # layer4 = self.actionlayer(layer4).unsqueeze(dim=-1).unsqueeze(dim=-1)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        # print(x.shape)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        # print(x.shape)
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # print(x.shape)
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        # print(x.shape)
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        # print(x.shape)
        out = self.conv_last(x)
        # print(out.shape)
        return out.expand(-1, 3, -1, -1) + input 
    
    # def encode(self, input):
    #     x_original = self.conv_original_size0(input)
    #     x_original = self.conv_original_size1(x_original)

    #     layer0 = self.layer0(input)
    #     layer1 = self.layer1(layer0)
    #     layer2 = self.layer2(layer1)
    #     layer3 = self.layer3(layer2)
    #     layer4 = self.layer4(layer3)
    #     return layer4
    
    # def decode(self, input):
    #     pass
    
def freeze(model):
    for idx, child in enumerate(model.children()):
        if idx == 137: # do not freeze the action layer
            continue
        for param in child.parameters():
            param.requires_grad = False
        
def unfreeze(model):
    for idx, child in enumerate(model.children()):
        for param in child.parameters():
            param.requires_grad = True


def calc_loss(pred, target, metrics):
    loss = nn.MSELoss()(pred, target)
    metrics['loss'] += loss.detach() * target.size(0)
    return loss


def train_model(model, optimizer, scheduler, num_epochs=25):
    '''
    This function will train the encoder-decoder network for recovering input images
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'valid']:
            
            model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            # for inputs, actions, labels in tqdm(dataloaders[phase]):
            for inputs in dataloaders[phase]:
                epoch_samples += 1
                inputs = inputs.to(device)
                # actions = actions.to(device)
                # labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs, actions)
                    # loss = calc_loss(outputs, labels, metrics)
                    outputs = model(inputs)
                    loss = calc_loss(outputs, inputs, metrics)
                    
                    if phase == 'train':
                        loss.backward()
                        # optimizer.step()
                        
                # if epoch_samples == 1:
                #     print(inputs[0][0][15])
                #     print(outputs[0][0][15])
                #     print(loss.detach())
                #     print(phase)
            print(phase, ': ', metrics['loss'])        
            epoch_loss = metrics['loss']
            
            if phase == 'train' and epoch_loss < best_loss:
                print('save best model')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'train':
                scheduler.step()
        
    model.load_state_dict(best_model_wts)
    return model


num_class = 1
model = ResNetUNet(num_class).to(device)
print(summary(model, input_size=(3, 32, 32)))
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=10)
torch.save(model.state_dict(), 'unet.pt')
model.load_state_dict(torch.load('unet.pt'))
model.eval()

# inputs, actions, labels = next(iter(dataloaders['test']))
# inputs = inputs.to(device)
# actions = actions.to(device)
# labels = labels.to(device)
# pred = model(inputs, actions).cpu()
inputs = next(iter(dataloaders['train']))
inputs = inputs.to(device)
pred = model(inputs).cpu()
image = transforms.ToPILImage()(pred[1])
image.save('pred.png')
image = transforms.ToPILImage()(inputs[1])
image.save('gt.png')