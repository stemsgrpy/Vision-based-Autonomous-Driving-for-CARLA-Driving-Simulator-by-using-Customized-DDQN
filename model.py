import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = F.relu(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Create_DQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(Create_DQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(

            nn.Conv2d(self.inut_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            
            nn.Linear(64*7*7, 512), # nn.Linear(self.features_size(), 512),         
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

class Create_ResDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(Create_ResDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.inchannel = 32
        self.conv1 = nn.Sequential(

            nn.Conv2d(self.inut_shape, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 1, stride=2)

        self.fc = nn.Sequential(
            # 2 -> (v, a)
            nn.Linear(512+2, 512), # nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        
        # v = x[:,6,0,0]
        # a = x[:,7,0,0]

        v=torch.reshape(x[:,6,0,0],(-1,1))
        a=torch.reshape(x[:,7,0,0],(-1,1))
        x = x[:,:6,:,:]

        # print(v.shape, a.shape, x.shape)
        # print(v, a)
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 6)
        x = x.view(x.size(0), -1)

        c = torch.cat( (x,v), 1)
        # print(c.shape, c[:,512])
        c = torch.cat( (c,a), 1) 
        # print(c.shape, c[:,513])
        # exit()
        x = self.fc(c)
        return x