import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicNet(nn.Module):
    def __init__(self, config):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(config['input_channels'], 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        
        self.fc1 = nn.Linear(9216 if config['dataset'] == 'mnist' else 12544, 128)
        self.fc2 = nn.Linear(128, config['num_classes'])
    
    def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            return x
        

if __name__ == '__main__':
    config = {'input_channels': 1,
                'num_classes': 10,
                'dataset' : 'cifar10'
                }
    model = BasicNet(config).cuda()
    summary(model, (1, 32, 32))
    