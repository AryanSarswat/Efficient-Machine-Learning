from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1, downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)
        return out

class ResNet18_Modified(nn.Module):
    def __init__(self, config, num_layers: int = 18,block: Type[BasicBlock] = BasicBlock) -> None:
        super(ResNet18_Modified, self).__init__()
        
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=config['input_channels'],
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.layer1 = self._make_layer(block, 64, layers[0], depthwise=config['depthwise_convs'][0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, depthwise=config['depthwise_convs'][1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, depthwise=config['depthwise_convs'][2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, depthwise=config['depthwise_convs'][3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, config['num_classes'])
        self.dropout = nn.Dropout(0.25)
        
    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1, depthwise: bool = False) -> nn.Sequential:
        downsample = None
        if stride != 1 and not depthwise:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        
        if depthwise:
            layers.append(
                DepthwiseSeparableConv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels
                )
            )
        else:
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    downsample=downsample
                )
            )
            
        self.in_channels = out_channels * self.expansion
        
        for _ in range(1, blocks):
            if depthwise:
                layers.append(
                    DepthwiseSeparableConv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channels
                    )
                )
            else:
                layers.append(
                    block(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        expansion=self.expansion
                    )
                )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    

    
if __name__ == '__main__':
    config = {'input_channels': 1,
                'num_classes': 10,
                'depthwise_convs': [True, True, True, False]}
    
    model = ResNet18_Modified(config).cuda()
    summary(model, (1, 28, 28))
    
    