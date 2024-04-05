"""
Contains PyTorch model code to ResNet50-ResNet152 models.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import data_setup


class ResNetblock(nn.Module):
    '''
        Również do opisania ładnego teraz wyjebane jajca nie mam czasu na tłumaczenie tego gówna eeeelo
    '''
    def __init__(self,in_channels,out_channels,stride=1,identity_downsample=None) -> None:
        super(ResNetblock,self).__init__()

        self.expansion=4

        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channels)
       
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(out_channels)
       
        self.conv3=nn.Conv2d(in_channels=out_channels,out_channels=out_channels*self.expansion,kernel_size=3,padding=1,stride=1)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        
        
        self.relu=nn.ReLU()

        self.identity_downsample=identity_downsample
    
    def forward(self,x):
        identity=x
        # print(f"identity shape x{identity.shape}")
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        # print(f"after conv1 shape {x.shape}")

        x=self.conv2(x)
        x=self.bn2(x)
        # print(f"after conv2 shape {x.shape}")

        # print(f"before conv3 shape {x.shape}")
        x=self.conv3(x)
        x=self.bn3(x)
        # print(f"after conv3 shape {x.shape}")
        
        if self.identity_downsample != None:
            identity=self.identity_downsample(identity)
            # print(f"identity shape after downsample {identity.shape}")

        x+=identity
        x=self.relu(x)
        # print(f"x shape {x.shape}")

        return x
    


class ResNet(nn.Module):
    '''
        Creates the ResNet50+ architecture

        Replicates the ResNet50 architecture from the https://github.com/Machmurka/UnderstandingDeepLearning/blob/main/Learn%20PyTorch%20for%20Deep%20Learning/ResNet18layers.ipynb
        See the original architecture here: # https://arxiv.org/pdf/1512.03385.pdf

          Args:
            block: ResNetblock class
            # do napisania teraz wyjebane jajca
    '''
    def __init__(self,block:ResNetblock,img_channels,num_classes,block_num:list) -> None:
        super(ResNet,self).__init__()
        
        self.in_channels=64

        self.conv1=nn.Conv2d(img_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer2=self._make_layer(block,block_num[0],64,1)
        self.layer3=self._make_layer(block,block_num[1],128,2)
        self.layer4=self._make_layer(block,block_num[2],256,2)
        self.layer5=self._make_layer(block,block_num[3],512,2)

        self.avg=nn.AvgPool2d((1,1))
        self.fc=nn.Linear(76*76*2,num_classes)


    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.avg(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        


        # print(f"shape after layer1 {x.shape}")
        # print("\n\n LAYER 2 \n\n")
        # x=self.layer2(x)
        # print(f"shape after layer2 {x.shape}")
        # print("\n\n LAYER 3 \n\n")  
        # x=self.layer3(x)
        # print(f"shape after layer3 {x.shape}")
        # print("\n\n LAYER 4 \n\n")  
        # x=self.layer4(x)
        # print(f"shape after layer4 {x.shape}")
        # print("\n\n LAYER 5 \n\n")  
        # x=self.layer5(x)
        # print(f"shape after layer5 {x.shape}")
        # x=self.avg(x)
        # print(f"shape after avg {x.shape}")
        # x=x.reshape(x.shape[0],-1)
        # print(f"shape after reshape {x.shape}")
        # x=self.fc(x)
        # print(f"output shape {x.shape}")
        
        return x
    
    def _make_layer(self,block:ResNetblock,num_blocks,out_channels,stride):
        identity_downsample=None
        layers=[]

        if stride!=1 or self.in_channels!=out_channels*4:
            identity_downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*4,1,stride,padding=0),
                nn.BatchNorm2d(out_channels*4)
            )
            # print("stride test")
        
        layers.append(block(self.in_channels,out_channels=out_channels,stride=stride,identity_downsample=identity_downsample))
        self.in_channels=out_channels*4

        for i in range(num_blocks-1):
            layers.append(block(self.in_channels,out_channels=out_channels))

        return nn.Sequential(*layers)