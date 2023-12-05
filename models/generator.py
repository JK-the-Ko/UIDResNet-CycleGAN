import torch
from torch import nn
import torch.nn.functional as F

from models.architecture import Conv2dModule, ResidualBlock, Upsample


class AutoEncoder(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(AutoEncoder, self).__init__()
        
        # Initialize Variable
        inputDim = opt.inputDim
        channels = opt.channelsG
        
        # Create Encoder Component Instances
        self.EB0 = Conv2dModule(inputDim, channels, 7, 1, 3, 1, True, True)
        self.EB1 = ResidualBlock(channels, channels*2, 3, 1, 1, True)
        self.EB2 = ResidualBlock(channels*2, channels*4, 3, 1, 1, True)
        self.EB3 = ResidualBlock(channels*4, channels*4, 3, 1, 1, True)
        
        # Create Decoder Component Instances
        self.DB0 = Upsample(channels*4, channels*4, channels*4)
        self.DB1 = Upsample(channels*4, channels*2, channels*2)
        self.DB2 = Upsample(channels*2, channels, channels)
        self.DB3 = Conv2dModule(channels, inputDim, 7, 1, 3, 1, True, False)
        
    def forward(self, input) :
        EB0 = self.EB0(input)
        EB1 = self.EB1(EB0)
        EB2 = self.EB2(EB1)
        EB3 = self.EB3(EB2)
        
        DB0 = self.DB0(EB3, EB2)
        DB1 = self.DB1(DB0, EB1)
        DB2 = self.DB2(DB1, EB0)
        DB3 = self.DB3(DB2)
        
        return torch.tanh(DB3)