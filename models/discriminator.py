import torch
from torch import nn

from models.normalization import spectralNormalization


class Discriminator(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Discriminator, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        inputDim = opt.inputDim
        channels = opt.channelsD
        
        # Create Spectral Normalization Instance
        spectralNorm = spectralNormalization(opt.noSpectralNormD)
        
        # Create Convolutional Layer Instance
        self.conv0 = spectralNorm(nn.Conv2d(inputDim, channels, kernel_size=4, stride=2, padding=1))
        self.conv1 = spectralNorm(nn.Conv2d(channels, channels*2, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectralNorm(nn.Conv2d(channels*2, channels*4, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectralNorm(nn.Conv2d(channels*4, channels*8, kernel_size=4, stride=1, padding=1))
        self.conv4 = spectralNorm(nn.Conv2d(channels*8, 1, kernel_size=4, stride=1, padding=1))
        
        # Create Activation Function Instance
        self.act = nn.LeakyReLU(0.2) 

    def forward(self, fakeImage, realImage) :
        # Concatenate Input
        input = torch.cat([fakeImage, realImage], dim=0)
        
        # Feed-Forward
        output = self.act(self.conv0(input))
        output = self.act(self.conv1(output))
        output = self.act(self.conv2(output))
        output = self.act(self.conv3(output))
        output = self.conv4(output)
        
        return output