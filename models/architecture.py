import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class Conv2dModule(nn.Module) :
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding, dilation, useBias, useAct) :
        # Inheritance
        super(Conv2dModule, self).__init__()
        
        # Create Convolutional Layer Instance
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation, bias=useBias, padding_mode="reflect")
            
        # Create Activation Function Instance
        if useAct :
            self.act = nn.LeakyReLU(0.2)
            
    def forward(self, input) :
        output = self.conv(input)
        
        if hasattr(self, "act") :
            output = self.act(output)
            
        return output


class ResidualBlock(nn.Module) :
    def __init__(self, inChannels, outChannels, kernelSize, padding, dilation, isDown) :
        # Inheritance
        super(ResidualBlock, self).__init__()
        
        # Create Convolutional Module Instance
        self.conv0 = Conv2dModule(inChannels, outChannels, kernelSize, 2 if isDown else 1, padding, dilation, True, True)
        self.conv1 = Conv2dModule(outChannels, outChannels, kernelSize, 1, padding, dilation, True, False)
        self.bottleneck = Conv2dModule(inChannels, outChannels, 1, 2 if isDown else 1, 0, 1, False, False)
        
        # Create Activation Function Instance
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, input) :
        output = self.conv0(input)
        output = self.conv1(output)
        output = self.act(output+self.bottleneck(input))

        return output


class Upsample(nn.Module) :
    def __init__(self, inChannels, skChannels, outChannels) :
        # Inheritance
        super(Upsample, self).__init__()
        
        # Create Convolutional Module Instance
        self.up = nn.ConvTranspose2d(inChannels, inChannels, 4, 2, 1)
        
        # Create Residual Block Instance
        self.RB = ResidualBlock(inChannels+skChannels, outChannels, 3, 1, 1, False)
        
    def forward(self, input, skipConnection) :
        output = torch.cat([self.up(input), skipConnection], dim=1)
        output = self.RB(output)
        
        return output
        

class VGG19(nn.Module):
    def __init__(self):
        # Inheritance
        super(VGG19, self).__init__()
        
        # Load Pretrained Vgg Network
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Create List Instance for Adding Layers
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # Add Layers
        for layer in range(2):
            self.slice1.add_module(str(layer), model[layer])
        for layer in range(2, 7):
            self.slice2.add_module(str(layer), model[layer])
        for layer in range(7, 12):
            self.slice3.add_module(str(layer), model[layer])
        for layer in range(12, 21):
            self.slice4.add_module(str(layer), model[layer])
        for layer in range(21, 30):
            self.slice5.add_module(str(layer), model[layer])
        
        # Fix Gradient Flow
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        hRelu1 = self.slice1(input)
        hRelu2 = self.slice2(hRelu1)
        hRelu3 = self.slice3(hRelu2)
        hRelu4 = self.slice4(hRelu3)
        hRelu5 = self.slice5(hRelu4)

        return [hRelu1, hRelu2, hRelu3, hRelu4, hRelu5]