import torch
from torch import nn
import torch.nn.functional as F

from models.architecture import VGG19


class GANLoss(nn.Module) :
    def __init__(self, GANMode, targetRealLabel=1, targetFakeLabel=0.0, tensor=torch.FloatTensor, opt=None) :
        # Inheritance
        super(GANLoss, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        self.realLabel = targetRealLabel
        self.fakeLabel = targetFakeLabel
        self.realLabelTensor, self.fakeLabelTensor = None, None
        self.zeroTensor = None
        self.Tensor = tensor
        self.GANMode = GANMode
        
        # Select GAN Loss Mode
        GANModeList = ["vanilla", "w", "ls", "hinge"]
        if GANMode not in GANModeList :
            raise ValueError(f"{GANMode} not supported")

    def computeTargetTensor(self, input, targetIsReal) :
        if targetIsReal :
            if self.realLabelTensor is None :
                # Compute Real Label Tensor
                self.realLabelTensor = self.Tensor(1).fill_(self.realLabel)
                self.realLabelTensor.requires_grad_(False)
                
            return self.realLabelTensor.expand_as(input)
        else :
            if self.fakeLabelTensor is None :
                # Compute Fake Label Tensor
                self.fakeLabelTensor = self.Tensor(1).fill_(self.fakeLabel)
                self.fakeLabelTensor.requires_grad_(False)
                
            return self.fakeLabelTensor.expand_as(input)

    def computeZeroTensor(self, input) :
        if self.zeroTensor is None :
            # Compute Zero Tensor
            self.zeroTensor = self.Tensor(1).fill_(0)
            self.zeroTensor.requires_grad_(False)
            
        return self.zeroTensor.expand_as(input)

    def computeLoss(self, input, targetIsReal, forDiscriminator=True) :
        if self.GANMode == "vanilla" :
            # Compute Binary Cross Entropy Loss
            targetTensor = self.computeTargetTensor(input, targetIsReal)
            loss = F.binary_cross_entropy_with_logits(input, targetTensor)
            return loss
        elif self.GANMode == "ls" :
            # Compute Least Square Loss
            targetTensor = self.computeTargetTensor(input, targetIsReal)
            loss = F.mse_loss(input, targetTensor)
            return loss
        elif self.GANMode == "hinge" :
            # Compute Hinge GAN Loss
            if forDiscriminator :
                if targetIsReal :
                    minVal = torch.min(input-1, self.computeZeroTensor(input))
                    loss = -torch.mean(minVal)
                else :
                    minVal = torch.min(-input-1, self.computeZeroTensor(input))
                    loss = -torch.mean(minVal)
            else :
                assert targetIsReal, "Generator must aim for real"
                loss = -torch.mean(input)
            return loss
        else :
            # Compute Wasserstein GAN Loss
            if targetIsReal :
                loss = -input.mean()
                return loss
            else :
                loss = input.mean()
                return loss

    def forward(self, input, targetIsReal, forDiscriminator=True) :
        # Multiscale Discriminator
        if isinstance(input, list) :
            loss = 0
            for subPred in input :
                if isinstance(subPred, list) :
                    subPred = subPred[-1]
                lossTensor = self.computeLoss(subPred, targetIsReal, forDiscriminator)
                B = 1 if len(lossTensor.size()) == 0 else lossTensor.size(0)
                newLoss = torch.mean(lossTensor.view(B, -1), dim=1)
                loss += newLoss
            return loss/len(input)
        else :
            loss = self.computeLoss(input, targetIsReal, forDiscriminator)
            return loss


class VGGLoss(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(VGGLoss, self).__init__()
        
        # Create VGG Instance
        self.VGG = VGG19()

    def forward(self, fakeImage, realImage) :
        featureFakeImage, featureRealImage = self.VGG(fakeImage), self.VGG(realImage)
        
        perceptualLoss, styleLoss = 0, 0
        for i in range(len(featureFakeImage)) :
            perceptualLoss += F.l1_loss(featureFakeImage[i], featureRealImage[i].detach())
            styleLoss += F.l1_loss(gramMatrix(featureFakeImage[i]), gramMatrix(featureRealImage[i]).detach())
            
        return perceptualLoss, styleLoss


def gramMatrix(input) :
    # Get Shape of Input Tensor
    n, c, h, w = input.size() 

    # Compute Gram Matrix
    features = input.view(n*c, h*w)
    G = torch.mm(features, features.t())
    
    return G.div(n*c*h*w)


class TVLoss(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(TVLoss, self).__init__()

    def forward(self, input) :
        # Get Shape of Input Tensor
        n, _, h, w = input.size()
        
        # Compute Parameters for TV Loss
        countH = (h-1)*w
        countW = (w-1)*h
        
        # Compute TV Loss
        tvH = torch.pow((input[:, :, 1:, :] - input[:, :, :h-1, :]), 2).sum()
        tvW = torch.pow((input[:, :, :, 1:] - input[:, :, :, :w-1]), 2).sum()
        
        return 2*(tvH/countH + tvW/countW)/n