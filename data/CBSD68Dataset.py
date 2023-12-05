from os import listdir
from os.path import join

import random

from PIL import Image

import numpy as np

from natsort import natsorted

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CBSD68Dataset(Dataset) :
    def __init__(self, opt, forMetrics, noRotation=False) :
        # Inheritance
        super(CBSD68Dataset, self).__init__()
        
        # Initialize Variables
        self.opt = opt
        self.forMetrics = forMetrics
        self.noRotation = noRotation
        self.sourceDataset, self.targetDataset = self.getPathList()

    def __getitem__(self, index):
        if self.forMetrics :
            # Load Data
            source = Image.open(join(self.sourceDataset[1], self.sourceDataset[0][index])).convert("RGB")
            target = Image.open(join(self.targetDataset[1], self.targetDataset[0][index])).convert("RGB")

            # Transform Data
            source, target = self.transforms(source, target)
            
            return {"source":source, "target":target, "name":self.sourceDataset[0][index]}
        else :
            # Load Data
            source = Image.open(join(self.sourceDataset[1], self.sourceDataset[0][index])).convert("RGB")
            target = Image.open(join(self.targetDataset[1], random.choice(self.targetDataset[0]))).convert("RGB")

            # Transform Data
            source, target = self.transforms(source, target)
            
            return {"source":source, "target":target}

    def __len__(self):
        return len(self.sourceDataset[0])

    def getPathList(self) :
        # Get Absolute Parent Path
        if self.forMetrics :
            sourcePath = join(self.opt.dataRoot, "CBSD68", f"noisy-sigma-{self.opt.sigma}")
            targetPath = join(self.opt.dataRoot, "CBSD68", "clean")
        else :
            sourcePath = join(self.opt.dataRoot, "DIV2K", "patch", "noisy")
            targetPath = join(self.opt.dataRoot, "DIV2K", "patch", "clean")
        
        # Create List Instance for Adding Dataset Path
        sourcePathList = listdir(sourcePath)
        targetPathList = listdir(targetPath)
        
        # Create List Instance for Adding File Name
        sourceNameList = [imageName for imageName in sourcePathList if ".png" in imageName]
        targetNameList = [imageName for imageName in targetPathList if ".png" in imageName]
        
        # Sort List Instance
        sourceNameList = natsorted(sourceNameList)
        targetNameList = natsorted(targetNameList)
        
        return (sourceNameList, sourcePath), (targetNameList, targetPath)
    
    def transforms(self, source, target) :
        if not self.forMetrics :
            # Apply Random Crop
            width, height = source.size
            cropX = random.randint(0, np.maximum(0, width-self.opt.patchSize)-1)
            cropY = random.randint(0, np.maximum(0, height-self.opt.patchSize)-1)
            source = source.crop((cropX, cropY, cropX+self.opt.patchSize, cropY+self.opt.patchSize))
            
            width, height = target.size
            cropX = random.randint(0, np.maximum(0, width-self.opt.patchSize)-1)
            cropY = random.randint(0, np.maximum(0, height-self.opt.patchSize)-1)
            target = target.crop((cropX, cropY, cropX+self.opt.patchSize, cropY+self.opt.patchSize))
                
            # Apply Random Horizontal Flip
            if random.random() < 0.5 :
                source = TF.hflip(source)
                target = TF.hflip(target)
        
        else :
            # Resize Image
            width, height = source.size
            if width > height :
                source = TF.resize(source, (320, 480), TF.InterpolationMode.NEAREST)
                target = TF.resize(target, (320, 480), TF.InterpolationMode.NEAREST)
            else :
                if self.noRotation :
                    source = TF.resize(source, (480, 320), TF.InterpolationMode.NEAREST)
                    target = TF.resize(target, (480, 320), TF.InterpolationMode.NEAREST)
                else :
                    source = TF.rotate(TF.resize(source, (480, 320), TF.InterpolationMode.NEAREST), 90, expand=True)
                    target = TF.rotate(TF.resize(target, (480, 320), TF.InterpolationMode.NEAREST), 90, expand=True)
                    
        # Convert to PyTorch Tensor
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        
        # Add Gaussian Noise
        if not self.forMetrics :
            noise = torch.normal(torch.zeros(source.size()), self.opt.sigma/255.0)
            source = source + noise
            source = torch.clamp(source, 0, 1)  
            
        # Apply Normalization
        source = TF.normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        target = TF.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return source, target