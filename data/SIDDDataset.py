from os import listdir
from os.path import join

import random

from PIL import Image

import numpy as np

from natsort import natsorted

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SIDDDataset(Dataset) :
    def __init__(self, opt, forMetrics, noRotation) :
        # Inheritance
        super(SIDDDataset, self).__init__()
        
        # Initialize Variables
        self.opt = opt
        self.forMetrics = forMetrics
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
            source = Image.open(join(self.sourceDataset[1], random.choice(self.sourceDataset[0]))).convert("RGB")
            target = Image.open(join(self.targetDataset[1], self.targetDataset[0][index])).convert("RGB")

            # Transform Data
            source, target = self.transforms(source, target)
            
            return {"source":source, "target":target}

    def __len__(self):
        return len(self.targetDataset[0])

    def getPathList(self) :
        # Get Absolute Parent Path
        if self.forMetrics :
            sourcePath = join(self.opt.dataRoot, "SIDD", "test", "noisy")
            targetPath = join(self.opt.dataRoot, "SIDD", "test", "clean")
        else :
            sourcePath = join(self.opt.dataRoot, "SIDD", "train", "patch", "noisy")
            targetPath = join(self.opt.dataRoot, "SIDD", "train", "patch", "clean")
        
        # Create List Instance for Adding Dataset Path
        sourcePathList = listdir(sourcePath)
        targetPathList = listdir(targetPath)
        
        # Create List Instance for Adding File Name
        sourceNameList = [imageName for imageName in sourcePathList if ".png" in imageName or ".PNG" in imageName]
        targetNameList = [imageName for imageName in targetPathList if ".png" in imageName or ".PNG" in imageName]
        
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
                
        # Convert to PyTorch Tensor
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
            
        # Apply Normalization
        source = TF.normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        target = TF.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return source, target