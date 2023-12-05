from os.path import join

from tqdm import tqdm

import config
from data import dataloaders
from models import models
from utils import utils


def main() : 
    # Read Options
    opt = config.readArguments(train=False)
    opt.gpuIds = "0"
    opt.batchSize = 1

    _, validDataloader = dataloaders.getDataLoaders(opt, noRotation=True)
    
    # Create Model Instance
    model = models.UIDCycleGAN(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Make Directory Path for Saving Results
    n2cSavePath = join("results", opt.name, opt.dataType, "Noise2Clean")
    c2nSavePath = join("results", opt.name, opt.dataType, "Clean2Noise")
    n2c2nSavePath = join("results", opt.name, opt.dataType, "Noise2Clean2Noise")
    c2n2cSavePath = join("results", opt.name, opt.dataType, "Clean2Noise2Clean")
    noiseSavePath = join("results", opt.name, opt.dataType, "TargetNoise")
    
    # Generate Directory
    utils.mkdirs(n2cSavePath)
    utils.mkdirs(c2nSavePath)
    utils.mkdirs(n2c2nSavePath)
    utils.mkdirs(c2n2cSavePath)
    utils.mkdirs(noiseSavePath)
    
    # Inference
    validBar = tqdm(validDataloader) 
    for data in validBar :
        source, target, name = data["source"], data["target"], data["name"][0]
        source, target = models.assignDevice(opt, source, target)
        
        source, target = models.assignDevice(opt, source, target)
        S2T, T2S, S2T2S, T2S2T = model(source, target, mode="inference")
        
        utils.saveImage(S2T, n2cSavePath, name)
        utils.saveImage(T2S, c2nSavePath, name)
        utils.saveImage(S2T2S, n2c2nSavePath, name)
        utils.saveImage(T2S2T, c2n2cSavePath, name)
        utils.saveNoise(target, source, noiseSavePath, name)
            
        validBar.set_description(desc=f"[Test] < Saving Results! >")
        

if __name__ == "__main__" :
    main()