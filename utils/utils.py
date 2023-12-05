from os import makedirs
from os.path import join, exists

import random

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import lpips

import torch
from torch import nn


class bcolors :
    # Set Color of Fonts
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fixSeed(seed) :
    # Fix Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class AverageMeter(object) :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) :
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


def saveNetwork(opt, model, numEpoch, latest=False, best=False) :
    # Get Save Directory
    path = join(opt.checkpointsDir, opt.name, opt.dataType, "models")
    
    # Generate Directory
    mkdirs(path)
    
    # Save Pre-Trained Models
    if latest :
        torch.save(model.module.netGenS2T.state_dict(), f"{path}/iter-{numEpoch}-G-S2T.pth")
        torch.save(model.module.netGenT2S.state_dict(), f"{path}/iter-{numEpoch}-G-T2S.pth")
        torch.save(model.module.netGenS2T.state_dict(), f"{path}/latest-G-S2T.pth")
        torch.save(model.module.netGenT2S.state_dict(), f"{path}/latest-G-T2S.pth")
    elif best :
        torch.save(model.module.netGenS2T.state_dict(), f"{path}/best-G-S2T.pth")
        torch.save(model.module.netGenT2S.state_dict(), f"{path}/best-G-T2S.pth")


def loadNetwork(network, networkType, saveType, opt) :
    # Get Path Directory
    saveFileName = f"{saveType}-{networkType}.pth"
    savePath = join(opt.checkpointsDir, opt.name, opt.dataType, "models", saveFileName)
    
    # Load Network
    weights = torch.load(savePath)
    network.load_state_dict(weights)
    
    return network


class LPIPS(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(LPIPS, self).__init__()
        
        # Create LPIPS Instance
        self.model = lpips.LPIPS(net="alex")
        
        # Assign Device
        if opt.gpuIds != "-1" :
            self.model = self.model.cuda()

    def forward(self, fakeImage, realImage) :
        # Compute LPIPS
        dist = self.model.forward(fakeImage, realImage)
    
        return dist.mean()


def computePSNR(fakeImage, realImage) :
    # Compute PSNR
    psnr = 0
    for i in range(len(fakeImage)) :
        psnr += (peak_signal_noise_ratio(tensorToImage(realImage[i].detach()), 
                                         tensorToImage(fakeImage[i].detach()))/len(fakeImage))

    return psnr


def computeSSIM(fakeImage, realImage) :
    # Compute SSIM
    ssim = 0
    for i in range(len(fakeImage)) :
        ssim += (structural_similarity(tensorToImage(realImage[i].detach()), 
                                       tensorToImage(fakeImage[i].detach()), 
                                       channel_axis=2, 
                                       full=True)[0]/len(fakeImage))

    return ssim


class imageSaver :
    def __init__(self, opt) :
        self.opt = opt

    def visualizeBatch(self, resultList) :
        imageList = []
        for i in range(len(resultList[0])) :
            subImageList = []
            for subImage in resultList :
                subImageList.append(tensorToImage(subImage[i]))
            imageSet = np.hstack(subImageList)
            imageList.append(imageSet)
        return np.vstack(imageList)


def tensorToImage(tensor, affine=True) :
    if affine :
        imageTensor = (tensor+1)/2
    else :
        imageTensor = tensor
    imageTensor = torch.clamp(imageTensor*255, 0, 255).detach().cpu().tolist()
    imageNumpy = np.array(imageTensor, dtype="uint8").transpose(1, 2, 0)
    return imageNumpy


def saveImage(tensor, imageSavePath, name, affine=True) :
    image = tensorToImage(tensor[0,:,:,:], affine)
    cv2.imwrite(join(imageSavePath, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def saveNoise(clean, noise, imageSavePath, name, affine=True) :
    clean = tensorToImage(clean[0,:,:,:], affine)
    noise = tensorToImage(noise[0,:,:,:], affine)
    residual = noise-clean
    cv2.imwrite(join(imageSavePath, name), cv2.cvtColor(residual, cv2.COLOR_RGB2BGR))


def mkdirs(path) :
    # Make Directory
    if not exists(path) :
        makedirs(path)