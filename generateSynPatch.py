from os import listdir
from os.path import join

import cv2

from tqdm import tqdm

import config


def main() :
    # Read Options
    opt = config.readArguments(train=True)
    
    # Make Directory Path for Saving Patches
    sourcePath = join(opt.dataRoot, "DIV2K", "original", "noisy")
    targetPath = join(opt.dataRoot, "DIV2K", "original", "clean")
    
    # Set Patch Size and Stride
    patchSize, stride = 300, 300
    
    # Generate Patch
    for imageName in listdir(sourcePath) :
        imagePath = join(sourcePath, imageName)
        
        image = cv2.imread(imagePath)
        height, width = image.shape[:2]
        
        patchNum = 0
        with tqdm(total=(height//stride)*(width//stride)) as pBar :
            for i in range(height//stride) :
                for j in range(width//stride) :
                    patch = image[i*stride:i*stride+patchSize, j*stride:j*stride+patchSize, :]
                    cv2.imwrite(join(opt.dataRoot, "DIV2K", "patch", "noisy", imageName.replace(".png", f"-{patchNum}.png")), patch)
                    patchNum += 1
                    
                    pBar.set_description(f"Noisy Dataset: {imageName}"), pBar.update()

    # Generate Patch
    for imageName in listdir(targetPath) :
        imagePath = join(targetPath, imageName)
        
        image = cv2.imread(imagePath)
        height, width = image.shape[:2]
        
        patchNum = 0
        with tqdm(total=(height//stride)*(width//stride)) as pBar :
            for i in range(height//stride) :
                for j in range(width//stride) :
                    patch = image[i*stride:i*stride+patchSize, j*stride:j*stride+patchSize, :]
                    cv2.imwrite(join(opt.dataRoot, "DIV2K", "patch", "clean", imageName.replace(".png", f"-{patchNum}.png")), patch)
                    patchNum += 1
                    
                    pBar.set_description(f"Clean Dataset: {imageName}"), pBar.update()


if __name__ == "__main__" :
    main()
        