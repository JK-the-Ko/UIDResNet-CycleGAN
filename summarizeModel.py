import torch

import config
from models import models


def main() : 
    # Read Options
    opt = config.readArguments(train=True)

    # Create Model Instance
    model = models.UIDCycleGAN(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Create Dummy Input
    dummyInput = torch.randn((1, opt.inputDim, opt.patchSize, opt.patchSize))
    
    # Assign Device
    if opt.gpuIds != "-1" :
        dummyInput = dummyInput.cuda()
    
    # Show Model Architecture
    print(model.module.netGenS2T)
    print(model.module.netGenT2S)
    print(model.module.netDisS2T)
    print(model.module.netDisT2S)
    
    # Show Output
    outputG = model.module.netGenS2T(dummyInput)
    print("Feed-Forward Successful! (Gen. S2T)")
    print(f"Output Size (G): {outputG.size()}")
    
    # Show Output
    outputG = model.module.netGenT2S(dummyInput)
    print("Feed-Forward Successful! (Gen. T2S)")
    print(f"Output Size (G): {outputG.size()}")
    
    # Show Output
    outputD = model.module.netDisS2T(dummyInput, dummyInput)
    print("Feed-Forward Successful! (Dis. S2T)")
    print(f"Output Size (D): {outputD.size()}")        
    
    # Show Output
    outputD = model.module.netDisT2S(dummyInput, dummyInput)
    print("Feed-Forward Successful! (Dis. T2S)")
    print(f"Output Size (D): {outputD.size()}")
        

if __name__ == "__main__" :
    main()
