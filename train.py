import wandb

from itertools import chain

import torch
from torch import optim

from tqdm import tqdm

import config
from data import dataloaders
from models import models
from utils import utils


def main() : 
    # Read Options
    opt = config.readArguments(train=True)
    
    # Create Dataloader Instance
    trainDataLoader, validDataLoader = dataloaders.getDataLoaders(opt)

    # Create Model Instance
    model = models.UIDCycleGAN(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Create Wandb Instance
    if not opt.noWandb :
        wandb.init(config=opt, 
                   project=f"{opt.dataType}-Sigma-{opt.sigma}" if opt.dataType == "CBSD68" else opt.dataType)
        wandb.run.name = opt.name
        wandb.watch(model.module.netGenS2T), wandb.watch(model.module.netGenT2S) 
        wandb.watch(model.module.netDisS2T), wandb.watch(model.module.netDisT2S)
        imageSaver = utils.imageSaver(opt)
    
    # Create Optimizer Instance
    optimizerG = optim.RMSprop(chain(model.module.netGenS2T.parameters(), model.module.netGenT2S.parameters()),
                               lr=opt.lrG, 
                               alpha=opt.alpha)
    optimizerD = optim.RMSprop(chain(model.module.netDisS2T.parameters(), model.module.netDisT2S.parameters()),
                               lr=opt.lrD, 
                               alpha=opt.alpha)
    
    # Create Scheduler Instance
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, 
                                                      T_max=opt.numEpochs, 
                                                      eta_min=opt.lrG*opt.decayRate)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, 
                                                      T_max=opt.numEpochs, 
                                                      eta_min=opt.lrD*opt.decayRate)
    
    # Initialize Variables for Saving Weights
    bestLPIPS = torch.inf
    computeLPIPS = utils.LPIPS(opt)
    
    # Create AverageMeter Instance
    trainAdvG, trainAdvRealD, trainAdvFakeD = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    trainCycle, trainIdn, trainPerc, trainStyle, trainTV = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    validLPIPS, validSSIM, validPSNR = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    
    # Start Training
    for currentEpoch in range(opt.numEpochs) :
        # Create TQDM Instance
        trainBar = tqdm(trainDataLoader)
        
        # Reset AverageMeter Instance
        trainAdvG.reset(), trainAdvRealD.reset(), trainAdvFakeD.reset() 
        trainCycle.reset(), trainIdn.reset(), trainPerc.reset(), trainStyle.reset(), trainTV.reset()
            
        for data in trainBar :
            # Load Dataset and Assign Device
            source, target = data["source"], data["target"]
            source, target = models.assignDevice(opt, source, target)
            
            # Update Generator Weights
            optimizerG.zero_grad()
            lossG, S2T, T2S = model(source, target, mode="generator")
            trainAdvG.update((lossG["Adv-S2T"]+lossG["Adv-T2S"]).mean().detach().item())
            trainCycle.update((lossG["Cycle-S2T2S"]+lossG["Cycle-T2S2T"]).mean().detach().item())
            trainIdn.update((lossG["Idn-S2T"]+lossG["Idn-T2S"]).mean().detach().item())
            trainPerc.update((lossG["Perc-S2T"]+lossG["Perc-T2S"]).mean().detach().item())
            trainStyle.update((lossG["Style-S2T"]+lossG["Style-T2S"]).mean().detach().item())
            trainTV.update(lossG["TV-S2T"].mean().detach().item())
            sum(lossG.values()).mean().backward()
            optimizerG.step()
            
            # Update Discriminator Weights
            optimizerD.zero_grad()
            lossD = model(source, target, mode="discriminator")
            trainAdvRealD.update((lossD["Adv-S2T-Real"]+lossD["Adv-T2S-Real"]).mean().detach().item())
            trainAdvFakeD.update((lossD["Adv-S2T-Fake"]+lossD["Adv-T2S-Fake"]).mean().detach().item())
            sum(lossD.values()).mean().backward()
            optimizerD.step()
            
            # Show Training Status
            trainBar.set_description(desc=f"[Train] [{currentEpoch+1}/{opt.numEpochs}] < Adv-G:{trainAdvG.avg:.4f} | Adv-D-Real:{trainAdvRealD.avg:.4f} | Adv-D-Fake:{trainAdvFakeD.avg:.4f} | Cycle:{trainCycle.avg:.4f} | Idn.:{trainIdn.avg:.4f} | Perc.:{trainPerc.avg:.4f} | Style:{trainStyle.avg:.4f} | TV:{trainTV.avg:.4f} >")

        # Create TQDM Instance
        validBar = tqdm(validDataLoader)
        
        # Reset AverageMeter Instance
        validLPIPS.reset(), validSSIM.reset(), validPSNR.reset()
        
        if not opt.noWandb :
            imageSet = []

        for data in validBar :
            # Load Dataset and Assign Device
            source, target = data["source"], data["target"]
            source, target = models.assignDevice(opt, source, target)
            
            # Get Final Results
            S2T, T2S, _, _ = model(source, target, mode="inference")
            
            if not opt.noWandb :
                subImageSet = imageSaver.visualizeBatch([target, S2T, source, T2S, source-target, source-S2T, T2S-target])
                if len(imageSet) <= 108 :
                    imageSet.append(subImageSet)
            
            # Compute Metric
            validLPIPS.update(computeLPIPS(S2T, target).detach().item())
            validSSIM.update(utils.computeSSIM(S2T, target))
            validPSNR.update(utils.computePSNR(S2T, target))
            
            # Show Validation Status
            validBar.set_description(desc=f"[Valid] [{currentEpoch+1}/{opt.numEpochs}] < LPIPS:{validLPIPS.avg:.4f} | SSIM:{validSSIM.avg:.4f} | PSNR:{validPSNR.avg:.4f} >")
        
        # Upload Images
        if not opt.noWandb :
            inferenceResult = []
            for i, subImageSet in enumerate(imageSet) :
                inferenceResult.append(wandb.Image(subImageSet, caption=f"Batch-{i+1}"))
        
        # Save Weights
        if validLPIPS.avg < bestLPIPS :
            bestLPIPS = validLPIPS.avg
            utils.saveNetwork(opt, model, currentEpoch, latest=False, best=True)
        utils.saveNetwork(opt, model, currentEpoch, latest=True, best=False)
        
        # Upload Metrics
        if not opt.noWandb :
            wandb.log({"Training Cycle Loss":trainCycle.avg,
                       "Validation LPIPS":validLPIPS.avg,
                       "Validation SSIM":validSSIM.avg,
                       "Validation PSNR":validPSNR.avg,
                       "Inference Result":inferenceResult})
        
        # Update Learning Rate
        schedulerG.step(), schedulerD.step()


if __name__ == "__main__" :
    main()