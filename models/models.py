import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from models.generator import AutoEncoder
from models.discriminator import Discriminator
from models.loss import GANLoss, VGGLoss, TVLoss

from utils import utils


class UIDCycleGAN(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(UIDCycleGAN, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        self.OKBLUE, self.ENDC = utils.bcolors.OKBLUE, utils.bcolors.ENDC
        
        # Create Model Instance
        self.netGenS2T, self.netGenT2S = AutoEncoder(opt), AutoEncoder(opt)
        
        # Create Discriminator Instance
        if opt.phase == "train" :
            self.netDisS2T, self.netDisT2S = Discriminator(opt), Discriminator(opt)
            self.FloatTensor = torch.cuda.FloatTensor if self.useGPU() else torch.FloatTensor
        
        # Compute Number of Parameters
        self.computeNumParameter()
        
        # Weight Initialization
        utils.fixSeed(opt.seed)
        self.initializeNetwork()
        self.loadCheckpoints()
        
        # Set Loss Function
        if opt.phase == "train" :
            self.criterionGAN = GANLoss(opt.GANMode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionVGG = VGGLoss()
            self.criterionTV = TVLoss()

    def forward(self, source, target, mode) :
        # Feed-Forward Process
        if mode == "generator" :
            loss, maskedImage, outputImage = self.computeGeneratorLoss(source, target)
            return loss, maskedImage, outputImage
        elif mode == "discriminator" :
            loss = self.computeDiscriminatorLoss(source, target)
            return loss
        elif mode == "inference" :
            self.netGenS2T.eval(), self.netGenT2S.eval() 
            with torch.no_grad() :
                S2T, T2S = self.netGenS2T(source), self.netGenT2S(target)
                S2T2S, T2S2T = self.netGenT2S(S2T), self.netGenS2T(T2S)
            return S2T, T2S, S2T2S, T2S2T
        else :
            raise ValueError(f"{mode} is not supported")

    def loadCheckpoints(self) :
        # Load Pre-Trained Weights
        if self.opt.phase == "test" :
            saveType = self.opt.saveType
            self.netGenS2T = utils.loadNetwork(self.netGenS2T, "G-S2T", saveType, self.opt)
            self.netGenT2S = utils.loadNetwork(self.netGenT2S, "G-T2S", saveType, self.opt)
        
    ############################################################################
    # Private helper methods
    ############################################################################

    def computeNumParameter(self) :
        if self.opt.phase == "train" :
            networkList = [self.netGenS2T, self.netGenT2S, self.netDisS2T, self.netDisT2S]
        else :
            networkList = [self.netGenS2T, self.netGenT2S]
        print(f"{self.OKBLUE}UIDCycleGAN{self.ENDC}: Now Computing Model Parameters.")
        for network in networkList :
            numParameter = 0
            for _, module in network.named_modules() :
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) :
                    numParameter += sum([p.data.nelement() for p in module.parameters()])
            print(f"{self.OKBLUE}UIDCycleGAN{self.ENDC}: {utils.bcolors.OKGREEN}[{network.__class__.__name__}]{self.ENDC} Total params : {numParameter:,}.")
        print(f"{self.OKBLUE}UIDCycleGAN{self.ENDC}: Finished Computing Model Parameters.")

    def initializeNetwork(self) :
        if self.opt.phase=="train" :
            def init_weights(m, gain=0.02) :
                className = m.__class__.__name__
                # Initialize Convolution Weights
                if hasattr(m, "weight") and className.find("Conv") != -1 :
                    if self.opt.initType == "normal" :
                        init.normal_(m.weight.data, 0.0, gain)
                    elif self.opt.initType == "xavier" :
                        init.xavier_normal_(m.weight.data, gain=gain)
                    elif self.opt.initType == "xavier_uniform" :
                        init.xavier_uniform_(m.weight.data, gain=gain)
                    elif self.opt.initType == "kaiming" :
                        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                    elif self.opt.initType == "orthogonal" :
                        init.orthogonal_(m.weight.data, gain=gain)
                    elif self.opt.initType == "none" :
                        m.reset_parameters()
                    else:
                        raise NotImplementedError(f"{self.opt.initType} method is not supported")
                    if hasattr(m, "bias") and m.bias is not None :
                        init.constant_(m.bias.data, 0.0)

            # Create List Instance for Adding Network
            if self.opt.phase == "train" :
                networkList = [self.netGenS2T, self.netGenT2S, self.netDisS2T, self.netDisT2S]
            else :
                networkList = [self.netGenS2T, self.netGenT2S]
            
            #I Initialize Network Weights
            for network in networkList :
                network.apply(init_weights)

    def computeGeneratorLoss(self, source, target) :
        # Create Dictionary Instance for Adding Loss
        lossG = {}
        
        # Get Inference Result
        S2T, T2S = self.netGenS2T(source), self.netGenT2S(target)
        S2T2S, T2S2T = self.netGenT2S(S2T), self.netGenS2T(T2S)
        predS2T, predT2S = self.discriminate(S2T, T2S, source, target)
        
        # Compute Adversarial Loss
        lossG["Adv-S2T"] = self.criterionGAN(predS2T[0], True, forDiscriminator=False)*self.opt.lambdaAdv
        lossG["Adv-T2S"] = self.criterionGAN(predT2S[0], True, forDiscriminator=False)*self.opt.lambdaAdv
        
        # Compute Cycle Loss
        lossG["Cycle-S2T2S"] = F.l1_loss(S2T2S, source)*self.opt.lambdaCycle
        lossG["Cycle-T2S2T"] = F.l1_loss(T2S2T, target)*self.opt.lambdaCycle 
        
        # Compute Identity Loss
        lossG["Idn-S2T"] = F.l1_loss(S2T, source)*self.opt.lambdaIdn
        lossG["Idn-T2S"] = F.l1_loss(T2S, target)*self.opt.lambdaIdn
        
        # Compute Perceptual and Style Loss
        percS2T, styleS2T = self.criterionVGG(S2T, source)
        percT2S, styleT2S = self.criterionVGG(T2S, target)
        lossG["Perc-S2T"] = percS2T*self.opt.lambdaPerc
        lossG["Perc-T2S"] = percT2S*self.opt.lambdaPerc
        lossG["Style-S2T"] = styleS2T*self.opt.lambdaStyle
        lossG["Style-T2S"] = styleT2S*self.opt.lambdaStyle
        
        # Compute Total Variation Loss
        lossG["TV-S2T"] = self.criterionTV(S2T)*self.opt.lambdaTV
        
        return lossG, S2T, T2S

    def computeDiscriminatorLoss(self, source, target) :
        # Create Dictionary Instance for Adding Loss
        lossD = {}
        
        # Fix Generator Weights Gradient
        with torch.no_grad() :
            S2T, T2S = self.netGenS2T(source), self.netGenT2S(target)
            S2T, T2S = S2T.detach(), T2S.detach()
            
        # Get Inference Result
        predS2T, predT2S = self.discriminate(S2T, T2S, source, target)
        
        # Compute Adversarial Loss
        lossD["Adv-S2T-Fake"] = self.criterionGAN(predS2T[0], False, forDiscriminator=True)
        lossD["Adv-S2T-Real"] = self.criterionGAN(predS2T[1], True, forDiscriminator=True)
        lossD["Adv-T2S-Fake"] = self.criterionGAN(predT2S[0], False, forDiscriminator=True)
        lossD["Adv-T2S-Real"] = self.criterionGAN(predT2S[1], True, forDiscriminator=True)
        
        return lossD
    
    def discriminate(self, S2T, T2S, source, target) :
        # Get Image Inference Result
        disS2TOut = self.netDisS2T(S2T, target)
        disT2SOut = self.netDisT2S(T2S, source)
        
        # Divide Inference Results
        predS2TFake, predS2TReal = self.dividePrediction(disS2TOut)
        predT2SFake, predT2SReal = self.dividePrediction(disT2SOut)
        
        return (predS2TFake, predS2TReal), (predT2SFake, predT2SReal)
    
    def dividePrediction(self, pred) :
        if isinstance(pred, list) :
            fake, real = [], []
            for subPred in pred :
                fake.append([tensor[:tensor.size(0)//2] for tensor in subPred])
                real.append([tensor[tensor.size(0)//2:] for tensor in subPred])
        else :
            fake = pred[:pred.size(0)//2]
            real = pred[pred.size(0)//2:]
            
        return fake, real
    
    def useGPU(self) :
        return self.opt.gpuIds != "-1"


def assignOnMultiGpus(opt, model):
    if opt.gpuIds != "-1" :
        gpus = list(map(int, opt.gpuIds.split(",")))
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpuIds.split(",")) == 0 or opt.batchSize % len(opt.gpuIds.split(",")) == 0
    
    return model


def assignDevice(opt, source, target) :
    # Assign Device
    if opt.gpuIds != "-1" :
        source, target = source.cuda(), target.cuda()
    
    return source, target