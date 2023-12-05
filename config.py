import argparse
from utils import utils


def readArguments(train=True) :
    parser = argparse.ArgumentParser()
    parser = addAllArguments(parser, train)
    parser.add_argument("--phase", type=str, default="train")
    opt=parser.parse_args()
    opt.phase="train" if train else "test"
    utils.fixSeed(opt.seed)
    
    return opt


def addAllArguments(parser, train):
    # General Options
    parser.add_argument("--name", type=str, default="UIDResNet-CycleGAN-Sigma-15", help="name of the experiment. It decides where to store samples and models")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpuIds", type=str, default="0,1", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpointsDir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--batchSize", type=int, default=8, help="input batch size")
    parser.add_argument("--dataRoot", type=str, default="./dataset", help="path to dataset root")
    parser.add_argument("--dataType", type=str, default="CBSD68", help="this option indicates which dataset should be loaded")
    parser.add_argument("--sigma", type=int, default=15, help="standard deviation of gaussian noise")
    parser.add_argument("--patchSize", type=int, default=256, help="input image size")
    parser.add_argument("--numWorkers", type=int, default=20, help="num_workers argument for dataloader")

    # For Generator
    parser.add_argument("--channelsG", type=int, default=64, help="# of generator filters in first conv layer in generator")
    parser.add_argument("--inputDim", type=int, default=3, help="dimension of the input data")

    # For Training
    if train :
        parser.add_argument("--noWandb", action="store_true", help="if specified, do not use wandb library")
        parser.add_argument("--initType", type=str, default="normal", help="selects weight initialization type")
        parser.add_argument("--numEpochs", type=int, default=200, help="number of iterations to train")
        parser.add_argument("--alpha", type=float, default=0.99, help="alpha term of RMSprop")
        parser.add_argument("--lrG", type=float, default=1e-4, help="G learning rate, default=0.0001")
        parser.add_argument("--lrD", type=float, default=4e-4, help="D learning rate, default=0.0004")
        parser.add_argument("--decayRate", type=float, default=1e-2, help="learning rate decay")
        parser.add_argument("--GANMode", type=str, default="ls", help="{vanilla | ls | hinge}")
        parser.add_argument("--lambdaAdv", type=float, default=1, help="weight for adversarial loss")
        parser.add_argument("--lambdaCycle", type=float, default=10, help="weight for cycle loss")
        parser.add_argument("--lambdaIdn", type=float, default=1, help="weight for identity loss")
        parser.add_argument("--lambdaPerc", type=float, default=1e-1, help="weight for perceptual loss")
        parser.add_argument("--lambdaStyle", type=float, default=1e-1, help="weight for style loss")
        parser.add_argument("--lambdaTV", type=float, default=1e-2, help="weight for tv loss")
        
        # For discriminator
        parser.add_argument("--channelsD", type=int, default=64, help="# of discriminator filters in first conv layer in discriminator")
        parser.add_argument("--noSpectralNormD", action="store_true", help="if specified, do not use spectral normalization")

    # For Inference
    else:
        parser.add_argument("--saveType", type=str, default="best", help="save type for loading model")

    return parser