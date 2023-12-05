from torch import nn
from torch.nn.utils import spectral_norm


def spectralNormalization(noSpectralNorm) :
    if noSpectralNorm :
        # Return Identity Layer
        return nn.Identity()
    else :
        # Return Spectral Normalization Layer
        return spectral_norm