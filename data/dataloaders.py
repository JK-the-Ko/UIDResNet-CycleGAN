import importlib
from torch.utils.data import DataLoader


def selectDatasetName(mode) :
    # Select Dataset Name
    if mode == "CBSD68" :
        return "CBSD68Dataset"
    elif mode == "SIDD" :
        return "SIDDDataset"
    else :
        raise NotImplementedError(f"Dataset {mode} is not implemented!")


def getDataLoaders(opt, noRotation=False) :
    # Select Dataset Name
    datasetName = selectDatasetName(opt.dataType)
    
    # Import Python Code
    fileName = importlib.import_module(f"data.{datasetName}")
    
    # Create Dataset Instance
    trainDataset = fileName.__dict__[datasetName](opt, False, noRotation)
    validDataset = fileName.__dict__[datasetName](opt, True, noRotation)
    
    # Train PyTorch DataLoader Instance
    trainDataLoader = DataLoader(trainDataset, 
                                 batch_size=opt.batchSize, 
                                 shuffle=True, 
                                 drop_last=True, 
                                 num_workers=opt.numWorkers)
    validDataLoader = DataLoader(validDataset, 
                                 batch_size=opt.batchSize,
                                 shuffle=False, 
                                 drop_last=False, 
                                 num_workers=opt.numWorkers)

    return trainDataLoader, validDataLoader