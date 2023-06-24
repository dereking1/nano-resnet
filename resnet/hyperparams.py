
import numpy as np

class Resnet_Hyperparams():
    def __init__(self, train_loader):        
        self.iterations = len(train_loader)
        self.epochs =  64000 // self.iterations                                             # Authors cite 64k iterations
        self.lr = 0.1                                                    # authors cite 0.1
        self.momentum = 0.9                                              # authors cite 0.9
        self.weight_decay = 0.0001                                       # authors cite 0.0001
        self.milestones = [32000 // self.iterations, 
                           48000 // self.iterations]                     # authors cite divide it by 10 at 32k and 48k iterations
        self.lr_multiplier = 1/10                                           # ^