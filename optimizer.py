from sched import scheduler
import torch
import torch.nn as nn
import torch_optimizer 
from scheduler import WarmupCosineLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR

class Optimizer:
    def __init__(
        self,
        net : torch.nn.Module, 
        myhparams: list, 
        ):
        
        if myhparams.optimizer == "sgd":
            self.optimizers = torch.optim.SGD(
                filter(lambda x: x.requires_grad, net.parameters()),
                lr=myhparams.learning_rate,
                weight_decay=myhparams.weight_decay,
                momentum=myhparams.momentum,
                nesterov=True
            )
        elif myhparams.optimizer == "adam":
            self.optimizers = torch.optim.Adam(
                filter(lambda x: x.requires_grad, net.parameters()),
                lr=myhparams.learning_rate,
                weight_decay=myhparams.weight_decay
            )
        elif myhparams.optimizer== "adagrad":
            self.optimizers = torch.optim.Adagrad(
                filter(lambda x: x.requires_grad, net.parameters()),
                lr=myhparams.learning_rate,
                weight_decay=myhparams.weight_decay
            )
        elif myhparams.optimizer == "rmsprop":
            self.optimizers = torch.optim.RMSprop(
                filter(lambda x: x.requires_grad, net.parameters()),
                lr=myhparams.learning_rate,
                weight_decay=myhparams.weight_decay
            )
        elif myhparams.optimizer == "adamw":
            self.optimizers = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, net.parameters()),
                lr=myhparams.learning_rate,
                weight_decay=myhparams.weight_decay
            )
        elif myhparams.optimizer == "adahessian":
            self.optimizers = torch_optimizer.Adahessian(
                net.parameters(),
                lr= myhparams.learning_rate,
                weight_decay=myhparams.weight_decay/myhparams.learning_rate,
            )
class Scheduler:
    def __init__(
        self,
        optimizers,
        myhparams
        ):
        epochs = myhparams.max_epochs
        if myhparams.scheduler == "WarmupCosine":
            self.schedulers = WarmupCosineLR(optimizers, warmup_epochs=epochs * 0.3, max_epochs=epochs)
        elif myhparams.scheduler == "Step":
            self.schedulers = StepLR(optimizers, step_size=30, gamma=0.1)
        elif myhparams.scheduler == "Step_wide":
            self.schedulers = StepLR(optimizers, step_size=60, gamma=0.2)
        elif myhparams.scheduler == "FrankleStep":
            self.schedulers = MultiStepLR(optimizers, milestones=[80, 120], gamma=0.1)
        elif myhparams.scheduler == "Cosine":
            self.schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=epochs)
        else:
            self.schedulers = None
       