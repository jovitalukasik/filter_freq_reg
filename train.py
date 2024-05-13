import torch
import models
import data
import os
import argparse
import yaml
import sys
import logging
from utils import (
    CSVLogger,
    ConsoleLogger,
    WandBLogger,
    CheckpointCallback,
    none2str,
    str2bool,
    prepend_key_prefix,
    seed_everything,
    get_arg,
)
from optimizer import Optimizer, Scheduler
import datetime
from dct_conv import replace_conv2d, Weight_Decomposition, Signal_Decomposition
from efficientnet_pytorch import EfficientNet
import efficientnet_pytorch
import torch.nn.functional as F
from dct_losses import l2_reg
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np




class Trainer:
    def __init__(self, args):
        self.model = Trainer.prepare_model(
            args, args.model_in_channels, args.model_num_classes
        )
        self.device = args.device
        self.args = args

    def prepare_model(args, in_channels, num_classes):

        logging.info(f"Initializing {args.model}")

        if "efficientnet" in args.model: 
            model = EfficientNet.from_name(args.model, num_classes=num_classes)
            model._conv_stem.stride = (1,1)
        else:
            model = models.get_model(args.model)(
                in_channels=in_channels, num_classes=num_classes
            )
        
        if args.basis_filter == "WD":
            replace_conv2d(model, Weight_Decomposition,kernel_size=args.cnn_kernel_size, dct_kernel_size=args.dct_kernel_size,  init_scale=args.init_scale)
            if "efficientnet" in args.model: 
                replace_conv2d(model, Weight_Decomposition, kernel_size=5, dct_kernel_size=5, init_scale=args.init_scale)

        if args.basis_filter == "SD":
            replace_conv2d(model, Signal_Decomposition, kernel_size=args.cnn_kernel_size, dct_kernel_size=args.dct_kernel_size)
            if "efficientnet" in args.model: 
                replace_conv2d(model, Signal_Decomposition, kernel_size=5, dct_kernel_size=5)

        if args.load_checkpoint is not None:

            logging.info(f"Loading state from {args.load_checkpoint}")

            state = torch.load(args.load_checkpoint, map_location="cpu")

            if "state_dict" in state:
                state = state["state_dict"]

            model.load_state_dict(state)

        model.to(args.device)

        print(model)

        logging.info(
            f"TOTAL: {sum(list(map(lambda p: p.numel(), filter(lambda p: p.requires_grad, model.parameters()))))}"
        )


        return model

    def train(self, model, trainloader, opt, criterion, device, scheduler=None, loggers=[], writer=None, output_dir='output/', dataset=None):
        correct = 0
        total = 0
        total_loss = 0
        l2_loss = 0
        model.train()

        for i, (x, y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            

            if self.args.l2_reg and self.args.basis_filter in ["WD", "SD"]:
                l2 = l2_reg(model, self.args)
                if l2 == 0:
                    l2 = torch.zeros_like(loss)
                loss = loss + l2
            else:
                l2 = torch.zeros(1)

            if self.args.optimizer == "adahessian":
                loss.backward(create_graph=True)
            else:
                loss.backward()
            opt.step()

            batch_loss = loss.item() * len(y)
            batch_correct = (y_pred.argmax(axis=1) == y).sum().item()

            l2_loss += l2.item()* len(y)

            correct += batch_correct
            total_loss += batch_loss

            total += len(y)
            self.steps += 1

            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), self.epoch)
                writer.add_scalar("Acc/train",  batch_correct / len(y), self.epoch)

        if scheduler:
            scheduler.step()

        for logger in loggers:
            logger.log(self.epoch, {"train/batch_acc": correct / total, "train/batch_loss": total_loss / total, "train/l2_loss": l2_loss/ total}, silent=True)

        return {"acc": correct / total, "loss": total_loss / total, "l2_loss": l2_loss/ total}

    def validate(self, model, valloader, criterion, device, loggers=[],  writer=None):
        correct = 0
        total = 0
        total_loss = 0
        l2_loss = 0

        model.eval()
        with torch.no_grad():
            for x, y in valloader:
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)
                
                total_loss += loss.item() * len(y)

                correct += (y_pred.argmax(axis=1) == y).sum().item()
                total += len(y)

                if writer is not None:
                    writer.add_scalar("Loss/val", loss, self.epoch)
                    writer.add_scalar("Acc/val", ((y_pred.argmax(axis=1) == y).sum().item())/len(y), self.epoch)

                if self.args.l2_reg and self.args.basis_filter in ["WD", "SD"]:
                    l2 = l2_reg(model, self.args)
                    if l2 == 0:
                        l2 = torch.zeros_like(loss)
                    loss = loss + l2
                else:
                    l2 = torch.zeros(1)
            
                l2_loss += l2.item()* len(y)
                

        return {"acc": correct / total, "loss": total_loss / total, "l2_loss": l2_loss/ total}

    def fit(self, dataset, output_dir=None):
        writer = SummaryWriter() if self.args.tensorboard_project else None

        trainloader = dataset.train_dataloader(self.args.batch_size, self.args.num_workers)
        valloader = dataset.val_dataloader(self.args.batch_size, self.args.num_workers)

        # # Setup Adam optimizers for G 
        self.opt = Optimizer(self.model, self.args).optimizers
        self.scheduler = Scheduler(self.opt, self.args).schedulers


        self.criterion = torch.nn.CrossEntropyLoss()

        loggers = []
        loggers.append(ConsoleLogger())
        if output_dir:
            loggers.append(CSVLogger(os.path.join(output_dir, "metrics.csv")))
        if self.args.wandb_project:
            loggers.append(WandBLogger(self.model, self.args.wandb_project, self.args, output_dir))

        if output_dir:
            self.checkpoint = CheckpointCallback(
                os.path.join(output_dir, "checkpoints"),
                mode=self.args.checkpoints,
                args=vars(self.args),
            )

        self.epoch = 0
        self.steps = 0

        val_metrics = self.validate(self.model, valloader, self.criterion, self.device)
        for logger in loggers:
            logger.log(0, prepend_key_prefix(val_metrics, "val/"))
        if output_dir:
            self.checkpoint.save(0, self.model, {})

        val_acc_max = 0
        best_epoch = 0

        for epoch in range(self.args.max_epochs):

            self.epoch = epoch

            train_metrics = self.train(
                self.model,
                trainloader,
                self.opt,
                self.criterion,
                self.device,
                self.scheduler,
                loggers=loggers, 
                writer=writer, 
                output_dir=output_dir,
                dataset=dataset
            )

            val_metrics = self.validate(
                self.model, valloader, self.criterion, self.device, 
                writer=writer
            )

            metrics = {
                **prepend_key_prefix(train_metrics, "train/"),
                **prepend_key_prefix(val_metrics, "val/"),
                "val/acc_max": val_acc_max,
                "best_epoch": best_epoch,
            }

            if val_acc_max < metrics["val/acc"]:
                val_acc_max = metrics["val/acc"]
                best_epoch = epoch

                metrics["val/acc_max"] = val_acc_max
                metrics["best_epoch"] = best_epoch
                self.checkpoint.save(epoch, self.model, metrics, force=True, name="best")


            for logger in loggers:
                logger.log(epoch, metrics)
            if output_dir:
                self.checkpoint.save(epoch, self.model, metrics)



def main(args):
    print(args)
    if args.reg_diff_all:
        args.reg_all_freq = 0 
        args.reg_diff = 0

    logging.basicConfig(level=logging.INFO)
    if get_arg(args, "verbose"):
        logging.basicConfig(level=logging.DEBUG)

    dataset = data.get_dataset(args.dataset)(
        os.path.join(args.data_dir, args.dataset)
    )

    seed_everything(args.seed)

    if args.model_in_channels == -1:
        vars(args)["model_in_channels"] = dataset.in_channels

    if args.model_num_classes == -1:
        vars(args)["model_num_classes"] = dataset.num_classes

    output_dir = args.output_dir
    now = datetime.datetime.now()
    runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")

    output_dir = os.path.join(args.output_dir, args.dataset, args.model, args.basis_filter, runfolder+"_"+str(args.seed))


    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "hparams.yaml"), "w") as file:
        yaml.dump(vars(args), file)


    Trainer(args).fit(dataset, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",                type=str, default="cifar10")
    parser.add_argument("--data_dir",               type=str, default="data/")
    parser.add_argument("--output_dir",             type=str, default="output")
    parser.add_argument("--device",                 type=str, default="cuda:0")
    parser.add_argument("--checkpoints",            type=none2str, default=10, choices=["all", None, 10])
    parser.add_argument("--load_checkpoint",        type=none2str, default=None)
    parser.add_argument("--model",                  type=str, default='lowres_resnet9')

    parser.add_argument("--model_in_channels",      type=int, default=-1)
    parser.add_argument("--model_num_classes",      type=int, default=-1)

    parser.add_argument("--max_epochs",             type=int, default=125)
    parser.add_argument("--batch_size",             type=int, default=256)
    parser.add_argument("--num_workers",            type=int, default=min(8, os.cpu_count()))

    # optimizer
    parser.add_argument("--optimizer",              type=str, default="adamw", choices=["adam", "sgd", "adagrad", "rmsprop", "adamw", "adahessian"])
    parser.add_argument("--learning_rate",          type=float, default=0.1)
    parser.add_argument("--weight_decay",           type=float, default=0.05)
    parser.add_argument("--momentum",               type=float, default=0.9)
    parser.add_argument("--scheduler",              type=none2str, default="Step",
                                                        choices=["WarmupCosine", "Step", "FrankleStep", "None", None, "Cosine", "Step_wide"])
    parser.add_argument("--seed",                   type=int, default=2)

    parser.add_argument("--verbose",                type=str2bool, default=True)

    parser.add_argument("--wandb_project",          type=none2str, default=None)
    parser.add_argument("--tensorboard_project",    type=int, default=0)#

    parser.add_argument("--basis_filter",           type=str, default="None", help="Choose between None, WD, or SD")
    parser.add_argument("--init_scale",             type=str, default="kaiming_uniform", help="Choose between kaiming_uniform and kaiming_normal")
    
    # L2 regularization:
    parser.add_argument("--l2_reg",                 type=int, default=0, help="Wether to include L2 regularization on frequency (only for DCT)")
    parser.add_argument("--l2_lambda",              type=float, default=0.01, help="Regularization parameter for L2")
    parser.add_argument("--reg_depth",              type=float, default=3, help="Depth of network to Reg, a third, half, all")
    parser.add_argument("--reg_all_freq",           type=int, default=0, help="Wether to regularize all frequency (except 0) or only highest")
    parser.add_argument("--reg_diff",               type=int, default=0, help="Wether to force to focus on low frequency")
    parser.add_argument("--reg_diff_all",           type=int, default=0, help="Wether to force to focus on low frequency")

    # Manually define Kernel size for DCT_basis
    parser.add_argument("--dct_kernel_size",        type=int, default=3, help="Which kernel size information used for DCT")
    parser.add_argument("--cnn_kernel_size",        type=int, default=3, help="Which conv kernel size to replace")

    
    _args = parser.parse_args()

    main(_args)
    sys.exit(0)