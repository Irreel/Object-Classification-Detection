import os
import math

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    # grad_norm_clip = 1.0
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, device, log, train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.log = log
        self.steps = 0

        # print("Using tensorboards")

        # take over whatever gpus are on the system
        self.device = device
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth')
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self, args):
        
        model, config = self.model, self.config
        if args.cuda: model.cuda()
        
        ## Optimizer and Scheduler
        optimizer = optim.Adam(list(model.parameters()),
                            lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                        gamma=args.lr_weight)
        
        # get function handles of loss and metrics
        lossFunc = torch.nn.MSELoss(reduction="mean")
        # metrics = 

        # Train model
        best_val_loss = np.inf
        # best_val_metric = 0
        best_epoch = 0

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=args.batch_size,
                                num_workers=config.num_workers)

            losses = []
            progress = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for batch_idx, (x, y) in progress:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                if is_train: optimizer.zero_grad()

                # TODO forward the model
                with torch.set_grad_enabled(is_train):
                    x_hat = model(x)
                    # TODO sample run
                    print("data load Succeed")
                    raise Exception
                    # TODO: update the model architecture & the output dimension
                    loss = lossFunc(x_hat, x)
                    # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(lossFunc.item())

                if is_train:
                    # backprop and update the parameters
                    lossFunc.backward()
                    optimizer.step()
                    self.steps += 1
                    
                    ## Decay strategy (TODO) may not compatible with scheduler
                    # decay the learning rate based on our progress 
                    # if config.lr_decay:
                    #     if self.steps < config.warmup_iters:
                    #         # linear warmup
                    #         lr_mult = float(self.steps) / float(max(1, config.warmup_iters))
                    #     else:
                    #         # cosine learning rate decay
                    #         progress = float(self.steps - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                    #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            
                    #     lr = config.learning_rate * lr_mult
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    # else:
                    #     lr = config.learning_rate

                    if is_train:
                        # report progress
                        print("train loss: {:04d}".format(lossFunc.item()), file=self.log)
                        print("lr: {:04d}".format(lr), file=self.log)
                        print("epoch {:04d}".format(epoch+1), file=self.log)
                        print("step {:04d}".format(self.steps), file=self.log)
                        
                        progress.set_description(f"epoch {epoch+1} iter {batch_idx}: train loss {lossFunc.item():.5f}. lr {lr:e}")

            if is_train:
                scheduler.step()
                torch.cuda.empty_cache()
                return train_loss, None # train_metric                
            else:
                print("-------TEST-------")
                val_loss = float(np.mean(losses))
                print("val loss: %f", val_loss, file=self.log)
                # print("val metric: %f", None, file=self.log)
                print("step: %d", self.steps, file=self.log)
                self.log.flush()
                return val_loss, None #valid_metric
        
        ## train stage and valid stage
        for epoch in range(config.max_epochs):
            train_loss, train_metric = run_epoch('train')
            if self.valid_dataset is not None:
                with torch.no_grad():
                    valid_loss, valid_metric = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            is_good_model = self.valid_dataset is None or valid_loss < best_val_loss
            if self.config.ckpt_dir is not None and is_good_model:
                best_val_loss = valid_loss
                # best_val_metric
                best_epoch = epoch
                self.save_checkpoint()
     
        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=self.log)
            print(args, file=self.log)
            self.log.flush()