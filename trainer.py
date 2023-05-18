import os

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from torch.utils.data.dataloader import DataLoader
# from train_aug import *

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

    def __init__(self, model, device, log, method,train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.log = log
        self.method = method
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
        
        model, config, method = self.model, self.config, self.method
        if method not in ['baseline', 'mixup', 'cutout', 'cutmix']:
            print('method error!')

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

                target_a = y
                target_b = y
                lam = 1.0
                r = np.random.rand(1)
                length = 16

                # data augmentation
                if method!='baseline' and is_train:
                    if method=='cutout':
                        _,_,h,w = x.shape
                        h = x.shape[2]
                        w = x.shape[3]
                        lam = 1-(length**2/(h*w))
                        for _ in range(1):
                            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    else:
                        lam = np.random.beta(1.0,1.0)
                        rand_index = torch.randperm(x.size()[0]).to(device)
                        target_a = y
                        target_b = y[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                        if method=='cutmix':
                            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                        elif method=='mixup':
                            x = lam * input + (1 - lam) * input[rand_index, :, :]

                # TODO forward the model
                with torch.set_grad_enabled(is_train):
                    x_hat = model(x)
                    # TODO sample run
                    print("data load Succeed")
                    raise Exception
                    # TODO: update the model architecture & the output dimension
                    # loss = lossFunc(x_hat, x)
                    loss = lossFunc(x_hat,target_a)*lam + lossFunc(x_hat,target_b)*(1.-lam)
                    # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(lossFunc.item())
                    # losses.append(loss.item())

                # visualize
                num_imgs = 3
                fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
                for numkk in range(num_imgs):
                    ax = fig.add_subplot(1, num_imgs, numkk + 1, xticks=[], yticks=[])
                    img = x[numkk].cpu().numpy().transpose(1, 2, 0)
                    img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                    if method == 'baseline':
                        title = "{}\nlabel:{}".format(method, class_indices[int(y[numkk].cpu().numpy())])
                    elif method == 'cutout':
                        title = "{}\nlabel:{}({})".format(method, class_indices[int(y[numkk].cpu().numpy())],
                                                          np.round(lam, 2))
                    else:
                        title = "{}\nlabel:{}({})\nadd label:{}({})".format(method,
                                                                            class_indices[
                                                                                int(target_a[numkk].cpu().numpy())],
                                                                            np.round(lam, 2),
                                                                            class_indices[
                                                                                int(target_b[numkk].cpu().numpy())],
                                                                            np.round(1 - lam, 2))
                    ax.set_title(title)
                    plt.axis('off')
                    plt.imshow(img.astype('uint8'))
                plt.show()

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
                        # print("train loss: {:04d}".format(loss.item()), file=self.log)
                        print("lr: {:04d}".format(lr), file=self.log)
                        print("epoch {:04d}".format(epoch+1), file=self.log)
                        print("step {:04d}".format(self.steps), file=self.log)
                        
                        progress.set_description(f"epoch {epoch+1} iter {batch_idx}: train loss {lossFunc.item():.5f}. lr {lr:e}")

            if is_train:
                scheduler.step()
                torch.cuda.empty_cache()
                return train_loss, None # train_metric
                # return losses, None
            else:
                print("-------TEST-------")
                val_loss = float(np.mean(losses))
                print("val loss: %f", val_loss, file=self.log)
                # print("val metric: %f", None, file=self.log)
                print("step: %d", self.steps, file=self.log)
                self.log.flush()
                return val_loss, None #valid_metric

        def rand_bbox(self,size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

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