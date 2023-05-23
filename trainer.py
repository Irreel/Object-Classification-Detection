import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader

from utils.transf import rand_bbox
from utils.metrics import topKAcc


class_indices = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}


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
    top = 5 # K value for topK accuracy

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

        self.device = device

    def save_checkpoint(self):
        raw_model = self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth')
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self, args):
        
        model, config, method = self.model, self.config, self.method
        if method not in ['baseline', 'mixup', 'cutout', 'cutmix']:
            print('method error!')

        if args.cuda: model.to(self.device)
        
        ## Optimizer and Scheduler
        optimizer = optim.Adam(list(model.parameters()),
                            lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                        gamma=args.lr_weight)
        
        # get function handles of loss and metrics
        lossFunc = torch.nn.CrossEntropyLoss()
        # metrics = 

        # Train model
        best_val_loss = np.inf
        best_val_metric = -np.inf
        best_epoch = 0

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=args.batch_size,
                                num_workers=config.num_workers)

            losses = [] # loss list
            accs = [] # accuracy list
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
                            x[:, :, bbx1:bbx2, bby1:bby2] = 0.
                    else:
                        lam = np.random.beta(1.0,1.0)
                        rand_index = torch.randperm(x.size()[0]).to(self.device)
                        target_a = y
                        target_b = y[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                        if method=='cutmix':
                            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                        elif method=='mixup':
                            x = lam * x + (1 - lam) * x[rand_index, :, :]

                # Create a directory to save the images
                save_dir = "image_output"
                os.makedirs(save_dir, exist_ok=True)

                # visualize
                num_imgs = 3

                fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
                for numkk in range(num_imgs):
                    if method == 'baseline':    # baseline不需要可视化，为了对照，这里也可视化一下
                        img_path = os.path.join(save_dir,'baseline')
                        os.makedirs(img_path,exist_ok=True)
                        img_path = os.path.join(save_dir,'baseline',f"image_{str(batch_idx)}.png")
                        ax = fig.add_subplot(4, num_imgs, numkk + 1, xticks=[], yticks=[])
                        img = x[numkk].cpu().numpy().transpose(1, 2, 0)
                        img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                        title = "Original\nlabel:{}".format(class_indices[int(y[numkk].cpu().numpy())])
                        ax.set_title(title)
                        plt.axis('off')
                        plt.imshow(img.astype('uint8'))
                        if numkk==2:
                            plt.savefig(img_path)
                    elif method == 'cutmix':
                        cutmix_img_path = os.path.join(save_dir, 'cutmix')
                        os.makedirs(cutmix_img_path, exist_ok=True)
                        cutmix_img_path = os.path.join(save_dir, 'cutmix', f"cutmix_image_{str(batch_idx)}.png")
                        ax = fig.add_subplot(4, num_imgs, numkk + num_imgs + 1, xticks=[], yticks=[])
                        cutmix_img = x[numkk].cpu().numpy().transpose(1, 2, 0)
                        cutmix_img = (cutmix_img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                        title = "CutMix\nlabel:{}\nadd label:{}".format(
                            class_indices[int(target_a[numkk].cpu().numpy())],
                            class_indices[int(target_b[numkk].cpu().numpy())]
                        )
                        ax.set_title(title)
                        plt.axis('off')
                        plt.imshow(cutmix_img.astype('uint8'))
                        if numkk == 2:
                            plt.savefig(cutmix_img_path)
                    elif method == 'cutout':
                        cutout_img_path = os.path.join(save_dir, 'cutout')
                        os.makedirs(cutout_img_path, exist_ok=True)
                        cutout_img_path = os.path.join(save_dir, 'cutout', f"cutout_image_{str(batch_idx)}.png")
                        ax = fig.add_subplot(4, num_imgs, numkk + 2 * num_imgs + 1, xticks=[], yticks=[])
                        cutout_img = x[numkk].cpu().numpy().transpose(1, 2, 0)
                        cutout_img = (cutout_img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                        title = "Cutout\nlabel:{}\narea:{}".format(
                            class_indices[int(y[numkk].cpu().numpy())],
                            np.round(lam, 2)
                        )
                        ax.set_title(title)
                        plt.axis('off')
                        plt.imshow(cutout_img.astype('uint8'))
                        if numkk == 2:
                            plt.savefig(cutout_img_path)
                    elif method == 'mixup':
                        mixup_img_path = os.path.join(save_dir,'mixup')
                        os.makedirs(mixup_img_path,exist_ok=True)
                        mixup_img_path = os.path.join(save_dir, 'mixup', f"mixup_image_{str(batch_idx)}.png")
                        ax = fig.add_subplot(4, num_imgs, numkk + 3 * num_imgs + 1, xticks=[], yticks=[])
                        mixup_img = x[numkk].cpu().numpy().transpose(1,2,0)
                        mixup_img = (mixup_img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
                        title = "Mixup\nlabel:{}\nadd label:{}".format(
                            class_indices[int(target_a[numkk].cpu().numpy())],
                            class_indices[int(target_b[numkk].cpu().numpy())]
                        )
                        ax.set_title(title)
                        plt.axis('off')
                        plt.imshow(mixup_img.astype('uint8'))
                        if numkk == 2:
                            plt.savefig(mixup_img_path)

                plt.tight_layout()
                plt.close(fig)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits = model(x)
                    loss = lossFunc(logits,target_a)*lam + lossFunc(logits,target_b)*(1.-lam)                    
                    # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    
                    # Calculate metrics
                    if not is_train:
                        acc = topKAcc(logits, y, K=config.top)
                        accs.append(acc)     

                if is_train:
                    # backprop and update the parameters
                    loss.backward()
                    optimizer.step()
                    self.steps += 1

                    # report progress
                    print("train loss: {:04f}".format(loss.item()), file=self.log)                      
                    # print("lr: {:04d}".format(lr), file=self.log)
                    print("epoch {:04d}".format(epoch+1), file=self.log)
                        
                    progress.set_description(f"epoch {epoch+1} iter {batch_idx}: train loss {loss.item():.5f}.")

            if is_train:
                scheduler.step()
                torch.cuda.empty_cache()
                train_loss = float(np.mean(losses))
                return train_loss, None
            else:
                print("-------TEST-------")
                val_loss = float(np.mean(losses))
                val_metric = float(np.mean(accs))
                print("val loss: %f", val_loss, file=self.log)
                print("val metric: %f", val_metric, file=self.log)
                self.log.flush()
                return val_loss, val_metric


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