import os
import time
import pickle
import argparse
import datetime
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from trainer import TrainerConfig, Trainer
from torch.utils.tensorboard import SummaryWriter

from utils.transf import transfm_baseline
from model.task1 import get_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser()
# Environment and datasetss
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device-id', type=str, default='2', 
                        help='Available gpu id. Disable when no-cuda is True')
parser.add_argument('--validation-split', type=float, default=0.1, 
                        help='Split ratio for valid data')
parser.add_argument('--method',type=str,default='baseline',
                    help='Other data augmentation methods are mixup, cutmix and cutout')
# Training parameters
parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs to train.')
parser.add_argument('--early-stop', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=40,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--lr-weight', type=float, default=0.1,
                        help='LR decay factor.')
# Save and visualize
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--tensorboard', type=bool, default=True)
    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
method = args.method
print(args)
print("method",method)
    
# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if args.cuda:
    torch.cuda.manual_seed(SEED)
    device_ids = args.device_id.split(",")
    print(f"device_ids: {device_ids}")
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, device_ids))
else:
    device = torch.device("cpu")
    
# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/date{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    # model_file = os.path.join(save_folder, 'model.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    
    pickle.dump({'args': args}, open(meta_file, "wb"))

    if args.tensorboard: writer = SummaryWriter('{}/date{}/tensorboard/'.format(args.save_folder, timestamp))


if __name__ == '__main__':
    ## prepare for (multi-device) GPU training
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    ## setup data_loader instances
    train_dataset = CIFAR100('data/', train=True, download=False,transform=transfm_baseline, target_transform=None)
    valid_dataset = CIFAR100('data/', train=False, download=False,transform=transfm_baseline, target_transform=None)

    ## setup model
    model = get_model()
    # print(model)
    
    cfg = TrainerConfig(max_epoch = args.epochs, ckpt_dir=save_folder)
    trainer = Trainer(model, device, log, method, train_dataset=train_dataset, valid_dataset=valid_dataset, config=cfg, tsrbd=writer)
    trainer.train(args)

    log.close()