import os
import time
import pickle
import argparse
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
# from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser()
# Environment and datasetss
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device-id', type=str, default='2', 
                        help='Available gpu id. Disable when no-cuda is True')
parser.add_argument('--validation-split', type=float, default=0.1, 
                        help='Split ratio for valid data')
# Training parameters
parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
parser.add_argument('--early-stop', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=50,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--lr-weight', type=float, default=0.1,
                        help='LR decay factor.')
# Save and visualize
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--tensorboard', type=bool, default=True)
    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
    
# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if args.cuda:
    torch.cuda.manual_seed(SEED)
    device_ids = args.device_id.split(",")
    # device = torch.device("cuda:{}".format("0") if torch.cuda.is_available() else "cpu")
    print(f"device_ids: {device_ids}")
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, device_ids))
    
# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/date{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')

    model_file = os.path.join(save_folder, 'model.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))

    if args.tensorboard: writer = SummaryWriter('{}/date{}/tensorboard/'.format(args.save_folder, timestamp))


def train(epoch, best_val_loss, best_val_metric):
    t = time.time()
    ## Training stage
    train_loss = []
    # TODO
    
    ## Valid stages
    val_loss = []
    # TODO
    
    # Save model
    if args.save_folder and np.mean(val_loss) < best_val_loss:
        ## torch.save(model.cpu().state_dict(), model_file)
        ## torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
            # 'train_loss: {:.10f}'.format(train_loss),
            # 'val_loss: {:.10f}'.format((val_loss)),              
            'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
        
    # return np.mean(train_loss), train_metric, np.mean(val_loss), val_metric


if __name__ == '__main__':
    
    ## setup data_loader instances
    # data_loader = 
    # valid_data_loader = 

    ## build model architecture, then print to console
    # model = 
    # print(model.info())
    # if args.cuda: model.cuda()
    # model.train()

    ## prepare for (multi-device) GPU training
    # device, device_ids = 
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    ## get function handles of loss and metrics
    # loss = 
    # metrics = 

    ## Optimizer and Scheduler
    # optimizer = optim.Adam(list(model.parameters()),
    #                     lr=args.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
    #                                 gamma=args.gamma)

    # Train model
    t_total = time.time()
    best_val_loss = np.inf
    # best_val_metric = 0
    best_epoch = 0
    
    ## Pass args to train function
    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   args=args,
    #                   device=device,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   scheduler=scheduler)
    
    ## train stage and valid stage
    for epoch in range(args.epochs):
        
        # train_loss, train_metric, val_loss, val_metric = trainer.train(epoch, best_val_loss, best_val_metric, ...)
        
        # if val_loss < best_val_loss:
            # best_val_loss = val_loss
            # best_epoch = epoch
        
        pass
    
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        print(args, file=log)
        log.flush()

    log.close()