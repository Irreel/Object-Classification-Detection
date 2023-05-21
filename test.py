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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser()
# Environment and datasetss
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device-id', type=str, default='2', 
                        help='Available gpu id. Disable when no-cuda is True')
# parser.add_argument('--validation-split', type=float, default=0.1, 
#                         help='Split ratio for valid data')

# Load and visualize
parser.add_argument('--load-folder', type=str, default='saved',
                    help='Where to load the trained model')
parser.add_argument('--tensorboard', type=bool, default=True)
parser.add_argument('--method',type=str,default='baseline',
                    help='Other data augmentation methods are mixup, cutmix and cutout')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device_ids = args.device_id.split(",")
    # device = torch.device("cuda:{}".format("0") if torch.cuda.is_available() else "cpu")
    print(f"device_ids: {device_ids}")
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, device_ids))
else:
    device = torch.device("cpu")
    
    
def main(args):
    
    ## setup test data_loader 
    # test_data_loader = 
    
    ## load model
    # model = 
    if args.load_folder:
        # model_file = os.path.join(args.load_folder, 'model.pt')
        # model.load_state_dict(torch.load(model_file))
        pass
    # print(model.info())
    # model.eval()
    
    ## Test
    
    return

if __name__ == '__main__':
    main(args)