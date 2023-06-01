# Object-Classification-Detection

## Environment Setup
- Pytorch >= 1.10.2 (CUDA 11.3 / CPU)
- torchvision
- tensorboard
- numpy
- matlablib
- tqdm (Optional)


Download dataset into `data/` folder.

Model is saved into `logs/` by default, you can customized it via `--save-folder` argument.

## Training and Testing
Simply run
`python train.py --method baseline`
Other `method` could be `mixup`, `cutmix` and `cutout`

- Batch size: 128
- Optimizer: Adam
- Scheduler: [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html?highlight=steplr#torch.optim.lr_scheduler.StepLR)
- Loss function: CrossEntrophyLoss
- Metrics: Top5 Accuracy

model    | batch size | lr | epoch | method | train loss | valid loss | acc@5
---------|-----|-------|-----|--------|--------|--------|--------|
ResNet18 | 128 | 1e-3 | 30   | baseline  | 1.147292  | 1.967063   | 0.788469  |
ResNet18 | 128 | 1e-3 | 80   | baseline  | 0.101848  | 3.024865   | 0.774229  | 
ResNet101| 128 | 1e-3 | 30   | baseline  | 2.297507  | 14.033395  | 0.620847  | 


## Visualization results
- Tensorboard results:
Given the file in `/logs`, open tensorboard via `tensorboard --logdir ./tensorboard`

For example:

Model architecture: ResNet18

Method: Baseline

![Train loss](https://github.com/Irreel/Object-Classification-Detection/blob/main/loss_train.png?#pic_left=300x)

![Valid loss](https://github.com/Irreel/Object-Classification-Detection/blob/main/loss_valid.png?#pic_left=300x)

![Acc@5](https://github.com/Irreel/Object-Classification-Detection/blob/main/acc.png?#pic_left=300x)

(Smoothing == 0.6)