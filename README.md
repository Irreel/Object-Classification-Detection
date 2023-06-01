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

## Training
Simply run
`python train.py --method baseline`
Other `method` could be `mixup`, `cutmix` and `cutout`

- Batch size: 128
- Optimizer: Adam
- Scheduler: [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html?highlight=steplr#torch.optim.lr_scheduler.StepLR)
- Loss function: CrossEntrophyLoss
- Metrics: Top5 accuracy

## Visualization results
- Tensorboard results:
Given the file in `/logs`,
open tensorboard via `tensorboard --logdir ./tensorboard`