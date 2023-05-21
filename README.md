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
`python train.py --method baseline`

- Batch size: 128
- Optimizer: Adam
- Scheduler: [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html?highlight=steplr#torch.optim.lr_scheduler.StepLR)
- Loss function: CrossEntrophyLoss
- Metrics: Top5 accuracy


[实验报告(Google Docs)](https://docs.google.com/document/d/1f-ELCHgz8NBzv_fX20z3Q0d2B0p74GKE2hnWBn19QKQ/edit?usp=sharing)

详细的实验报告包括实验设置：

数据集介绍，训练测试集划分，网络结构，batch size，learning rate，优化器，iteration，epoch，loss function，评价指标，检测/分割结果可视化；
利用Tensorboard可视化训练和测试的loss曲线、测试mAP/Acc/mIoU 曲线。