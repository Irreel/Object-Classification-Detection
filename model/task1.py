import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, alexnet

N_CLASS = 100 #CIFAR100

def get_model():
    """
    Load models from torchvision and modify their architecture
    """
    model = resnet18()
    # model = resnet34()
    # model = resnet101()
    # model = alexnet() # the last layer is different
    print(model)
    # Modify the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, N_CLASS, bias=True)
    return model