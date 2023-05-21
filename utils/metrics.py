import torch


def topKAcc(logits, label, K= 5):
    _, pred = torch.topk(logits, k=K)
    num_correct = torch.sum(torch.eq(pred, label.view(-1, 1))).item()
    
    return num_correct / label.size(0)