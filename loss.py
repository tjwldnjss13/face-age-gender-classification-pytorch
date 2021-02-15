import torch
import torch.nn as nn


def custom_softmax_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    predict = nn.Softmax(dim=len(predict.shape) - 1)(predict)
    num_batches = predict.shape[0]
    loss_temp = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))

    return loss_temp.sum() / num_batches
