import torch
import torch.nn as nn


def custom_age_gender_loss(predict, target, weight_factor_age, lambda_age=1, lambda_gender=1):
    assert predict.shape == target.shape

    pred_gender = predict[:, :2]
    pred_age = predict[:, 2:]

    target_gender = target[:, :2]
    target_age = target[:, 2:]

    # for b in range(predict.shape[0]):
    #     print(f'(1)PREDICT: {pred_gender[b]} {pred_age[b]}')
    #     print(f'(2)TARGET: {target_gender[b]} {target_age[b]}')

    loss_gender = lambda_gender * custom_softmax_cross_entropy_loss(pred_gender, target_gender)
    loss_age = lambda_age * custom_softmax_cross_entropy_loss(pred_age, target_age) * weight_factor_age

    return loss_gender + loss_age, loss_gender, loss_age


def custom_weighted_softmax_focal_loss(predict, target, weight_factor, gamma=0):
    assert predict.shape == target.shape

    predict = nn.Softmax(dim=len(predict.shape) - 1)(predict)
    num_batches = predict.shape[0]
    loss_temp = -target * torch.pow(1 - predict, exponent=gamma) * torch.log2(predict + 1e-20)

    return loss_temp.sum() / num_batches


def custom_weighted_softmax_cross_entropy_loss(predict, target, weight_factor):
    assert predict.shape == target.shape

    predict = nn.Softmax(dim=len(predict.shape) - 1)(predict)
    num_batches = predict.shape[0]
    loss_temp = -target * torch.log2(predict + 1e-20) * weight_factor

    return loss_temp.sum() / num_batches


def custom_softmax_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    predict = nn.Softmax(dim=len(predict.shape) - 1)(predict)
    # print(f'Predict: {predict[0]')
    # print(f'Target: {target[0]}')
    num_batches = predict.shape[0]
    # loss_temp = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))
    loss_temp = -target * torch.log2(predict + 1e-20)

    return loss_temp.sum() / num_batches


def custom_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    num_batches = predict.shape[0]
    # loss_temp = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))
    loss_temp = -target * torch.log2(predict + 1e-20)

    return loss_temp.sum() / num_batches
