import torch


def accuracy(predict, target):
    assert predict.shape == target.shape

    and_sum = torch.bitwise_and(predict.type(torch.IntTensor), target.type(torch.IntTensor)).sum()
    or_sum = torch.bitwise_or(predict.type(torch.IntTensor), target.type(torch.IntTensor)).sum()

    return and_sum / or_sum / predict.shape[0]


def accuracy_argmax(predict, target):
    """
    :param predict: tensor, [num batches, (categorical probabilities)]
    :param target: tensor, [num_batches, (categorical probabilities)]

    :return: tensor, accuracy
    """
    assert predict.shape == target.shape

    pred_argmax = predict.argmax(dim=1)
    tar_argmax = target.argmax(dim=1)

    match = (pred_argmax == tar_argmax)

    return match.sum() / predict.shape[0]


def exact_match_ratio(predict, target):
    assert predict.shape == target.shape

    match = 0
    for i in range(predict.shape[0]):
        if False not in (predict[i] == target[i]):
            match += 1

    return match / predict.shape[0]