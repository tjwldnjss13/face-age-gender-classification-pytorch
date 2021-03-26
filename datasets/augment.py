import torch
import numpy as np

class GaussianNoise(object):
    def __init__(self, mean=0., std=.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}'


class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=.1, prob=.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if np.random.random() <= self.prob:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}'