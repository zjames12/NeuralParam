# Feature extractor definitions for neural parametrization

import torch
import torch.nn as nn


class LargeFeatureExtractor(nn.Sequential):
    """
    Deep feature extractor network with L2 regularization and dropout.

    Args:
        n_dim: Input dimensionality
        dropout_p: Dropout probability (default: 0.2)
    """
    # def __init__(self, n_dim, dropout_p=0.00, sigmoid = False):
    #     super(LargeFeatureExtractor, self).__init__()

    #     # Deep network architecture with dropout for regularization
    #     self.add_module('linear1', nn.Linear(n_dim, 25))
    #     self.add_module('batchnorm1', nn.BatchNorm1d(25))
    #     self.add_module('relu1', nn.ReLU())

    #     self.add_module('linear2', nn.Linear(25, 25))
    #     self.add_module('batchnorm2', nn.BatchNorm1d(25))
    #     self.add_module('relu2', nn.ReLU())

    #     self.add_module('linear3', nn.Linear(25, 10))
    #     self.add_module('batchnorm3', nn.BatchNorm1d(10))
    #     self.add_module('relu3', nn.ReLU())

    #     self.add_module('linear4', nn.Linear(10, 1))
    #     if sigmoid:
    #         self.add_module("sigmoid", nn.Sigmoid())
    #     else:
    #         self.add_module('softplus', nn.Softplus())

        # self._initialize_weights()

    def __init__(self, n_dim, dropout_p=0.00, sigmoid = False):
        super(LargeFeatureExtractor, self).__init__()

        self.add_module('linear1', nn.Linear(n_dim, 64))
        self.add_module('batchnorm1', nn.BatchNorm1d(64))
        self.add_module('gelu1', nn.GELU())

        self.add_module('linear2', nn.Linear(64, 128))
        self.add_module('batchnorm2', nn.BatchNorm1d(128))
        self.add_module('gelu2', nn.GELU())

        self.add_module('linear3', nn.Linear(128, 128))
        self.add_module('batchnorm3', nn.BatchNorm1d(128))
        self.add_module('gelu3', nn.GELU())

        self.add_module('linear4', nn.Linear(128, 64))
        self.add_module('batchnorm4', nn.BatchNorm1d(64))
        self.add_module('gelu4', nn.GELU())

        self.add_module('linear5', nn.Linear(64, 32))
        self.add_module('batchnorm5', nn.BatchNorm1d(32))
        self.add_module('gelu5', nn.GELU())

        self.add_module('linear6', nn.Linear(32, 2))
        if sigmoid:
            self.add_module("sigmoid", nn.Sigmoid())
        else:
            self.add_module('softplus', nn.Softplus())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.01)   # ← swap this out
                nn.init.zeros_(m.bias)