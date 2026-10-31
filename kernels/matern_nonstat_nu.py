import math
import torch
import gpytorch

from typing import Optional
from gpytorch.kernels import Kernel
from gpytorch.settings import trace_mode

class MaternNonstatNuKernel(Kernel):
    has_lengthscale = True
    def __init__(self, lf, **kwargs):
        super(MaternNonstatNuKernel, self).__init__(**kwargs)
        self.feature_extractor = lf
    def forward(self, x1, x2, diag=False, **params):
        res1 = self.feature_extractor(x1)
        res2 = self.feature_extractor(x2)
        sigma05_1 = res1[:,0].unsqueeze(-1)
        sigma05_2 = res2[:,0].unsqueeze(-1)
        sigma25_1 = res1[:,1].unsqueeze(-1)
        sigma25_2 = res2[:,1].unsqueeze(-1)

        if diag:
            # w = (sigma05_1 * sigma05_2).squeeze(-1)
            non_var05 = (sigma05_1 * sigma05_2).squeeze(-1)
            non_var25 = (sigma25_1 * sigma25_2).squeeze(-1)
        else:
            # w = sigma05_1 * sigma05_2.T
            non_var05 = sigma05_1 * sigma05_2.T
            non_var25 = sigma25_1 * sigma25_2.T
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.mean(dim=-2, keepdim=True)

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component05 = torch.exp(-math.sqrt(0.5 * 2) * distance)
            exp_component25 = torch.exp(-math.sqrt(2.5 * 2) * distance)

            constant_component05 = 1
            constant_component25 = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)

            matern_kernel = non_var05 * constant_component05 * exp_component05 + non_var25 * constant_component25 * exp_component25
            # matern_kernel = w * constant_component05 * exp_component05 + (1 - w) * constant_component25 * exp_component25
        else:
            matern_kernel05 = gpytorch.functions.MaternCovariance.apply(
            x1, x2, self.lengthscale, 0.5, lambda x1, x2: self.covar_dist(x1, x2, **params))
            matern_kernel25 = gpytorch.functions.MaternCovariance.apply(
            x1, x2, self.lengthscale, 2.5, lambda x1, x2: self.covar_dist(x1, x2, **params))
            # matern_kernel = matern_kernel05 * w + matern_kernel25 * (1 - w)
            matern_kernel = matern_kernel05 * non_var05 + matern_kernel25 * non_var25
        return matern_kernel