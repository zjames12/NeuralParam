import math
from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.settings import trace_mode

from functions.matern_nonstat_lengthscale_covariance import MaternNonstatLengthscaleCovariance

class MaternNonstatLengthscaleKernel(Kernel):
    has_lengthscale = True
    def __init__(self, lf, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternNonstatLengthscaleKernel, self).__init__(**kwargs)
        self.nu = nu
        self.feature_extractor = lf
    def forward(self, x1, x2, diag=False, **params):
        
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            l1 = self.feature_extractor(x1)
            l2 = self.feature_extractor(x2)
            
            mean = x1.mean(dim=-2, keepdim=True)

            x1_ = (x1 - mean).div(l1)
            x2_ = (x2 - mean).div(l2)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            matern_kernel = constant_component * exp_component
        else:
            matern_kernel = MaternNonstatLengthscaleCovariance.apply(
            x1, x2, self.lf, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params))
        return matern_kernel