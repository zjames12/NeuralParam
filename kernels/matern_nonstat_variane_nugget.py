import math
import torch
import gpytorch

from typing import Optional
from gpytorch.kernels import Kernel
from gpytorch.settings import trace_mode


class CustomKernelVarNug(Kernel):
        has_lengthscale = True
        def __init__(self, lf, nu: Optional[float] = 2.5, **kwargs):
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
            super(CustomKernelVarNug, self).__init__(**kwargs)
            self.nu = nu
            self.feature_extractor = lf
        def forward(self, x1, x2, diag=False, **params):
            res1 = self.feature_extractor(x1)
            res2 = self.feature_extractor(x2)
            sigma1 = res1[:,0].unsqueeze(-1)
            sigma2 = res2[:,0].unsqueeze(-1)
            nug1 = res1[:,1].unsqueeze(-1)
            nug2 = res2[:,1].unsqueeze(-1)
            if diag:
                non_var = (sigma1 * sigma2).squeeze(-1)
                non_nug = nug1.squeeze(-1)
            else:
                non_var = sigma1 * sigma2.T
                mask = (x1[:, None, :] == x2[None, :, :]).all(dim=2)
                non_nug = torch.where(mask, nug1.expand(-1, nug2.shape[0]), torch.tensor(0.0))
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
                exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

                if self.nu == 0.5:
                    constant_component = 1
                elif self.nu == 1.5:
                    constant_component = (math.sqrt(3) * distance).add(1)
                elif self.nu == 2.5:
                    constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
                matern_kernel = constant_component * exp_component
            else:
                matern_kernel = gpytorch.functions.MaternCovariance.apply(
                x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params))
            return matern_kernel * non_var + non_nug