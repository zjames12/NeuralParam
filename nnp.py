# This file includes the new kerneel classes that were created for implementing the framework in the paper.

import torch
import gpytorch
import numpy as np
import pandas as pd
import time
import math
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel, InducingPointKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal


def fit_model(X_train, y_train, X_validation, y_validation, X_test, y_test, training_iterations = 800, learning_rate = 0.05, deep = 0):

    valid_log_scores = []
    lrs = []
    iterations = []
    mses = []
    maes = []
    log_scores = []
    varmses = []
    varmaes = []

    np.random.seed(0)
    torch.manual_seed(0)

    n_dim = X_train.shape[1]

    if deep == 0:
        class LargeFeatureExtractor(torch.nn.Sequential):
            def __init__(self):
                super(LargeFeatureExtractor, self).__init__()
                self.add_module('linear1', torch.nn.Linear(n_dim, 2))
                self.add_module('softplus', torch.nn.Softplus())
    else:
        num_nodes = 50
        class LargeFeatureExtractor(torch.nn.Sequential):
            def __init__(self):
                super(LargeFeatureExtractor, self).__init__()
                self.add_module('linear1', torch.nn.Linear(n_dim, num_nodes))
                self.add_module('batchnorm1', torch.nn.BatchNorm1d(num_nodes))
                self.add_module('relu1', torch.nn.ReLU())
                self.add_module('linear2', torch.nn.Linear(num_nodes, 2))
                self.add_module('softplus', torch.nn.Softplus())
        

    from typing import Optional
    from gpytorch.lazy import LazyTensor, MatmulLazyTensor, MulLazyTensor, NonLazyTensor
    from gpytorch.kernels import Kernel
    from gpytorch.settings import trace_mode
    class CustomKernel(Kernel):
        has_lengthscale = True
        def __init__(self, lf, nu: Optional[float] = 2.5, **kwargs):
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
            super(CustomKernel, self).__init__(**kwargs)
            self.nu = nu
            self.feature_extractor = lf
        def forward(self, x1, x2, diag=False, **params):
            sigma1 = self.feature_extractor(x1)
            sigma2 = self.feature_extractor(x2)

            if diag:
                non_var = (sigma1 * sigma2).squeeze(-1)
            else:
                non_var = sigma1 * sigma2.T
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
            return matern_kernel * non_var

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

    



    from gpytorch.constraints import Interval, Positive



    

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lf, lf2):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            # self.base_covar_module = ScaleKernel(CustomKernel(nu=0.5, lf=lf), ard_num_dims=train_x.shape[1])
            self.base_covar_module = ScaleKernel(MaternKernel(nu=0.5), ard_num_dims=train_x.shape[1])
            self.base_covar_module = ScaleKernel(CustomKernelVarNug(nu=0.5, lf=lf), ard_num_dims=train_x.shape[1])

            
            inducing_points=train_x[np.random.choice(train_x.shape[0], 100, replace = False).tolist(), :].clone()
            self.covar_module = InducingPointKernel(
                self.base_covar_module, 
                inducing_points=inducing_points, likelihood=likelihood)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)

            return MultivariateNormal(mean_x, covar_x)


    if torch.cuda.is_available():
        X_train, y_train, X_test, X_validation = X_train.cuda(), y_train.cuda(), X_test.cuda(), X_validation.cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    

    l = LargeFeatureExtractor()
    l2 = LargeFeatureExtractor()
    
    model = GPRegressionModel(X_train, y_train, likelihood, lf=l, lf2=l2)
    # model = GPRegressionModel(X_train, y_train, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    
   
    model.train()
    likelihood.train()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def train():
        t = time.time()
        for i in range(training_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if i % 100 == 0:
                model.eval()
                likelihood.eval()
                l.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
                    preds = model.likelihood(model(X_validation))
                    valid_log_score = -0.5 * torch.sum((y_validation - preds.mean.cpu()) ** 2 / preds.variance.cpu() + torch.log(2 * torch.pi * preds.variance.cpu()))
                    preds = model.likelihood(model(X_test))
                    mse = torch.mean(torch.square(preds.mean.cpu() - y_test))
                    mae = torch.mean(torch.abs(preds.mean.cpu() - y_test))
                    log_score = -0.5 * torch.sum((y_test - preds.mean.cpu()) ** 2 / preds.variance.cpu() + torch.log(2 * torch.pi * preds.variance.cpu()))
                    s = time.time()
                    # print('Time: %.3f Iter: %d LR: %.3f - MSE: %.3f  LS: %.3f  VLS: %.3f' % (s - t, i, learning_rate, mse, log_score, valid_log_score))
                    t = s
                    mses.append(mse)
                    maes.append(mae)
                    log_scores.append(log_score)
                    lrs.append(learning_rate)
                    iterations.append(i)
                    valid_log_scores.append(valid_log_score)
                l.train()
                model.train()
                likelihood.train()
                pass
            optimizer.step()
            torch.cuda.empty_cache()
    start = time.time()
    train()
    end = time.time()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model.likelihood(model(X_test))
        mse = torch.mean(torch.square(preds.mean.cpu() - y_test))
        mae = torch.mean(torch.abs(preds.mean.cpu() - y_test))
        log_score = -0.5 * torch.sum((y_test - preds.mean.cpu()) ** 2 / preds.variance.cpu() + torch.log(2 * torch.pi * preds.variance.cpu()))

    return mses, maes, log_scores, lrs, iterations, valid_log_scores
