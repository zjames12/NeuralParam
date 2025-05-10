# The hierarchical nonstationary variance model fit with variational inference

import torch
import gpytorch
import numpy as np
import pandas as pd
import time
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

def fit_svgp(X_train, y_train, X_validation, y_validation, X_test, y_test, training_iterations = 800, learning_rate = 0.05):

    valid_log_scores = []
    lrs = []
    iterations = []
    mses = []
    maes = []
    log_scores = []

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        X_train, y_train, X_test, y_test, X_validation, y_validation = X_train.cuda(), y_train.cuda(), X_test.cuda(), y_test.cuda(), X_validation.cuda(), y_validation.cuda()

    class HiddenLayer(DeepGPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=100, mean_type='constant'):
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape
            )

            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )

            super(HiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

            self.mean_module = ConstantMean(batch_shape=batch_shape)#LinearMean(input_dims)
            self.covar_module = ScaleKernel(
                MaternKernel(nu=0.5,batch_shape=batch_shape, ard_num_dims=None),
                batch_shape=batch_shape, ard_num_dims=None
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

        def __call__(self, x, *other_inputs, **kwargs):
            if len(other_inputs):
                if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                    x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    for inp in other_inputs
                ]

                x = torch.cat([x] + processed_inputs, dim=-1)

            return super().__call__(x, are_samples=bool(len(other_inputs)))
    class OutputLayer(DeepGPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=100, mean_type='constant'):
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])

            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape
            )

            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )

            super(OutputLayer, self).__init__(variational_strategy, input_dims, output_dims)

            self.mean_module = ConstantMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=None),
                batch_shape=batch_shape, ard_num_dims=input_dims-1
            )

        def forward(self, x):
            mean_x = self.mean_module(x[:,:,1:])
            z = torch.exp(x[:,:,0])
            z = torch.matmul(z.unsqueeze(2), z.unsqueeze(1))
            covar_x = self.covar_module(x[:,:,1:]) * z
            return MultivariateNormal(mean_x, covar_x)

        def __call__(self, x, *other_inputs, **kwargs):
            """
            Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
            easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
            hidden layer's outputs and the input data to hidden_layer2.
            """
            if len(other_inputs):
                if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                    x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    for inp in other_inputs
                ]

                x = torch.cat([x] + processed_inputs, dim=-1)

            return super().__call__(x, are_samples=bool(len(other_inputs)))

    num_hidden_dims = 1
    class DeepGP(gpytorch.models.deep_gps.DeepGP):
        def __init__(self, train_x_shape):
            hidden_layer = HiddenLayer(
                input_dims=train_x_shape[-1],
                output_dims=num_hidden_dims,
                mean_type='linear',
            )

            last_layer = OutputLayer(
                input_dims=hidden_layer.output_dims+hidden_layer.input_dims,
                output_dims=None,
                mean_type='constant',
            )

            super().__init__()

            self.hidden_layer = hidden_layer
            self.last_layer = last_layer
            self.likelihood = GaussianLikelihood()

        def forward(self, inputs):
            hidden_rep1 = self.hidden_layer(inputs)
            output = self.last_layer(hidden_rep1, inputs)
            return output

        def predict(self, test_loader):
            with torch.no_grad():
                mus = []
                variances = []
                lls = []
                for x_batch, y_batch in test_loader:
                    preds = self.likelihood(self(x_batch))
                    mus.append(preds.mean)
                    variances.append(preds.variance)
                    lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

            return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

    model = DeepGP(X_train.shape)
    if torch.cuda.is_available():
        model = model.cuda()

    num_epochs = training_iterations
    num_samples = 20

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=learning_rate)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024)

    validation_dataset = TensorDataset(X_validation, y_validation)
    validation_loader = DataLoader(validation_dataset, batch_size=1024)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    j = 0
    epochs_iter = range(num_epochs)
    for i in epochs_iter:
        # print(i)
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = train_loader
        for x_batch, y_batch in minibatch_iter:
            # print(j)
            # j += 1
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
        model.eval()
   
        
        with torch.no_grad():
            predictive_means, predictive_variances, test_lls = model.predict(validation_loader)
            predictive_means, predictive_variances = predictive_means.mean(0), predictive_variances.mean(0)
            valid_log_score = -0.5 * torch.sum((y_validation - predictive_means) ** 2 / predictive_variances + torch.log(2 * torch.pi * predictive_variances))
            # valid_log_score = torch.mean(torch.square(predictive_means - y_validation))
            predictive_means, predictive_variances, test_lls = model.predict(test_loader)
            predictive_means, predictive_variances = predictive_means.mean(0), predictive_variances.mean(0)
            mse = torch.mean(torch.square(predictive_means - y_test))
            mae = torch.mean(torch.abs(predictive_means - y_test))
            log_score = -0.5 * torch.sum((y_test - predictive_means) ** 2 / predictive_variances + torch.log(2 * torch.pi * predictive_variances))
            # print('Iter: %d LR: %.3f - MSE: %.3f  LS: %.3f  VLS: %.3f' % (i, learning_rate, mse, log_score, valid_log_score))
            maes.append(mae.cpu())
            mses.append(mse.cpu())
            log_scores.append(log_score.cpu())
            lrs.append(learning_rate)
            iterations.append(i)
            valid_log_scores.append(valid_log_score.cpu())
        model.train()
        # likelihood.train()
        # print(i, learning_rate, mse.item(), mae.item(), log_score.item(), valid_log_score.item())
    dataset = TensorDataset(torch.cat((X_train,X_validation,X_test), dim=0),torch.cat((y_train,y_validation,y_test), dim=-1))
    loader = DataLoader(dataset, batch_size=1024)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get the GP distribution at X_train
        
        dfs = []
        for x_batch, y_batch in loader:
            hidden_dist = model.hidden_layer(x_batch)
            # Sample from the distribution
            samples = hidden_dist.rsample()
            res = torch.mean(samples, dim=0).cpu()
            locs = x_batch.cpu()
            sf = torch.cat((res, locs), dim=1)
            dfs.append(pd.DataFrame(sf))
        
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv("vi_var_preds.csv",index=False)
        return mses, maes, log_scores, lrs, iterations, valid_log_scores