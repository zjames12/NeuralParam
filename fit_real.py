# Simplified version of the neural parametrization framework
# This version only uses training data and logs essential training metrics

import torch
import gpytorch
import numpy as np
import time
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from basis_function import BasisFunction
from kernels.matern_nonstat_nu import MaternNonstatNuKernel
from kernels.matern_nonstat_nugget import MaternNonstatNuggetKernel
from kernels.matern_nonstat_variane_nugget import CustomKernelVarNug
from kernels.matern_nonstat_variance import MaternNonstatVarianceKernel
from kernels.matern_nonstat_lengthscale import MaternNonstatLengthscaleKernel
from torch.utils.tensorboard import SummaryWriter
from feature_extractor import LargeFeatureExtractor

from tqdm import tqdm


def fit_model_real(X_train, y_train, X_test, y_test, training_iterations=800, learning_rate=0.05, 
                   log_dir=None, nonstationary = "nu", base = "basis", sphere = False):
    """
    Simplified fit_model function that uses training data and evaluates on test data with deep feature extractor.

    Args:
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        training_iterations: Number of training iterations (default: 800)
        learning_rate: Learning rate for optimizer (default: 0.05)
        log_dir: Directory for TensorBoard logs

    Returns:
        Trained model and likelihood
    """

    # Set GPytorch global settings for numerical stability
    # gpytorch.settings.cholesky_jitter._set_value(1e-2, 1e-2)  # Increase from default 1e-6 (double_value, half_value)
    # gpytorch.settings.max_cholesky_size._set_value(2000, 2000)

    # # Additional numerical stability settings
    # gpytorch.settings.min_preconditioning_size._set_value(0, 0)  # Always use preconditioning
    # gpytorch.settings.max_eager_kernel_size._set_value(100, 100)   # Use lazy tensors for stability

    # Initialize TensorBoard writer
    if log_dir is None:
        log_dir = f'runs/experiment_{int(time.time())}'
    writer = SummaryWriter(log_dir)

    np.random.seed(0)
    torch.manual_seed(0)

    n_dim = X_train.shape[1]

    # Feature extractors will be instantiated below

    # GP Model definition
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lf):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()

            if nonstationary == "nu":
                self.covar_module = ScaleKernel(MaternNonstatNuKernel(lf=lf))
            elif nonstationary == "noise":
                self.covar_module = ScaleKernel(MaternNonstatNuggetKernel(lf=lf, nu=0.5))
            elif nonstationary == "variance":
                self.covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)
            elif nonstationary == "none":
                self.covar_module = ScaleKernel(MaternKernel(nu=0.5))
            else:
                raise ValueError(f"Invalid nonstationary type: {nonstationary}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    # Move to GPU if available
    if torch.cuda.is_available():
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_test, y_test = X_test.cuda(), y_test.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # Initialize feature extractors and model
    

    if base == "basis":
        # lf = BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24,53), std=2, sphere=False, sigmoid = nonstationary == "nu")
        # lf = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.1, sphere=False, sigmoid = nonstationary == "nu")
        if sphere:
            lf = BasisFunction(q=84, p=42, x_bounds=(-180, 180), y_bounds=(-90,90), std=40/360, sphere=sphere, sigmoid = (nonstationary == "nu"))
        else:
            lf = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.4, sphere=False, sigmoid = nonstationary == "nu")
    elif base == "neural":
        lf = LargeFeatureExtractor(n_dim, sigmoid = nonstationary == "nu")
    model = GPRegressionModel(X_train, y_train, likelihood, lf=lf)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Set to training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Marginal log likelihood for GP training
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def evaluate_test_metrics(iteration, elapsed_time):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            test_output = model(X_test)
            y_predicted = likelihood(test_output).mean
            mse = torch.mean((y_predicted - y_test) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(y_predicted - y_test))
            test_distribution = likelihood(test_output)
            log_score = -0.5 * torch.sum((y_test - y_predicted) ** 2 / test_distribution.variance + torch.log(2 * torch.pi * test_distribution.variance))
        model.train()
        likelihood.train()
        writer.add_scalar('Test/RMSE', rmse.item(), iteration)
        writer.add_scalar('Test/MAE', mae.item(), iteration)
        writer.add_scalar('Test/LogScore', log_score.item(), iteration)
        print(f"  [Iter {iteration}] Time: {elapsed_time:.1f}s  Test RMSE: {rmse.item():.6f}  MAE: {mae.item():.6f}  Log-Score: {log_score.item():.6f}")

    def train():
        start_time = time.time()
        for i in tqdm(range(training_iterations)):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(X_train)

            # Calculate loss
            loss = -mll(output, y_train)
            loss.backward()

            # Calculate progress metrics
            elapsed_time = time.time() - start_time
            iterations_remaining = training_iterations - (i + 1)
            completion_percentage = ((i + 1) / training_iterations) * 100

            # Log training loss and progress metrics every iteration
            # writer.add_scalar('Loss/Train', loss.item(), i)
            # writer.add_scalar('Progress/Iteration', i + 1, i)
            # writer.add_scalar('Progress/Iterations_Remaining', iterations_remaining, i)
            # writer.add_scalar('Progress/Completion_Percentage', completion_percentage, i)
            # writer.add_scalar('Time/Elapsed_Seconds', elapsed_time, i)

            # Estimate remaining time based on current pace
            if i % 1000 == 0:
                avg_time_per_iter = elapsed_time / (i + 1)
                estimated_remaining_time = avg_time_per_iter * iterations_remaining
                writer.add_scalar('Time/Estimated_Remaining_Seconds', estimated_remaining_time, i)

            # Periodic logging of additional metrics
            if i % 1000 == 0:
                # Log learning rate
                writer.add_scalar('Hyperparameters/LearningRate', learning_rate, i)

                # Log model parameters for monitoring
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Parameters/{name}', param.data, i)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad.data, i)

                print(f'Time: {elapsed_time:.1f}s Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f} - ETA: {estimated_remaining_time/60:.1f}m' if i > 0 else f'Time: {elapsed_time:.1f}s Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f}')

            if (i + 1) % 2000 == 0:
                evaluate_test_metrics(i + 1, elapsed_time)

            optimizer.step()
            # torch.cuda.empty_cache()

    # Train the model
    training_start = time.time()
    train()
    training_end = time.time()

    # Log training duration
    writer.add_scalar('Training/Duration_Seconds', training_end - training_start, training_iterations)

    # Print training time
    training_duration = training_end - training_start
    print(f"\n--- Training Complete ---")
    print(f"Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    # Log final loss
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        final_output = model(X_train)
        final_loss = -mll(final_output, y_train)
        writer.add_scalar('Final/Loss', final_loss.item(), training_iterations)

    # Save feature extractor weights
    # torch.save(lf.state_dict(), f'{log_dir}/feature_extractor_weights.pth')

    # Save trained model and likelihood for later predictions
    torch.save(model.state_dict(), f'{log_dir}/trained_model.pth')
    torch.save(likelihood.state_dict(), f'{log_dir}/likelihood.pth')

    # Close TensorBoard writer
    writer.close()

    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")
    print(f"Final likelihood noise: {likelihood.noise.item():.6f}")
    # print(f"Final lengthscale: {model.covar_module.lengthscale.item():.6f}")
    # print(f"Final lengthscale: {model.covar_module.base_kernel.lengthscale.item():.6f}")
    # print(f"Final lengthscale: {model.covar_module.base_kernel.base_kernel.lengthscale.item():.6f}")

    # print(f"Final scale parameter: {model.covar_module.base_kernel.outputscale.item():.6f}")

    # Test evaluation
    print("\n--- Test Evaluation ---")
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        # Generate predictions
        test_output = model(X_test)
        y_predicted = likelihood(test_output).mean

        # Calculate RMSE
        mse = torch.mean((y_predicted - y_test) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(y_predicted - y_test))

        # Calculate log-score (negative log-likelihood)
        test_distribution = likelihood(test_output)
        log_score = -0.5 * torch.sum((y_test - y_predicted) ** 2 / test_distribution.variance + torch.log(2 * torch.pi * test_distribution.variance))

        print(f"Test RMSE: {rmse.item():.6f}")
        print(f"Test Log-Score: {log_score.item():.6f}")
        print(f"Test MAE: {mae.item():.6f}")
        print(f"Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    return model, likelihood


def fit_model_real_sgd(X_train, y_train, X_test, y_test, training_iterations=800,
                       learning_rate=0.05, log_dir=None, batch_size=256,
                       nonstationary = "nu", base = "basis"):
    """
    Minibatch variant of fit_model_real. Each iteration trains on a random
    minibatch of the training data instead of the full dataset.

    Args:
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        training_iterations: Number of training iterations (default: 800)
        learning_rate: Learning rate for optimizer (default: 0.05)
        log_dir: Directory for TensorBoard logs
        batch_size: Number of samples per minibatch (default: 256)

    Returns:
        Trained model and likelihood
    """

    # gpytorch.settings.cholesky_jitter._set_value(1e-2, 1e-2)
    # gpytorch.settings.max_cholesky_size._set_value(2000, 2000)
    # gpytorch.settings.min_preconditioning_size._set_value(0, 0)
    # gpytorch.settings.max_eager_kernel_size._set_value(100, 100)

    if log_dir is None:
        log_dir = f'runs/experiment_{int(time.time())}'
    writer = SummaryWriter(log_dir)

    np.random.seed(0)
    torch.manual_seed(0)

    n_dim = X_train.shape[1]

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lf):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            # self.base_covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)

            # inducing_points = train_x[np.random.choice(train_x.shape[0], 1000, replace=False).tolist(), :].clone()
            # self.covar_module = InducingPointKernel(
            #     self.base_covar_module,
            #     inducing_points=inducing_points,
            #     likelihood=likelihood)

            if nonstationary == "nu":
                self.covar_module = ScaleKernel(MaternNonstatNuKernel(lf=lf))
            elif nonstationary == "noise":
                self.covar_module = MaternNonstatNuggetKernel(lf=lf, nu=0.5)
            elif nonstationary == "variance":
                self.covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)
            elif nonstationary == "none":
                self.covar_module = ScaleKernel(MaternKernel(nu=0.5))
            else:
                raise ValueError(f"Invalid nonstationary type: {nonstationary}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    if torch.cuda.is_available():
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_test, y_test = X_test.cuda(), y_test.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if base == "basis":
        lf = BasisFunction(q=50, p=50, x_bounds=(-180, 180), y_bounds=(-90, 90), std=10.0)
    elif base == "neural":
        lf = LargeFeatureExtractor(n_dim)
    else:
        raise ValueError(f"Invalid base type: {base}")

    model = GPRegressionModel(X_train, y_train, likelihood, lf=lf)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_train = X_train.shape[0]
    actual_batch_size = min(batch_size, n_train)

    def train():
        start_time = time.time()
        for i in tqdm(range(training_iterations)):
            idx = torch.randperm(n_train)[:actual_batch_size]
            X_batch, y_batch = X_train[idx], y_train[idx]

            model.set_train_data(inputs=X_batch, targets=y_batch, strict=False)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch)
            loss.backward()

            elapsed_time = time.time() - start_time
            iterations_remaining = training_iterations - (i + 1)
            completion_percentage = ((i + 1) / training_iterations) * 100

            writer.add_scalar('Loss/Train', loss.item(), i)
            writer.add_scalar('Progress/Iteration', i + 1, i)
            writer.add_scalar('Progress/Iterations_Remaining', iterations_remaining, i)
            writer.add_scalar('Progress/Completion_Percentage', completion_percentage, i)
            writer.add_scalar('Time/Elapsed_Seconds', elapsed_time, i)

            if i > 0:
                avg_time_per_iter = elapsed_time / (i + 1)
                estimated_remaining_time = avg_time_per_iter * iterations_remaining
                writer.add_scalar('Time/Estimated_Remaining_Seconds', estimated_remaining_time, i)

            if i % 100 == 0:
                writer.add_scalar('Hyperparameters/LearningRate', learning_rate, i)

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Parameters/{name}', param.data, i)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad.data, i)

                print(f'Time: {elapsed_time:.1f}s Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f} - ETA: {estimated_remaining_time/60:.1f}m' if i > 0 else f'Time: {elapsed_time:.1f}s Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f}')

            optimizer.step()
            torch.cuda.empty_cache()

    training_start = time.time()
    train()
    training_end = time.time()

    writer.add_scalar('Training/Duration_Seconds', training_end - training_start, training_iterations)

    training_duration = training_end - training_start
    print(f"\n--- Training Complete ---")
    print(f"Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    # Restore full training data for final loss evaluation
    model.set_train_data(inputs=X_train, targets=y_train, strict=False)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        final_output = model(X_train)
        final_loss = -mll(final_output, y_train)
        writer.add_scalar('Final/Loss', final_loss.item(), training_iterations)

    torch.save(l.state_dict(), f'{log_dir}/feature_extractor_weights.pth')
    torch.save(model.state_dict(), f'{log_dir}/trained_model.pth')
    torch.save(likelihood.state_dict(), f'{log_dir}/likelihood.pth')

    writer.close()

    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")
    print(f"Final likelihood noise: {likelihood.noise.item():.6f}")

    print("\n--- Test Evaluation ---")
    with torch.no_grad():
        test_output = model(X_test)
        y_predicted = likelihood(test_output).mean

        mse = torch.mean((y_predicted - y_test) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(y_predicted - y_test))

        test_distribution = likelihood(test_output)
        log_score = -0.5 * torch.sum((y_test - y_predicted) ** 2 / test_distribution.variance + torch.log(2 * torch.pi * test_distribution.variance))

        print(f"Test RMSE: {rmse.item():.6f}")
        print(f"Test Log-Score: {log_score.item():.6f}")
        print(f"Test MAE: {mae.item():.6f}")

    return model, likelihood


def fit_model_real_cv(X_train, y_train, X_val, y_val, training_iterations=800, 
                      learning_rate=0.05, fold_num=None,
                      nonstationary = "nu", base = "basis"):
    """
    Simplified version of fit_model_real for cross-validation.
    Trains a model on training data and evaluates on validation data, returning metrics.

    Args:
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data
        y_val: Validation target data
        training_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        fold_num: Fold number for logging (optional)

    Returns:
        Tuple of (rmse, mae, log_score) metrics on validation data
    """

    # Set GPytorch global settings for numerical stability
    # gpytorch.settings.cholesky_jitter._set_value(1e-2, 1e-2)
    # gpytorch.settings.max_cholesky_size._set_value(2000, 2000)
    # gpytorch.settings.min_preconditioning_size._set_value(0, 0)
    # gpytorch.settings.max_eager_kernel_size._set_value(100, 100)

    np.random.seed(fold_num if fold_num is not None else 0)
    torch.manual_seed(fold_num if fold_num is not None else 0)

    n_dim = X_train.shape[1]

    # GP Model definition (same as in fit_model_real)
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lf):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            # self.base_covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)
            # self.base_covar_module = MaternNonstatNuKernel(lf=lf, nu=0.5)#, ard_num_dims=train_x.shape[1])
            # self.base_covar_module = ScaleKernel(MaternKernel(nu=0.5))            

            # inducing_points = train_x[np.random.choice(train_x.shape[0], min(500, train_x.shape[0]), replace=False).tolist(), :].clone()
            # self.covar_module = InducingPointKernel(
            #     self.base_covar_module,
            #     inducing_points=inducing_points,
            #     likelihood=likelihood)

            # self.covar_module = ScaleKernel(MaternKernel(nu=0.5))     
            # self.covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)
            # self.covar_module = ScaleKernel(MaternNonstatNuggetKernel(lf=lf, nu=0.5))
            if nonstationary == "nu":
                self.covar_module = ScaleKernel(MaternNonstatNuKernel(lf=lf))
            elif nonstationary == "noise":
                self.covar_module = ScaleKernel(MaternNonstatNuggetKernel(lf=lf, nu=0.5))
            elif nonstationary == "variance":
                self.covar_module = MaternNonstatVarianceKernel(lf=lf, nu=0.5)
            elif nonstationary == "none":
                self.covar_module = ScaleKernel(MaternKernel(nu=0.5))
            else:
                raise ValueError(f"Invalid nonstationary type: {nonstationary}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    # Move to GPU if available
    if torch.cuda.is_available():
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_val, y_val = X_val.cuda(), y_val.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Initialize feature extractors and model
    if base == "neural":
        l = LargeFeatureExtractor(n_dim)
    elif base == "basis":
        # l = BasisFunction(q=84, p=42, x_bounds=(-180, 180), y_bounds=(-90,90), std=40/360, sphere=True, sigmoid = nonstationary == "nu")
    #    l = BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24,53), std=2, sphere=False, sigmoid = nonstationary == "nu")
        l = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.4, sphere=False, sigmoid = nonstationary == "nu"   )
    else:
        raise ValueError(f"Unknown base type: {base}")

    # l = BasisFunction(q=12, p=8, x_bounds=(-125, -65), y_bounds=(24,53), std=6)
    # l = BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24,53), std=2)
    # l = BasisFunction(q=40, p=40, x_bounds=(0, 225), y_bounds=(0,150), std=2)
    # l = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.1)

    # l = BasisFunction(q=84, p=42, x_bounds=(-180, 180), y_bounds=(-90,90), std=40/360)
    model = GPRegressionModel(X_train, y_train, likelihood, lf=l)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Set to training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Marginal log likelihood for GP training
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop (simplified, no logging)
    start_time = time.time()
    for i in range(training_iterations):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(X_train)

        # Calculate loss
        loss = -mll(output, y_train)
        loss.backward()

        # Print progress every 1000 iterations
        if i % 1000 == 0:
            elapsed_time = time.time() - start_time
            completion_percentage = ((i + 1) / training_iterations) * 100
            print(f'  Fold {fold_num} - Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f} - Time: {elapsed_time:.1f}s')

        optimizer.step()
        torch.cuda.empty_cache()

    training_duration = time.time() - start_time
    print(f'  Fold {fold_num} training completed in {training_duration:.2f} seconds')

    # Evaluation on validation data
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        # Generate predictions on validation data
        val_output = model(X_val)
        y_predicted = likelihood(val_output).mean

        # Calculate RMSE
        mse = torch.mean((y_predicted - y_val) ** 2)
        rmse = torch.sqrt(mse)

        # Calculate MAE
        mae = torch.mean(torch.abs(y_predicted - y_val))

        # Calculate log-score (negative log-likelihood)
        val_distribution = likelihood(val_output)
        log_score = -0.5 * torch.sum((y_val - y_predicted) ** 2 / val_distribution.variance + torch.log(2 * torch.pi * val_distribution.variance))

        # Convert to CPU and numpy if necessary
        rmse_val = rmse.cpu().item() if rmse.is_cuda else rmse.item()
        mae_val = mae.cpu().item() if mae.is_cuda else mae.item()
        log_score_val = log_score.cpu().item() if log_score.is_cuda else log_score.item()

    return rmse_val, mae_val, log_score_val


def fit_model_real_cv_sgd(X_train, y_train, X_val, y_val, training_iterations=800,
                          learning_rate=0.05, fold_num=None,
                          nonstationary="nu", base="basis", batch_size=256):
    """
    Minibatch variant of fit_model_real_cv. Each iteration trains on a random
    minibatch of the training data instead of the full dataset.

    Args:
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data
        y_val: Validation target data
        training_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        fold_num: Fold number for logging (optional)
        nonstationary: Kernel type (default: "nu")
        base: Feature extractor type (default: "basis")
        batch_size: Number of samples per minibatch (default: 256)

    Returns:
        Tuple of (rmse, mae, log_score) metrics on validation data
    """

    np.random.seed(fold_num if fold_num is not None else 0)
    torch.manual_seed(fold_num if fold_num is not None else 0)

    n_dim = X_train.shape[1]

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lf):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            if nonstationary == "nu":
                self.covar_module = ScaleKernel(MaternNonstatNuKernel(lf=lf))
            elif nonstationary == "noise":
                self.covar_module = ScaleKernel(MaternNonstatNuggetKernel(lf=lf))
            elif nonstationary == "var":
                self.covar_module = ScaleKernel(MaternNonstatVarianceKernel(lf=lf))
            else:
                raise ValueError(f"Unknown nonstationary type: {nonstationary}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    if torch.cuda.is_available():
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_val, y_val = X_val.cuda(), y_val.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if base == "neural":
        l = LargeFeatureExtractor(n_dim)
    elif base == "basis":
        l = BasisFunction(q=84, p=42, x_bounds=(-180, 180), y_bounds=(-90, 90), std=40/360)        
    else:
        raise ValueError(f"Unknown base type: {base}")

    model = GPRegressionModel(X_train, y_train, likelihood, lf=l)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_train = X_train.shape[0]
    actual_batch_size = min(batch_size, n_train)

    start_time = time.time()
    for i in range(training_iterations):
        idx = torch.randperm(n_train)[:actual_batch_size]
        X_batch, y_batch = X_train[idx], y_train[idx]

        model.set_train_data(inputs=X_batch, targets=y_batch, strict=False)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = -mll(output, y_batch)
        loss.backward()

        if i % 1000 == 0:
            elapsed_time = time.time() - start_time
            completion_percentage = ((i + 1) / training_iterations) * 100
            print(f'  Fold {fold_num} - Iter: {i+1}/{training_iterations} ({completion_percentage:.1f}%) - Loss: {loss.item():.3f} - Time: {elapsed_time:.1f}s')

        optimizer.step()
        torch.cuda.empty_cache()

    training_duration = time.time() - start_time
    print(f'  Fold {fold_num} training completed in {training_duration:.2f} seconds')

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        val_output = model(X_val)
        y_predicted = likelihood(val_output).mean

        mse = torch.mean((y_predicted - y_val) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(y_predicted - y_val))

        val_distribution = likelihood(val_output)
        log_score = -0.5 * torch.sum((y_val - y_predicted) ** 2 / val_distribution.variance + torch.log(2 * torch.pi * val_distribution.variance))

        rmse_val = rmse.cpu().item() if rmse.is_cuda else rmse.item()
        mae_val = mae.cpu().item() if mae.is_cuda else mae.item()
        log_score_val = log_score.cpu().item() if log_score.is_cuda else log_score.item()

    return rmse_val, mae_val, log_score_val