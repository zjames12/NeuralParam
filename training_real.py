from rich.console import Console
from rich.rule import Rule

console = Console()
import gpytorch
import numpy as np
import pandas as pd
import torch
import time
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from fit_real import fit_model_real_cv, fit_model_real, fit_model_real_cv_sgd
from functions.spherical_distances import spherical_coordinates, spherical_coordinates_numpy

np.random.seed(0)
torch.manual_seed(0)

def kfold_cross_validation(df, n_folds=5, training_iterations=2000, learning_rate=0.01,
                          model_name="Model", save_latex=True, output_file="cv_results.tex",
                          nonstationary = "nu", base = "basis", sgd = False, batch_size=256):
    """
    Perform K-fold cross validation and compute average metrics with confidence intervals.

    Args:
        df: DataFrame containing the data
        n_folds: Number of folds for cross validation (default: 5)
        training_iterations: Number of training iterations per fold
        learning_rate: Learning rate for optimizer
        model_name: Name of the model for LaTeX table (default: "Model")
        save_latex: Whether to save results as LaTeX table (default: True)
        output_file: Output file path for LaTeX table (default: "cv_results.tex")

    Returns:
        Dictionary containing mean metrics and confidence intervals
    """

    # Prepare data
    X = df.iloc[:,1:].values  # Features (columns 1-3: x, y, z)
    y = df.iloc[:,0].values    # Target (column 0)

    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Storage for metrics
    rmse_scores = []
    mae_scores = []
    log_scores = []
    runtime_scores = []

    print(f"\n=== Starting {n_folds}-Fold Cross Validation ===")
    print(f"Training iterations per fold: {training_iterations}")
    print(f"Learning rate: {learning_rate}")

    # Perform K-fold CV
    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        # Split data for this fold
        X_train_fold = torch.from_numpy(X[train_idx]).float()
        y_train_fold = torch.from_numpy(y[train_idx]).float()
        X_val_fold = torch.from_numpy(X[val_idx]).float()
        y_val_fold = torch.from_numpy(y[val_idx]).float()

        print(f"Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")

        # Train model and get metrics for this fold
        fold_start_time = time.time()
        if sgd:
            rmse, mae, log_score = fit_model_real_cv_sgd(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                training_iterations=training_iterations,
                learning_rate=learning_rate,
                fold_num=fold,
                nonstationary = nonstationary, base = base,
                batch_size=batch_size
            )
        else:
            rmse, mae, log_score = fit_model_real_cv(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                training_iterations=training_iterations,
                learning_rate=learning_rate,
                fold_num=fold,
                nonstationary = nonstationary, base = base
            )
        fold_runtime = time.time() - fold_start_time

        # Store metrics
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        log_scores.append(log_score)
        runtime_scores.append(fold_runtime)

        print(f"Fold {fold} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, Log-Score: {log_score:.6f}, Runtime: {fold_runtime:.2f}s")

        fold += 1

    # Convert to numpy arrays for easier computation
    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    log_scores = np.array(log_scores)
    runtime_scores = np.array(runtime_scores)

    # Calculate statistics
    def calculate_stats(scores):
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)  # Sample standard deviation
        se_score = std_score / np.sqrt(len(scores))  # Standard error

        # 95% confidence interval using t-distribution
        t_value = stats.t.ppf(0.975, df=len(scores)-1)  # 97.5th percentile for 95% CI
        half_width = t_value * se_score
        ci_lower = mean_score - half_width
        ci_upper = mean_score + half_width

        return {
            'mean': mean_score,
            'std': std_score,
            'se': se_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'half_width': half_width,
            'individual_scores': scores.tolist()
        }

    # Calculate statistics for each metric
    results = {
        'rmse': calculate_stats(rmse_scores),
        'mae': calculate_stats(mae_scores),
        'log_score': calculate_stats(log_scores),
        'runtime': calculate_stats(runtime_scores),
        'n_folds': n_folds
    }

    # Print summary
    print(f"\n=== {n_folds}-Fold Cross Validation Results ===")
    print(f"Training iterations per fold: {training_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"RMSE: {results['rmse']['mean']:.6f} ± {results['rmse']['half_width']:.6f}")
    print(f"  95% CI: [{results['rmse']['ci_lower']:.6f}, {results['rmse']['ci_upper']:.6f}]")
    print(f"  Individual scores: {[f'{x:.6f}' for x in results['rmse']['individual_scores']]}")

    print(f"MAE:  {results['mae']['mean']:.6f} ± {results['mae']['half_width']:.6f}")
    print(f"  95% CI: [{results['mae']['ci_lower']:.6f}, {results['mae']['ci_upper']:.6f}]")
    print(f"  Individual scores: {[f'{x:.6f}' for x in results['mae']['individual_scores']]}")

    print(f"Log-Score: {results['log_score']['mean']:.6f} ± {results['log_score']['half_width']:.6f}")
    print(f"  95% CI: [{results['log_score']['ci_lower']:.6f}, {results['log_score']['ci_upper']:.6f}]")
    print(f"  Individual scores: {[f'{x:.6f}' for x in results['log_score']['individual_scores']]}")

    print(f"Runtime: {results['runtime']['mean']:.2f}s ± {results['runtime']['half_width']:.2f}s")
    print(f"  95% CI: [{results['runtime']['ci_lower']:.2f}s, {results['runtime']['ci_upper']:.2f}s]")
    print(f"  Individual scores: {[f'{x:.2f}s' for x in results['runtime']['individual_scores']]}")

    # Save results as LaTeX table if requested
    if save_latex:
        save_latex_table(results, model_name, output_file)
        print(f"\nLaTeX table saved to: {output_file}")

    return results

def save_latex_table(results, model_name, output_file):
    """
    Save cross-validation results as a LaTeX table.

    Args:
        results: Dictionary containing CV results
        model_name: Name of the model for the table
        output_file: Output file path
    """
    # Extract metrics with confidence intervals
    rmse_mean = results['rmse']['mean']
    rmse_ci = results['rmse']['half_width']

    mae_mean = results['mae']['mean']
    mae_ci = results['mae']['half_width']

    log_mean = results['log_score']['mean']
    log_ci = results['log_score']['half_width']

    runtime_mean = results['runtime']['mean']
    runtime_ci = results['runtime']['half_width']

    # Create LaTeX table
    latex_table = f"""\\begin{{table}}
  \\centering
  \\begin{{tabular}}{{lllll}}
    \\toprule
    Model                         & RMSE  & MAE & Log-score & Time (s) \\\\
    \\midrule
    {model_name:<29} & ${rmse_mean:.3f} \\pm {rmse_ci:.3f}$     & ${mae_mean:.3f} \\pm {mae_ci:.3f}$     & ${log_mean:.3f} \\pm {log_ci:.5f}$      & ${runtime_mean:.0f}\\pm{runtime_ci:.1f}$      \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_table)

    return latex_table

def train_and_evaluate(train_df, test_df, training_iterations=2000, learning_rate=0.005,
                       log_dir=None, nonstationary = "nu", base = "basis", sgd = False, batch_size=256,
                       sphere = False):
    """
    Fit a model on train_df and evaluate on test_df, printing RMSE, log-score, and fit time.

    Args:
        train_df: DataFrame with target in column 0 and features in remaining columns
        test_df:  DataFrame with the same layout as train_df
        training_iterations: Number of training iterations
        learning_rate: Learning rate for the Adam optimizer
        log_dir: Directory for TensorBoard logs (passed through to fit_model_real)

    Returns:
        Tuple of (model, likelihood) from fit_model_real
    """
    X_train = torch.from_numpy(train_df.iloc[:, 1:].values).float()
    y_train = torch.from_numpy(train_df.iloc[:, 0].values).float()
    X_test  = torch.from_numpy(test_df.iloc[:, 1:].values).float()
    y_test  = torch.from_numpy(test_df.iloc[:, 0].values).float()

    print(f"\n=== Train / Test Evaluation ===")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Training iterations: {training_iterations}, Learning rate: {learning_rate}")

    start_time = time.time()
    if sgd:
        model, likelihood = fit_model_real_cv_sgd(
            X_train, y_train, X_test, y_test,
            training_iterations=training_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size,
            nonstationary = nonstationary, base = base
        )
    else:
        model, likelihood = fit_model_real(
            X_train, y_train, X_test, y_test,
            training_iterations=training_iterations,
            learning_rate=learning_rate,
            nonstationary = nonstationary, base = base,
            sphere = sphere
        )
    fit_time = time.time() - start_time

    # Compute metrics on test set
    # model.eval()
    # likelihood.eval()
    # with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
    #     test_output = model(X_test)
    #     y_pred = likelihood(test_output).mean
    #     test_dist = likelihood(test_output)

    #     rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
    #     log_score = (-0.5 * torch.sum(
    #         (y_test - y_pred) ** 2 / test_dist.variance
    #         + torch.log(2 * torch.pi * test_dist.variance)
    #     )).item()

    # print(f"\n=== Results ===")
    # print(f"Fit time:  {fit_time:.2f}s ({fit_time / 60:.2f} min)")
    # print(f"RMSE:      {rmse:.6f}")
    # print(f"Log-score: {log_score:.6f}")

    return model, likelihood


# Load and preprocess data
# df = pd.read_csv("braceu.csv", header = 0)
# df = df.sample(n=10000, random_state=0).reset_index(drop=True)
# print("braceu")
# # Convert last two columns to spherical coordinates (3 columns)
# spherical_coords = spherical_coordinates_numpy(df.iloc[:,1:3].values)
# df = df.drop(df.columns[1:3], axis=1)  # Remove last two columns
# spherical_df = pd.DataFrame(spherical_coords, columns=['x', 'y', 'z'])
# df = pd.concat([df, spherical_df], axis=1)


# for nonstationary in ["nu"]:
#     for base in ["neural", "basis"]:
#         console.print(Rule(f"Testing {nonstationary} with {base} base", style="bold cyan"))
#         cv_results = kfold_cross_validation(df, n_folds=5, training_iterations=2000, learning_rate=0.005,
#                     model_name="Neural Parametric", output_file="neural_param_cv_results.tex",
#                     nonstationary = nonstationary, base = base, sgd = False, batch_size=10000)
        # train_and_evaluate(df, df, training_iterations=2000, learning_rate=0.005,
        #             nonstationary = nonstationary, base = base, sgd = False, batch_size=10000, sphere=True)
        
# cv_results = kfold_cross_validation(df, n_folds=5, training_iterations=4000, learning_rate=0.005,
#                     model_name="Neural Parametric", output_file="neural_param_cv_results.tex",
#                     nonstationary = "noise", base = "neural", sgd = False, batch_size=10000)

train_df = pd.read_csv("~/nonstat/heaton_train.csv", header = 0)
train_df = train_df.sample(n=10000, random_state=0).reset_index(drop=True)
test_df = pd.read_csv("~/nonstat/heaton_test.csv", header = 0)
for nonstationary in ["nu"]:#, "variance", "none", "nu", "noise"]:
    for base in ["basis", "neural"]:
        console.print(Rule(f"Testing {nonstationary} with {base} base", style="bold cyan"))
        try:  
            cv_results = train_and_evaluate(train_df, test_df, training_iterations=20000, learning_rate=0.0001,
                        nonstationary = nonstationary, base = base, sgd = False, batch_size=10000)
        except Exception as e:
            print(f"Error for {nonstationary} with {base} base: {e}")