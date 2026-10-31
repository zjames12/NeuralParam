# A Framework for Nonstationary Gaussian Processes with Neural Network Parameters

This repository contains the code for *A Framework for Nonstationary Gaussian Processes with Neural Network Parameters* by Zachary James and Joseph Guinness.

The framework models the parameters of a nonstationary kernel as the outputs of a neural network that takes spatial coordinates as input, so the user does not have to pre-specify basis functions. It is built on [GPyTorch](https://gpytorch.ai/) and supports three nonstationary models:

| Model | `nonstationary` option | Kernel |
|---|---|---|
| Stationary Matérn (baseline) | `"none"` | GPyTorch `MaternKernel` |
| Nonstationary variance | `"variance"` | `kernels/matern_nonstat_variance.py` |
| Nonstationary noise | `"noise"` | `kernels/matern_nonstat_nugget.py` |
| Pseudo-nonstationary smoothness | `"nu"` | `kernels/matern_nonstat_nu.py` |

Each nonstationary parameter can be represented by a neural network (`base="neural"`) or by a Gaussian basis function approximation (`base="basis"`).

## Repository layout

| Path | Contents |
|---|---|
| `training_real.py` | Entry point: loads a dataset, then runs cross-validation or a train/test evaluation |
| `fit_real.py` | GP model definitions and training loops |
| `feature_extractor.py` | Neural network for the nonstationary parameters |
| `basis_function.py` | Gaussian basis functions on a grid or a Fibonacci sphere |
| `kernels/`, `functions/` | Nonstationary kernels and coordinate utilities |
| `parameter_estimates.py` | Evaluates a trained model's parameter estimates across a dataset's locations |
| `plot.ipynb` | Notebook used to plot data and parameter estimates |
| `test/` | Unit tests |

## Requirements

The code was last run with Python 3.12, PyTorch 2.9, and GPyTorch 1.15. Install the dependencies with:

```bash
pip install torch gpytorch numpy scipy pandas scikit-learn tensorboard tqdm rich
```

`plot.ipynb` also needs:

```bash
pip install matplotlib seaborn cartopy altair datashader colorcet pillow
```

To run the unit tests:

```bash
python -m unittest discover -s test -t .
```

If Python crashes on macOS with an OpenMP error or a segmentation fault during training, set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`.

## Data

All datasets are publicly available. See the paper for details of each one:

| Paper section | Dataset | Source cited in the paper |
|---|---|---|
| 4.1 | NCAR-LENS surface temperature scaling field | Gerber and Nychka (2021) |
| 4.2 | NCAR-LENS relative humidity scaling field | NCAR Large Ensemble Project |
| 4.3 | Topography Weather minimum temperature, June 1, 2010 | Krock, Kleiber, and Becker (2021) |
| 4.4 | Satellite surface temperature, with its training and test split | Heaton et al. |

The code expects CSV files with a header row and the columns in this order: response, longitude, latitude.

## Training and evaluation

Experiments are configured by editing the code. To run one:

1. Set the CSV paths at the bottom of `training_real.py`. The file currently reads `~/nonstat/heaton_train.csv` and `~/nonstat/heaton_test.csv`.
2. Set the model and training options in the same section.
3. Check the settings in [Matching the paper's configurations](#matching-the-papers-configurations).
4. Run the script:

```bash
python training_real.py
```

Results are printed to the terminal. Train/test runs also write TensorBoard logs and model weights to `runs/experiment_<timestamp>/`. Cross-validation runs (`kfold_cross_validation`) don't save either, so train with `train_and_evaluate` if you need weights for `parameter_estimates.py`. To view the logs:

```bash
tensorboard --logdir runs
```

Commented-out code at the bottom of `training_real.py` shows how to call `kfold_cross_validation`, including the conversion of global coordinates to three dimensions.

### Matching the paper's configurations

**The code is currently set up for double-weight smoothness models, and the variance and noise models will fail until you change it.** The network and basis functions each output two values, which those two kernels can't use. To run the variance or noise models, change the output size from 2 to 1 in these two places:

- `nn.Linear(32, 2)` in `feature_extractor.py`
- `nn.Linear(self.n_basis, 2)` in `basis_function.py`

The other settings used in the paper are:

| Setting | Where | Value |
|---|---|---|
| Network for the variance and noise models (Table 1) | `feature_extractor.py` | Uncomment the 25-25-10 network, which is currently commented out in favor of the deeper one |
| Network for the smoothness model (Table 6) | `feature_extractor.py` | The deeper network, which is the current default |
| Single-weight smoothness | `kernels/matern_nonstat_nu.py` | Uncomment the `w` lines and remove the two-weight versions, then use one output |
| Double-weight smoothness (Appendix A.2) | `kernels/matern_nonstat_nu.py` | The current default |
| Global basis (Sections 4.1 and 4.2) | `fit_real.py` | `sphere=True`, which gives 2500 centers on a Fibonacci sphere |
| Weather station basis (Section 4.3) | `fit_real.py` | `BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24, 53), std=2, sphere=False, ...)`, which gives 1160 functions |
| Satellite basis (Section 4.4) | `fit_real.py` | `BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34, 37.5), std=0.4, sphere=False, ...)`, which gives 2500 functions |

`fit_real.py` defines its basis functions separately in each fitting function, so change the one your run uses. `fit_model_real` handles train/test runs, and `fit_model_real_cv` handles cross-validation.

The paper's models were trained on an NVIDIA A40 GPU with Adam. The code uses a GPU automatically if CUDA is available.

## Parameter estimates and plots

After training, `parameter_estimates.py` writes the estimated parameters for each location to a CSV file. Before running it, edit these values in the file:

- the weights path, pointing to a `trained_model.pth` from `runs/`
- the input CSV and output paths
- the `load_feature_extractor` setup, which must match the trained model

`plot.ipynb` is the working notebook used for plotting. It reads data and results from local paths, so update those before running a cell. Some cells are exploratory and don't correspond to a figure in the paper.

## Results

Each cell shows RMSE and log-score, reproduced from Tables 2–5 of the paper. Runtimes and confidence intervals are in the paper.

| Model | Surface temperature field (Table 2) | Relative humidity field (Table 3) | Weather stations (Table 4) | Satellite temperature (Table 5) |
|---|---|---|---|---|
| Stationary | 3.198, 3.999 | 5.092, −6.977 | 1.457, −1.643 | 2.807, −1.331 |
| Basis variance | 3.199, 4.039 | 4.914, −6.984 | 1.464, −1.647 | 2.505, −1.023 |
| Neural variance | 3.256, 4.372 | 4.290, −5.973 | 1.520, −1.686 | 2.524, −0.990 |
| Basis noise | 3.183, 4.066 | 5.463, −7.278 | 1.471, −1.650 | 3.853, −1.293 |
| Neural noise | 4.466, 4.074 | 5.506, −7.191 | 1.467, −1.608 | 3.570, −1.710 |
| Basis smoothness | 3.178, 4.073 | 4.783, −7.750 | 1.464, −1.647 | 2.713, −1.282 |
| Neural smoothness | 4.326, 3.333 | 4.279, −7.016 | 1.501, −1.669 | 2.904, −1.870 |

Units: surface temperature field RMSE ×10⁻² and log-score ×10³; relative humidity and weather station log-scores ×10³; satellite temperature log-score ×10⁵. Higher log-scores are better.

All models use a Matérn base covariance with ν = 1/2. The two scaling field results use a subset of 10⁴ points with cross-validation (80% training, 20% test). The weather station results also use cross-validation. The satellite results use a 10⁴-point subset of the training set, evaluated on the test set.

## License

See [LICENSE.txt](LICENSE.txt).
