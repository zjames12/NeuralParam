# A Framework for Nonstationary Gaussian Processes with Neural Network Parameters

This repository is the official implementation of *A Framework for Nonstationary Gaussian Processes with Neural Network Parameters*. 


## Requirements

Running this code requires installing [PyTorch](https://pytorch.org/get-started/locally/), [GPyTorch](https://gpytorch.ai/), Pandas, and scikit-learn.

## Training

To train and evaluate the models in the paper run:

```train
python training.py
```

The data source and model can be specified by modifiying the file.

## Data

Datasets can be found [here](https://github.com/treforevans/uci_datasets/tree/master).

## Results

Our models achieves the following performance on several UCI datasets:

### UCI Datasets

RMSE of two nonstationary models under our framework (NNP var, NNP var+noise), two similar models fit with variational inference (VI var, VI var+noise), and a stationary model when tested on several UCI datasets. The datasets were partitioned 72/18/10 into training/validation/test sets with results averaged over five such partitions. The lowest RMSE is **bolded**. The SGPR approximation is used for all methods with 100 inducing points. The Matérn kernel with ν = 0.5 is used for all GPs, including the prior in the VI models.

| Dataset                | Stationary | VI var | NNP var | VI var+noise | NNP var+noise |
|------------------------|------------|--------|---------|--------------|----------------|
| Gas ×10³               | 9.91       | 19.46  | 8.10    | 5.30         | **2.67**       |
| Skillcraft ×10³        | **2.65**   | 2.71   | 2.66    | 2.71         | 11.30          |
| SML                    | 4.60       | 3.35   | 2.60    | 3.34         | **2.48**       |
| Parkinsons             | 7.55       | 6.03   | **3.86**| 5.95         | 3.98           |
| Pumadyn               | 1.00       | **1.00**| 1.00    | 1.00         | 1.00           |
| PolTele                | 2.42       | 2.08   | 2.17    | **2.08**     | 2.26           |
| KEGG                   | 1.43       | 1.46   | **1.31**| 1.54         | 1.33           |
| Elevators ×10³         | 2.68       | 2.83   | **2.23**| 3.02         | 2.68           |
| KEGGU                  | 0.340      | 0.393  | **0.167**| 0.412        | 0.207          |
| Kin40k                 | 0.976      | 0.970  | 0.960   | 0.970        | **0.954**      |
| Protein                | 4.44       | 4.57   | 4.34    | 4.57         | **4.33**       |
| CTslice                | 17.1       | 27.4   | 16.8    | 27.4         | **16.5**       |



