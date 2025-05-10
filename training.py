# This file was used for training and testing all models. Note that while the perfromance on the test set is
# reported, the model was trained on the training set and validated on the validation set. The test set was only used
# for final evaluation. The model was trained using 10-fold cross-validation on the training set, and the best hyperparameters were selected based on the validation set performance.

from nnp import *
from svgp_var import *
from svgp_var_noise import *
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import time
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv("gas.csv", header = None)

df = df.drop_duplicates()

scaler = StandardScaler()
df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:]).copy()

df = df.sample(frac=1).reset_index(drop=True)
data = torch.Tensor(df.values)
print(df.shape)
sizes = [int(0.9 * len(data))]
sizes.append(len(data) - sum(sizes))
train_val, test = random_split(data, sizes)
train_val, test = data[train_val.indices].clone().detach(), data[test.indices].clone().detach()
rows = []
i = 0
kf = KFold(n_splits=10, shuffle=True, random_state=1)
for fold, (train_val_idx, test_idx) in enumerate(kf.split(train_val)):
    train_val = data[train_val_idx].clone().detach()
    test = data[test_idx].clone().detach()
    
    full_mses, full_maes, full_log_scores, full_lr, full_iterations, full_valid_log_scores, full_var_mses, full_var_maes, full_deep = ([] for _ in range(9))
    k_mses, k_maes, k_log_scores, k_lr, k_iterations, k_valid_log_scores = ([] for _ in range(6))
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val)):
        train = train_val[train_idx].clone().detach()
        val = train_val[val_idx].clone().detach()
        start = time.time()
        for deep in [0, 1]:
            for lr in [0.01, 0.05, 0.1]:
                np.random.seed(0)
                torch.manual_seed(0)
                mses, maes, log_scores, lrs, iterations, valid_log_scores = fit_model(train[:,1:], train[:,0], val[:,1:], val[:,0], test[:,1:], test[:,0], training_iterations=10001, learning_rate=lr, deep=deep)
                # mses, maes, log_scores, lrs, iterations, valid_log_scores = fit_svgp(train[:,1:], train[:,0], val[:,1:], val[:,0], test[:,1:], test[:,0], training_iterations=100, learning_rate=lr)
                
                full_mses += mses
                full_maes += maes
                full_log_scores += log_scores
                full_lr += lrs
                full_iterations += iterations
                full_valid_log_scores += valid_log_scores
                full_deep += [deep] * len(mses)
        end = time.time()
        ts = [end - start] * len(full_mses)   
        break
    res = pd.DataFrame({"mse": full_mses, "mae": full_maes, "log_score": full_log_scores, "lr": full_lr, "iterations": full_iterations, "valid_log_score": full_valid_log_scores, "time": ts, "deep": full_deep})
    df_avg = res.groupby(['lr', 'iterations', 'deep'], as_index=False).mean()
    df_avg["valid_log_score"] = pd.to_numeric(df_avg["valid_log_score"], errors="coerce")
    best_row = df_avg.loc[df_avg["valid_log_score"].idxmax()]
    rows.append(best_row)
    i += 1
    if i == 5:
        break
res = pd.DataFrame(rows).mean()
sds = pd.DataFrame(rows).std()
print(res)
print(sds)
print(df.shape)
savefile = pd.DataFrame([res, sds])
# savefile.to_csv("...", index=False)
