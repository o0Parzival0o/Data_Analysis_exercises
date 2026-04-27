import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
import logging
import os
filename = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("EX_2_5")

SEED = 42
np.random.seed(SEED)
shuffle = True
relative_eps = 1e-3
data_degree = 5
k_neighbors_max = 21
k_folds = [5, 7, 10]
n_try = 50

neigh = np.arange(1, k_neighbors_max)

def polynomial_func(X, coeffs):
    f = 0
    for i, c in enumerate(coeffs):
        f += c * X**i
    return f

def generate_data(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    noise = np.random.normal(noise_mean, noise_std * np.std(f), n)
    return f + noise


err_params = [0., 0.1]
func_params = np.random.uniform(1., 5., data_degree)


LOSS_TRAIN_HOLDOUT = np.zeros((k_neighbors_max-1, n_try))
LOSS_TEST_HOLDOUT = np.zeros((k_neighbors_max-1, n_try))
LOSS_TRUE_HOLDOUT = np.zeros((k_neighbors_max-1, n_try))
LOSS_TRAIN_K_FOLD = {k: np.zeros((k_neighbors_max-1, n_try)) for k in k_folds}
LOSS_TEST_K_FOLD = {k: np.zeros((k_neighbors_max-1, n_try)) for k in k_folds}
LOSS_TRUE_K_FOLD = {k: np.zeros((k_neighbors_max-1, n_try)) for k in k_folds}
LOSS_TRAIN_LOO = np.zeros((k_neighbors_max-1, n_try))
LOSS_TEST_LOO = np.zeros((k_neighbors_max-1, n_try))
LOSS_TRUE_LOO = np.zeros((k_neighbors_max-1, n_try))

# hold out
for i in range(n_try):

    X = np.random.uniform(-1., 1., size=200)
    y = generate_data(X, func_params, *err_params)
    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=SEED, shuffle=shuffle)

    y_true = polynomial_func(X_test, func_params)

    for neig in range(1, k_neighbors_max):

        knn = KNeighborsRegressor(neig)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        train_loss = mean_squared_error(y_train, y_pred)
        y_test_pred = knn.predict(X_test)
        test_loss = mean_squared_error(y_test, y_test_pred)
        theoretical_loss = mean_squared_error(y_true, y_test_pred)

        LOSS_TRAIN_HOLDOUT[neig-1, i] = train_loss
        LOSS_TEST_HOLDOUT[neig-1, i]  = test_loss
        LOSS_TRUE_HOLDOUT[neig-1, i]  = theoretical_loss
    
    if i % 10 == 0:
        logger.info(f"Hold out: {i+1} / {n_try} done.")

train_mean_holdout = np.mean(LOSS_TRAIN_HOLDOUT, axis=1)
train_std_holdout  = np.std(LOSS_TRAIN_HOLDOUT,  axis=1)
test_mean_holdout  = np.mean(LOSS_TEST_HOLDOUT,  axis=1)
test_std_holdout   = np.std(LOSS_TEST_HOLDOUT,   axis=1)
true_mean_holdout  = np.mean(LOSS_TRUE_HOLDOUT,  axis=1)
true_std_holdout   = np.std(LOSS_TRUE_HOLDOUT,   axis=1)

plt.figure(figsize=(10, 6))
for i in range(n_try):
    plt.plot(neigh, LOSS_TRAIN_HOLDOUT[:, i], c="b", ls="-",  alpha=0.05)
    plt.plot(neigh, LOSS_TEST_HOLDOUT[:, i],  c="r", ls="--", alpha=0.05)
plt.plot(neigh, train_mean_holdout, c="b", ls="-",  label=f"train (mean={train_mean_holdout[-1]:.4f}; std={train_std_holdout[-1]:.4f})")
plt.plot(neigh, test_mean_holdout,  c="r", ls="--", label=f"test (mean={test_mean_holdout[-1]:.4f}; std={test_std_holdout[-1]:.4f})")
plt.xlabel("k")
plt.ylabel("loss")
plt.yscale("log")
plt.title("Hold Out")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}_hold_out.pdf")

logger.info("Hold out done.")


# k-fold
train_mean_k_fold = {}
train_std_k_fold  = {}
test_mean_k_fold  = {}
test_std_k_fold   = {}
true_mean_k_fold  = {}
true_std_k_fold   = {}

for k in k_folds:

    for i in range(n_try):

        X = np.random.uniform(-1., 1., size=200)
        y = generate_data(X, func_params, *err_params)
        X = X.reshape(-1, 1)

        kfold = KFold(n_splits=k, shuffle=shuffle, random_state=SEED)
        
        for neig in range(1, k_neighbors_max):

            train_losses = []
            test_losses = []
            theoretical_losses = []

            for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):

                X_train = X[train_idx]
                X_test  = X[test_idx]
                y_train = y[train_idx]
                y_test  = y[test_idx]

                y_true = polynomial_func(X_test, func_params)

                knn = KNeighborsRegressor(neig)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_train)
                train_losses.append(mean_squared_error(y_train, y_pred))
                y_test_pred = knn.predict(X_test)
                test_losses.append(mean_squared_error(y_test, y_test_pred))
                theoretical_losses.append(mean_squared_error(y_true, y_test_pred))

            LOSS_TRAIN_K_FOLD[k][neig-1, i] = np.mean(train_losses)
            LOSS_TEST_K_FOLD[k][neig-1, i]  = np.mean(test_losses)
            LOSS_TRUE_K_FOLD[k][neig-1, i]  = np.mean(theoretical_losses)
    
        if i % 10 == 0:
            logger.info(f"{k}-fold: {i+1} / {n_try} done.")

    train_mean_k_fold[k] = np.mean(LOSS_TRAIN_K_FOLD[k], axis=1)
    train_std_k_fold[k]  = np.std(LOSS_TRAIN_K_FOLD[k], axis=1)
    test_mean_k_fold[k]  = np.mean(LOSS_TEST_K_FOLD[k], axis=1)
    test_std_k_fold[k]   = np.std(LOSS_TEST_K_FOLD[k], axis=1)
    true_mean_k_fold[k]  = np.mean(LOSS_TRUE_K_FOLD[k], axis=1)
    true_std_k_fold[k]   = np.std(LOSS_TRUE_K_FOLD[k], axis=1)

    plt.figure(figsize=(10, 6))
    for i in range(n_try):
        plt.plot(neigh, LOSS_TRAIN_K_FOLD[k][:, i], c="b", ls="-",  alpha=0.05)
        plt.plot(neigh, LOSS_TEST_K_FOLD[k][:, i],  c="r", ls="--", alpha=0.05)
    plt.plot(neigh, train_mean_k_fold[k], c="b", ls="-",  label=f"train (mean={train_mean_k_fold[k][-1]:.4f}; std={train_std_k_fold[k][-1]:.4f})")
    plt.plot(neigh, test_mean_k_fold[k],  c="r", ls="--", label=f"test (mean={test_mean_k_fold[k][-1]:.4f}; std={test_std_k_fold[k][-1]:.4f})")
    plt.xlabel("k")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(f"{k} Folds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}_{k}_folds.pdf")

logger.info("K fold done.")


# LOO
for i in range(n_try):

    X = np.random.uniform(-1., 1., size=200)
    y = generate_data(X, func_params, *err_params)
    X = X.reshape(-1, 1)

    loo = LeaveOneOut()
    
    for neig in range(1, k_neighbors_max):

        train_losses = []
        test_losses = []
        theoretical_losses = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):

            X_train = X[train_idx]
            X_test  = X[test_idx]
            y_train = y[train_idx]
            y_test  = y[test_idx]

            y_true = polynomial_func(X_test, func_params)

            knn = KNeighborsRegressor(neig)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_train)
            train_losses.append(mean_squared_error(y_train, y_pred))
            y_test_pred = knn.predict(X_test)
            test_losses.append(mean_squared_error(y_test, y_test_pred))
            theoretical_losses.append(mean_squared_error(y_true, y_test_pred))

        LOSS_TRAIN_LOO[neig-1, i] = np.mean(train_losses)
        LOSS_TEST_LOO[neig-1, i]  = np.mean(test_losses)
        LOSS_TRUE_LOO[neig-1, i]  = np.mean(theoretical_losses)
    
    if i % 10 == 0:
        logger.info(f"LOO: {i+1} / {n_try} done.")

train_mean_loo = np.mean(LOSS_TRAIN_LOO, axis=1)
train_std_loo  = np.std(LOSS_TRAIN_LOO, axis=1)
test_mean_loo  = np.mean(LOSS_TEST_LOO, axis=1)
test_std_loo   = np.std(LOSS_TEST_LOO, axis=1)
true_mean_loo  = np.mean(LOSS_TRUE_LOO, axis=1)
true_std_loo   = np.std(LOSS_TRUE_LOO, axis=1)

plt.figure(figsize=(10, 6))
for i in range(n_try):
    plt.plot(neigh, LOSS_TRAIN_LOO[:, i], c="b", ls="-",  alpha=0.05)
    plt.plot(neigh, LOSS_TEST_LOO[:, i],  c="r", ls="--", alpha=0.05)
plt.plot(neigh, train_mean_loo, c="b", ls="-",  label=f"train (mean={train_mean_loo[-1]:.4f}; std={train_std_loo[-1]:.4f})")
plt.plot(neigh, test_mean_loo,  c="r", ls="--", label=f"test (mean={test_mean_loo[-1]:.4f}; std={test_std_loo[-1]:.4f})")
plt.xlabel("k")
plt.ylabel("loss")
plt.yscale("log")
plt.title("LOO")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}_loo.pdf")

logger.info("LOO done.")


print("Hold Out:\n"
      f"    Train Loss:  {train_mean_holdout[-1]:.4f} +- {train_std_holdout[-1]:.4f}\n"
      f"    Test Loss:   {test_mean_holdout[-1]:.4f} +- {test_std_holdout[-1]:.4f}\n"
      f"    True Loss:   {true_mean_holdout[-1]:.4f} +- {true_std_holdout[-1]:.4f}")
for k in k_folds:
    print(f"{k}-fold:\n"
        f"    Train Loss:  {train_mean_k_fold[k][-1]:.4f} +- {train_std_k_fold[k][-1]:.4f}\n"
        f"    Test Loss:   {test_mean_k_fold[k][-1]:.4f} +- {test_std_k_fold[k][-1]:.4f}\n"
        f"    True Loss:   {true_mean_k_fold[k][-1]:.4f} +- {true_std_k_fold[k][-1]:.4f}")
print("LOO:\n"
      f"    Train Loss:  {train_mean_loo[-1]:.4f} +- {train_std_loo[-1]:.4f}\n"
      f"    Test Loss:   {test_mean_loo[-1]:.4f} +- {test_std_loo[-1]:.4f}\n"
      f"    True Loss:   {true_mean_loo[-1]:.4f} +- {true_std_loo[-1]:.4f}")




plt.figure(figsize=(10, 6))
plt.plot(neigh, true_mean_holdout, label=f"Hold Out (mean={true_mean_holdout[-1]:.4f}; {true_std_holdout[-1]:.4f})")
for k in k_folds:
    plt.plot(neigh, true_mean_k_fold[k], label=f"{k} Fold (mean={true_mean_k_fold[k][-1]:.4f}; {true_std_k_fold[k][-1]:.4f})")
plt.plot(neigh, true_mean_loo, label=f"LOO (mean={true_mean_loo[-1]:.4f}; std={true_std_loo[-1]:.4f})")
plt.xlabel("k")
plt.ylabel("loss")
plt.yscale("log")
plt.title(f"True Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}_true_error.pdf")
plt.show()