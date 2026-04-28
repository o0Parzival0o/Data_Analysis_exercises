import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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
data_degree = 10
fit_degrees = np.arange(1, 51)
lambda_reg = 1e-2
k_folds = 5


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


X = np.random.uniform(-1., 1., size=200)
y = generate_data(X, func_params, *err_params)
X = X.reshape(-1, 1)


loss_train = {k: [] for k in ["holdout", "kfold", "loo"]}
# risk       = {k: [] for k in ["holdout", "kfold", "loo"]}
loss_val   = {k: [] for k in ["holdout", "kfold", "loo"]}
best_loss  = {k: float("inf") for k in ["holdout", "kfold", "loo"]}

# hold out
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=shuffle, random_state=SEED)

for deg in fit_degrees:
    
    poly = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly.fit(X_train, y_train)
    y_pred = poly.predict(X_train)
    train_loss = mean_squared_error(y_train, y_pred)
    loss_train["holdout"].append(train_loss)
    # risk["holdout"].append(train_loss + lambda_reg * deg)
    y_val_pred = poly.predict(X_val)
    loss_val["holdout"].append(mean_squared_error(y_val, y_val_pred))

logger.info("Hold out done.")


# kfold
kfold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=SEED)

for deg in fit_degrees:

    train_losses = []
    # risks        = []
    val_losses   = []

    for fold, (tran_idx, val_idx) in enumerate(kfold.split(X)):

        X_train, X_val = X[tran_idx], X[val_idx]
        y_train, y_val = y[tran_idx], y[val_idx]
        
        poly = make_pipeline(
            PolynomialFeatures(degree=deg),
            LinearRegression()
        )
        poly.fit(X_train, y_train)
        y_pred = poly.predict(X_train)
        train_loss = mean_squared_error(y_train, y_pred)
        train_losses.append(train_loss)
        # risks.append(train_loss + lambda_reg * deg)
        y_val_pred = poly.predict(X_val)
        val_losses.append(mean_squared_error(y_val, y_val_pred))

    loss_train["kfold"].append(np.mean(train_losses))
    # risk["kfold"].append(np.mean(risks))
    loss_val["kfold"].append(np.mean(val_losses))
    
logger.info("K fold done.")

# loo
loo = LeaveOneOut()

for deg in fit_degrees:

    train_losses = []
    # risks        = []
    val_losses   = []

    for fold, (tran_idx, val_idx) in enumerate(loo.split(X)):

        X_train, X_val = X[tran_idx], X[val_idx]
        y_train, y_val = y[tran_idx], y[val_idx]
        
        poly = make_pipeline(
            PolynomialFeatures(degree=deg),
            LinearRegression()
        )
        poly.fit(X_train, y_train)
        y_pred = poly.predict(X_train)
        train_loss = mean_squared_error(y_train, y_pred)
        train_losses.append(train_loss)
        # risks.append(train_loss + lambda_reg * deg)
        y_val_pred = poly.predict(X_val)
        val_losses.append(mean_squared_error(y_val, y_val_pred))

    loss_train["loo"].append(np.mean(train_losses))
    # risk["loo"].append(np.mean(risks))
    loss_val["loo"].append(np.mean(val_losses))

logger.info("LOO done.")





for i in ["holdout", "kfold", "loo"]:

    best_idx = np.argmin(loss_val[i])
    best_loss_val = loss_val[i][best_idx]
    best_degree = fit_degrees[best_idx]

    logger.info(f"{i}:")
    logger.info(f"    Best Validation Loss: {best_loss_val:.4f}")
    logger.info(f"    Best Degree:          {best_degree}")

    plt.figure(figsize=(10, 6))
    # plt.plot(fit_degrees, risk[i], ls='-', lw=1, label="SRM risk")
    plt.plot(fit_degrees, loss_train[i], c='r', ls='-', lw=1, label="train loss")
    plt.plot(fit_degrees, loss_val[i], c='b', ls='--', lw=1, label="val loss")
    plt.xlabel("degree")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(f"{i} (best validation loss = {best_loss_val:.4f}; degree = {best_degree})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}_{i}.pdf")

plt.show()
