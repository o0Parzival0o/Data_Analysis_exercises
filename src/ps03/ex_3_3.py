'''
Exercise 3.3. Analyse training and test error in
dependence of the parameter λ with ridge re-
gression and lasso on the dataset Ames. Pro-
duce a picture of coefficients values with re-
spect to λ (estimate the test error with CV).
'''

from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.makedirs("results", exist_ok=True)
filename = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("EX_3_3")


SEED = 42
np.random.seed(SEED)
lambda_values = np.logspace(-4, 7, 20)
shuffle = True
k_folds = 7


ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data
y = ames.target

logger.debug(X.shape)
logger.debug(y.shape)
logger.debug(X.head())
logger.debug(y.head())

# one-hot-encoding
X_num = X.select_dtypes(include=[np.number])
X_num = X_num.fillna(X_num.median())
X_cat = X.select_dtypes(include=["str", "category"])
X_cat = X_cat.fillna(X_cat.mode().iloc[0])
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
X = pd.concat([X_num, X_cat_encoded], axis=1).values

y = y.astype(float).values


ridge_error = {"train": [], "test": []}
lasso_error = {"train": [], "test": []}
ridge_coefs = []
lasso_coefs = []

for i, l in enumerate(lambda_values):

    logger.info(f"Lambda: {i+1:2}/{len(lambda_values)}")

    ridge_loss = {"train": [], "test": []}
    lasso_loss = {"train": [], "test": []}
    coefs = {"ridge": [], "lasso": []}

    kfold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test  = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test  = scaler_y.transform(y_test.reshape(-1, 1))
        
        ridge = Ridge(alpha=l)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_train)
        y_test_pred = ridge.predict(X_test)
        ridge_loss["train"].append(mean_squared_error(y_train, y_pred))
        ridge_loss["test"].append(mean_squared_error(y_test, y_test_pred))
        coefs["ridge"].append(ridge.coef_)

        lasso = Lasso(alpha=l)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_train)
        y_test_pred = lasso.predict(X_test)
        lasso_loss["train"].append(mean_squared_error(y_train, y_pred))
        lasso_loss["test"].append(mean_squared_error(y_test, y_test_pred))
        coefs["lasso"].append(lasso.coef_)

    ridge_error["train"].append(np.mean(ridge_loss["train"]))
    ridge_error["test"].append(np.mean(ridge_loss["test"]))
    ridge_coefs.append(np.mean(coefs["ridge"], axis=0))
    lasso_error["train"].append(np.mean(lasso_loss["train"]))
    lasso_error["test"].append(np.mean(lasso_loss["test"]))
    lasso_coefs.append(np.mean(coefs["lasso"], axis=0))



fig, ax = plt.subplots(1,2)

ax[0].plot(lambda_values, ridge_error["train"], label='train', c='r', ls='-')
ax[0].plot(lambda_values, ridge_error["test"],  label='test',  c='b', ls='--')
ax[0].set_xlabel("Lambda")
ax[0].set_ylabel("Error")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_title("Ridge")
ax[0].legend()

ax[1].plot(lambda_values, lasso_error["train"], label='train', c='r', ls='-')
ax[1].plot(lambda_values, lasso_error["test"],  label='test',  c='b', ls='--')
ax[1].set_xlabel("Lambda")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_title("Lasso")
ax[1].legend()

fig.tight_layout()
plt.savefig(f"results/{filename}_error.pdf")


fig, ax = plt.subplots(1,2)

ax[0].plot(lambda_values, ridge_coefs)
ax[0].set_xlabel("Lambda")
ax[0].set_ylabel("Coeff.")
ax[0].set_xscale("log")
ax[0].set_title("Ridge")

ax[1].plot(lambda_values, lasso_coefs)
ax[1].set_xlabel("Lambda")
ax[1].set_ylabel("Coeff.")
ax[1].set_xscale("log")
ax[1].set_title("Lasso")

fig.tight_layout()
plt.savefig(f"results/{filename}_coeffs.pdf")


plt.show()
