from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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
logger = logging.getLogger("EX_2_2")


SEED = 42
np.random.seed(SEED)
train_splits = np.linspace(0.1, 0.9, 20)
shuffle = True


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


train_loss = []
test_loss  = []

for i, frac in enumerate(train_splits):

    logger.info(f"Split: {i+1}/{len(train_splits)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac, shuffle=shuffle, random_state=SEED)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test  = scaler_y.transform(y_test.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    train_loss.append(mean_squared_error(y_train, y_pred))
    y_test_pred = model.predict(X_test)
    test_loss.append(mean_squared_error(y_test, y_test_pred))


plt.figure()
plt.plot(train_splits, train_loss, label='train', c='r', ls='-',  marker='.')
plt.plot(train_splits, test_loss,  label='test',  c='b', ls='--', marker='.')
plt.xlabel("Train Fraction")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}_loss_vs_train_fraction.pdf")
plt.show()
