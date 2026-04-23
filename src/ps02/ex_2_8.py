from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
import logging
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs("results", exist_ok=True)
import pandas as pd

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("EX_2_2")

# HYPERPARAMETERS
SEED = 42
num_epochs = 1001
train_split = 0.7
shuffle = True
lr = 0.01
num_folds = 5


with open(f"results/ex_2_7_best_test_error_{num_folds}_folds.txt", "w") as f:
    f.write("")


np.random.seed(SEED)
torch.manual_seed(SEED)

# DATASET & MODEL
class AMESDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
         return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LinearModel(nn.Module):
    def __init__(self, n_inputs, n_ouputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_ouputs)

    def forward(self, x):
        return self.linear(x)
    



# DATA
ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data
y = ames.target

logger.debug(X.shape)
logger.debug(y.shape)
logger.debug(X.head())
logger.debug(y.head())

# PREPROCESS
# one-hot-encoding
X_num = X.select_dtypes(include=[np.number])
X_num = X_num.fillna(X_num.median())
X_cat = X.select_dtypes(include=["str", "category"])
X_cat = X_cat.fillna(X_cat.mode().iloc[0])
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
X = pd.concat([X_num, X_cat_encoded], axis=1)

y = y.astype(float).values

ALL_TRAIN_LOSS = {}
ALL_TEST_LOSS  = {}

kfold = KFold(n_splits=num_folds, shuffle=shuffle, random_state=SEED)
for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    logger.info(f"Fold: {fold+1}/{num_folds}")

    X_train_fold = X.iloc[train_idx].values
    X_test_fold  = X.iloc[test_idx].values
    y_train_fold = y[train_idx]
    y_test_fold  = y[test_idx]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_fold = scaler_X.fit_transform(X_train_fold)
    X_test_fold  = scaler_X.transform(X_test_fold)
    y_train_fold = scaler_y.fit_transform(y_train_fold.reshape(-1, 1))
    y_test_fold  = scaler_y.transform(y_test_fold.reshape(-1, 1))

    train_dataset = AMESDataset(X_train_fold, y_train_fold)
    test_dataset  = AMESDataset(X_test_fold,  y_test_fold)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=shuffle)
    test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset))

    # TRAINING
    model = LinearModel(X_train_fold.shape[1], 1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    TRAIN_LOSS = []
    TEST_LOSS = []
    best_test_loss = float("inf")
    for epoch in range(num_epochs):

        # training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        TRAIN_LOSS.append(train_loss)

        # evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        TEST_LOSS.append(test_loss)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Train Loss: {train_loss} | Test Loss: {test_loss}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # logger.info(f"\tNew best test loss:{best_test_loss}")

    with open(f"results/ex_2_7_best_test_error_{num_folds}_folds.txt", "a") as f:
        f.write(f"Fold:\t{fold+1}/{num_folds}\t-\tLoss:\t{best_test_loss:.5f}\n")

    ALL_TRAIN_LOSS[fold] = TRAIN_LOSS
    ALL_TEST_LOSS[fold]  = TEST_LOSS

plt.figure()
for fold in range(num_folds):
    alpha = 0.3 + 0.7 * (fold / (num_folds - 1))
    plt.plot(ALL_TRAIN_LOSS[fold], label='train' if fold==0 else None, c='r', ls='-',  alpha=alpha)
    plt.plot(ALL_TEST_LOSS[fold],  label='test'  if fold==0 else None, c='b', ls='--', alpha=alpha)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/ex_2_8_{num_folds}_folds.pdf")
plt.show()