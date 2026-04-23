from sklearn.model_selection import train_test_split
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
train_split = np.linspace(0.1, 0.9, 20)
shuffle = True
lr = 0.01


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

TRAIN = []
TEST = []

for i, frac in enumerate(train_split):
    logger.info(f"Split: {i+1}/{len(train_split)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac, shuffle=shuffle, random_state=SEED)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test  = scaler_y.transform(y_test.reshape(-1, 1))

    train_dataset = AMESDataset(X_train, y_train)
    test_dataset  = AMESDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=shuffle)
    test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset))

    # TRAINING
    model = LinearModel(X_train.shape[1], 1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)


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

        # evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Train Loss: {train_loss} | Test Loss: {test_loss}")
    
    TRAIN.append(train_loss)
    TEST.append(test_loss)

plt.figure()
plt.plot(train_split, TRAIN, label='train', c='r', ls='-',  marker='.')
plt.plot(train_split, TEST,  label='test',  c='b', ls='--', marker='.')
plt.xlabel("Train Fraction")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("results/ex_2_2_loss_vs_train_fraction.pdf")
plt.show()