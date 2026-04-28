import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import os
filename = os.path.splitext(os.path.basename(__file__))[0]


SEED = 42
np.random.seed(SEED)
data_degree = 10
fit_degrees = np.arange(1, 51)
lambda_reg = 1e-3


def polynomial_func(X, coeffs):
    f = 0
    for i, c in enumerate(coeffs):
        f += c * X**i
    return f

def generate_data_uniform(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    noise = np.random.normal(noise_mean, noise_std, n)
    return f + noise

def generate_data_non_uniform(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    sigma = noise_std * (1 + X**2)
    noise = np.random.normal(noise_mean, sigma, n)
    return f + noise


err_params = [0., 0.1]
func_params = np.random.uniform(1., 5., data_degree)

X_train = np.random.uniform(-1., 1., size=1000)
y_train_nu = generate_data_non_uniform(X_train, func_params, *err_params)
y_train_u = generate_data_uniform(X_train, func_params, *err_params)
X_train = X_train.reshape(-1, 1)

X_test = np.random.uniform(-1., 1., size=500)
y_test_nu = generate_data_non_uniform(X_test, func_params, *err_params)
y_test_u = generate_data_uniform(X_test, func_params, *err_params)
X_test = X_test.reshape(-1, 1)


# plt.figure(figsize=(10, 6))
# plt.scatter(X_train, y_train_u, c="g", marker=".", label="data uniform")
# plt.scatter(X_train, y_train_nu, c="r", marker=".", label="data non uniform")


loss_train_u = []
risk_u       = []
loss_test_u  = []
best_loss_u  = float(np.inf)
loss_train_nu = []
risk_nu       = []
loss_test_nu  = []
best_loss_nu  = float(np.inf)

# uniform
for deg in fit_degrees:
    
    poly_u = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly_u.fit(X_train, y_train_u)
    y_pred_u = poly_u.predict(X_train)
    train_loss_u = mean_squared_error(y_train_u, y_pred_u)
    loss_train_u.append(train_loss_u)
    risk_u.append(train_loss_u + lambda_reg * deg)
    y_test_pred_u = poly_u.predict(X_test)
    loss_test_u.append(mean_squared_error(y_test_u, y_test_pred_u))


# non uniform
sigma = err_params[1] * (1 + X_train.flatten()**2)
weights = 1 / sigma**2
for deg in fit_degrees:

    poly_nu = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly_nu.fit(X_train, y_train_nu, linearregression__sample_weight=weights)
    y_pred_nu = poly_nu.predict(X_train)
    train_loss_nu = mean_squared_error(y_train_nu, y_pred_nu)
    loss_train_nu.append(train_loss_nu)
    risk_nu.append(train_loss_nu + lambda_reg * deg)
    y_test_pred_nu = poly_nu.predict(X_test)
    loss_test_nu.append(mean_squared_error(y_test_nu, y_test_pred_nu))


plt.figure(figsize=(10, 6))
plt.plot(fit_degrees, risk_u, ls='-', lw=1, label="SRM risk (uniform)")
plt.plot(fit_degrees, risk_nu, ls='-', lw=2, label="SRM risk (non uniform)")
plt.plot(fit_degrees, loss_train_u, c='r', ls='-', lw=1, label="train (uniform)")
plt.plot(fit_degrees, loss_test_u, c='b', ls='--', lw=1, label="test (uniform)")
plt.plot(fit_degrees, loss_train_nu, c='r', ls='-', lw=2, label="train (non uniform)")
plt.plot(fit_degrees, loss_test_nu, c='b', ls='--', lw=2, label="test (non uniform)")
plt.xlabel("degree")
plt.ylabel("loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}.pdf")
plt.show()

best_idx_u  = np.argmin(risk_u)
best_idx_nu = np.argmin(risk_nu)

print(f"Best Risk (uniform):       {risk_u[best_idx_u]:.4f}")
print(f"Best Degree (uniform):     {fit_degrees[best_idx_u]}")
print(f"Best Risk (non uniform):   {risk_nu[best_idx_nu]:.4f}")
print(f"Best Degree (non uniform): {fit_degrees[best_idx_nu]}")