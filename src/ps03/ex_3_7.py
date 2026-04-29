'''
Exercise 3.7. Consider the MNIST dataset. Use
a classification algorithm of your choice and
compare accuracy based on the original images
and on the images obtained by PCA dimension
reduction.
'''

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
train_frac = 0.7
neighbors = 11
max_components = 64
components = range(10, max_components + 1)


digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_frac, shuffle=True, random_state=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=neighbors)
knn.fit(X_train, y_train)
train_acc = accuracy_score(y_train, knn.predict(X_train))
test_acc = accuracy_score(y_test, knn.predict(X_test))

train_acc_pca, test_acc_pca = [], []
for n in components:

    pca = PCA(n_components=n, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn_pca = KNeighborsClassifier(n_neighbors=neighbors)
    knn_pca.fit(X_train_pca, y_train)
    train_acc_pca.append(accuracy_score(y_train, knn_pca.predict(X_train_pca)))
    test_acc_pca.append(accuracy_score(y_test, knn_pca.predict(X_test_pca)))

best_component_index = np.argmax(test_acc_pca)
best_component = components[best_component_index]

print("Simple:\n"
      f"    Train Accuracy: {train_acc:.2%} | "
      f"Test Accuracy: {test_acc:.2%}")
print(f"PCA (num. components: {best_component}):\n"
    f"    Train Accuracy: {train_acc_pca[best_component_index]:.2%} | "
    f"Test Accuracy: {test_acc_pca[best_component_index]:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(components, train_acc_pca, c='r', ls='-', label="Train (PCA)")
plt.plot(components, test_acc_pca, c='b', ls='--', label="Test (PCA)")
plt.axhline(train_acc, c='tab:orange', ls=":", label="Train (simple)")
plt.axhline(test_acc, c='tab:blue', ls=":", label="Test (simple)")
plt.axvline(best_component, color="lightgray", linestyle="--", label=f"Best component: {best_component}")
plt.xlabel("Components")
plt.ylabel("Accuracy")
plt.title(f"KNN (k={neighbors}) accuracy vs. PCA")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/{filename}.pdf")
plt.show()