# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image, ImageOps

# Load MNIST data
data = pd.read_csv('train.csv')
X = data.drop(columns=['label']).values  # Input
y = data['label'].values  # Target

# Normalize pixel values to [0, 1]
X = X / 255.0





# PCA
def SelectNcomponents(data, variance_threshold=0.99):
    pca = PCA()
    pca.fit(data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    print(f"Selected number of components: {n_components}")
    return n_components

pca = PCA(n_components=SelectNcomponents(X))
X_pca = pca.fit_transform(X)

# Train/test split (with stratification)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pca, y, test_size=0.4, stratify=y, random_state=42
)

#KNN Hyperparameter Selection
def SelectK(X_train, Y_train):
    best_k = 1
    best_score = 0
    for k in range(3, 11, 2):  # Test odd values of k
        knn1 = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn1, X_train, Y_train, cv=5, scoring='accuracy')
        if scores.mean() > best_score:
            best_k = k
            best_score = scores.mean()
    print(f"Best k: {best_k}")
    return best_k


best_k = SelectK(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, Y_train)

# Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy on test set:", accuracy)

# Save models
joblib.dump(pca, 'pca_mnist.pkl')
joblib.dump(knn, 'knn_mnist.pkl')
