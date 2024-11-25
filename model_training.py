# import numpy as np 
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib




# data = pd.read_csv('train.csv')


# # input
# X = data.drop(columns=['label'])
# # Target
# y = data['label']

# # print(X.shape,y.shape)

# # Normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Select N_components function
# def SelectNcomponents(data):
#     pca1  = PCA()
#     pca1.fit(data)
#     cumsum = 0
#     count = 0
#     for i in pca1.explained_variance_ratio_:
#         if cumsum<0.90:
#             cumsum += i
#             count += 1
#         else:
#             break
#     print(count)
#     return count




# # PCA
# pca = PCA(n_components=SelectNcomponents(X))
# X = pca.fit_transform(X)
# print(X.shape)


# # Train test split
# X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)


# # SelectK function
# def SelectK(X_train,X_test,Y_train,Y_test):
#     max_score = 0
#     k = 1
#     for i in range(1,10):
#         knn1 = KNeighborsClassifier(i)
#         knn1.fit(X_train,Y_train)
#         y_pred = knn1.predict(X_test)
#         score = accuracy_score(Y_test,y_pred)
#         if score>max_score:
#             max_score = score
#             k = i
#     print(k)
#     return k


# # Train KNearestClassifier
# knn = KNeighborsClassifier(n_neighbors=SelectK(X_train,X_test,Y_train,Y_test))
# knn.fit(X_train,Y_train)

# # Evaluate model
# y_pred = knn.predict(X_test)
# acuracy = accuracy_score(Y_test,y_pred)
# print('accuracy_score : ', acuracy)


# # Save the PCA and KNN model
# joblib.dump(pca, 'pca_mnist.pkl')
# joblib.dump(knn, 'knn_mnist.pkl')



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
# ## -----------------------------------------------------------------------------------------------------------------------------------

# image = Image.open('Screenshot 2024-11-25 175438.png')

#     # Resize the image to 28x28 pixels (same as MNIST dataset)
# image = image.resize((28, 28))

#     # Convert to grayscale
# image = ImageOps.grayscale(image)

#     # Convert to a numpy array
# image_array = np.array(image)

#     # Normalize pixel values (0 to 1)
# image_array = image_array / 255.0

#     # Flatten the array to shape (1, 784)
# image_flattened = image_array.flatten().reshape(1, -1)

# # image_flattened = image_flattened/255
# image_flattened = pca.transform(image_flattened)


# print(knn.predict(image_flattened))
# Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy on test set:", accuracy)

# Save models
joblib.dump(pca, 'pca_mnist.pkl')
joblib.dump(knn, 'knn_mnist.pkl')