# Digit Recognizer using MNIST Dataset
This project allows you to recognize handwritten digits using a machine learning model built with PCA (Principal Component Analysis) and KNN (K-Nearest Neighbors) on the famous MNIST dataset. 
The app lets you either draw a digit on a canvas or upload an image for digit recognition.

## Features
* Draw or upload a digit image.
* Preprocess the image by resizing, centering, and normalizing it.
* Use PCA for dimensionality reduction and KNN for digit classification.
* Real-time digit prediction using a trained model.

## Technologies Used
* Streamlit for creating the interactive web app.
* Pandas for handling data.
* Scikit-learn for machine learning algorithms like PCA and KNN.
* Pillow for image processing.
* Joblib for saving and loading models.
* SciPy for image centering.

## Installation
* To run this project follow this Link : https://digit-recognizer-using-knn-pca.streamlit.app/

## Model Explanation
* PCA: This technique reduces the dimensionality of the MNIST data by transforming it into a set of orthogonal components that explain most of the variance in the dataset. It helps to speed up the training process.
* KNN (K-Nearest Neighbors): A classification algorithm that predicts the label of a sample based on the majority vote of its nearest neighbors in the training set.

## How It Works
* Draw a Digit: Use the drawing canvas to draw a digit. The image is preprocessed by resizing it to 28x28 pixels, converting it to grayscale, and normalizing the pixel values.
* Upload an Image: Upload an image of a handwritten digit. The image is also preprocessed before being passed through the trained PCA model.
* Prediction: Once the image is processed, the model applies PCA and uses the KNN classifier to predict the digit.

## Contributing
If you want to contribute to this project:

Fork the repo.
Create a new branch for your feature.
Make changes and commit them.
Push your changes to your fork.
Open a pull request.

## Acknowledgments
* Thanks to the MNIST dataset, which is widely used in the machine learning community.
* Thanks to Streamlit for providing a simple way to create web apps.
