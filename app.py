# ----------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift

# Load the saved PCA and KNN models
pca = joblib.load('pca_mnist.pkl')
knn = joblib.load('knn_mnist.pkl')



# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Convert to a numpy array and normalize pixel values
    image_array = np.array(image) / 255.0  # Normalize to [0.0, 1.0]

    # Thresholding to remove noise
    image_array[image_array < 0.5] = 0
    image_array[image_array >= 0.5] = 1

    # Center the image
    def center_image(image_array):
        from scipy.ndimage import center_of_mass
        from scipy.ndimage.interpolation import shift

        cy, cx = center_of_mass(image_array)
        offset = np.array(image_array.shape) // 2 - np.array([cy, cx])
        return shift(image_array, shift=offset, mode='constant', cval=0)

    centered_image = center_image(image_array)

    # Clamp values to avoid errors
    centered_image = np.clip(centered_image, 0.0, 1.0)  # Ensure values are in [0.0, 1.0]

    # Flatten the array
    image_flat = centered_image.flatten()
    return image_flat


# Streamlit App
st.title("Digit Recognizer App")
st.write("Draw a digit or upload an image to recognize it.")

# Sidebar for input options
input_option = st.sidebar.radio("Choose Input Method", ["Draw Digit", "Upload Image"])

if input_option == "Draw Digit":
    from streamlit_drawable_canvas import st_canvas

    # Streamlit Drawing Canvas
    st.write("**Draw a digit below:**")
    canvas = st_canvas(
        fill_color="#000000",  # Black
        stroke_width=15,
        stroke_color="#FFFFFF",  # White
        background_color="#000000",  # Black
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        # Convert canvas image to PIL format
        canvas_image = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
        # st.image(canvas_image, caption="Drawn Digit", width=140)

        # # Preprocess the image
        processed_image = preprocess_image(canvas_image)
        # st.image(processed_image.reshape(28, 28), caption="Preprocessed Image", width=140)

        # Apply PCA
        processed_image_pca = pca.transform([processed_image])

        # Predict
        if st.button("Predict"):
            prediction = knn.predict(processed_image_pca)
            st.write(f"**Predicted Digit:** {prediction[0]}")

elif input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Open the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", width=140)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_image)
        st.image(processed_image.reshape(28, 28), caption="Preprocessed Image", width=140)

        # Apply PCA
        processed_image_pca = pca.transform([processed_image])

        # Predict
        if st.button("Predict"):
            prediction = knn.predict(processed_image_pca)
            st.write(f"**Predicted Digit:** {prediction[0]}")


st.write('Created By Yogesh Chouhan')

