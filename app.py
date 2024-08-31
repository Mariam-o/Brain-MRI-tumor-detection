import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model('brain_mri_detection_model (1).h5')

def preprocess_image(img_path):
    # Load the image
    img = Image.open(img_path).convert('RGB')
    
    # Convert to grayscale
    gray_img = img.convert('L')
    
    # Apply Gaussian blur
    gray_img = gray_img.filter(ImageFilter.GaussianBlur(5))
    
    # Apply binary threshold
    threshold = 45
    gray_img = gray_img.point(lambda p: p > threshold and 255)
    
    # Find the bounding box of the brain region (contour)
    bbox = gray_img.getbbox()
    if bbox:
        # Crop the image to the bounding box
        cropped_img = img.crop(bbox)
    else:
        # If no contours are found, use the original image
        cropped_img = img
    
    # Resize the image to the target size (224, 224)
    resized_img = cropped_img.resize((224, 224))
    
    # Convert the image to array and normalize
    img_array = image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image to match the preprocessing during training
    
    return img_array

def predict_image(model, img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    return prediction, "Tumor Detected" if prediction[0] > 0.5 else "No Tumor Detected"

# Streamlit UI
st.set_page_config(page_title="Brain MRI Detection App", page_icon="🧠", layout="wide")

# Header section with an image and title
# st.image("https://example.com/header_image.jpg", use_column_width=True)  
st.title("🧠 Brain MRI Tumor Detection")
st.write("This application uses a deep learning model to predict the presence of a brain tumor from MRI scans. Simply upload an MRI image to get started.")

# Instructions
st.markdown("""
## How It Works
1. Upload a JPG image of a brain MRI scan.
2. The image will be processed to detect brain contours.
3. The model will analyze the scan and predict whether a brain tumor is present.
4. View the result.
""")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

# Display results only if an image is uploaded
if uploaded_file is not None:
    st.subheader("Uploaded MRI Image:")
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=False, width=300)  

    
    with st.spinner('Analyzing the MRI scan...'):
        prediction, result = predict_image(model, uploaded_file)
    
    st.subheader("Prediction Results:")
    st.write(f"**Prediction:** {result}")
    
    # Display advice or recommendations
    if result == "Tumor Detected":
        st.error("⚠️ A brain tumor has been detected in the MRI scan. Please consult a medical professional for further analysis.")
    else:
        st.success("✅ No brain tumor detected. The MRI scan appears to be normal.")
    