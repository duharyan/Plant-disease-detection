import tensorflow as tf
import numpy as np
import streamlit as st

# Function to load and predict using a specific model
def model_prediction(test_image, model_name):
    # Load the correct model based on user choice
    if model_name == "ResNet50":
        model_path = r"D:\streamlit_demo\resnet_model.keras"
        image_size = (224, 224)  # Use 224x224 for ResNet50
    elif model_name == "VGG16":
        model_path = r"D:\streamlit_demo\vgg_model.keras"
        image_size = (224, 224)  # Use 224x224 for VGG16
    elif model_name == "CNN":
        model_path = r"D:\streamlit_demo\trained_plant_disease_model.keras"
        image_size = (128, 128)  # Use 128x128 for the basic CNN model
    else:
        st.error("Invalid model selected.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

    # Process the uploaded image
    try:
        # Resize the image to the correct dimensions based on the model
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=image_size)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch format
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Streamlit Interface
st.set_page_config(page_title="Plant Disease Recognition System", page_icon="🌿", layout="centered")

# Add Custom CSS for Background Image
page_bg_img = '''
<style>
/* Set the background image for the app */
.stApp {
  background-image: url("https://camo.githubusercontent.com/fe44f6ed112e810177d70457c039aa7542565c2b95659383ca929d52af6a5fc5/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313230302f312a4673776c46346c5a5051346b545f676b796261635a772e6a706567");
  background-size: cover;
}

/* Universal font color for the whole app */
* {
  color: #FFFFFF;  /* Sea green color for all text */
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")

# App Modes
app_mode = st.sidebar.radio("Go to", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.title("🌿 Plant Disease Recognition System")
    st.markdown("""
    ## Welcome to the Plant Disease Recognition System!
    Utilize cutting-edge **Transfer Learning** techniques with pre-trained models like **ResNet50**, **VGG16**, and our custom **CNN** model to identify plant diseases accurately.
    
    ### 🌟 Features:
    - **Transfer Learning:** Using state-of-the-art models to detect diseases quickly.
    - **Accurate Diagnosis:** Leveraging advanced neural networks trained on a vast dataset.
    - **Easy to Use:** Upload a plant image and get results in seconds!

    ### 📖 How It Works:
    1. Go to the **Disease Recognition** page.
    2. Upload an image of the plant.
    3. Choose a pre-trained model (ResNet50, VGG16, or CNN).
    4. Get disease predictions and recommended actions.

    ### Get Started!
    Click on the **Disease Recognition** page in the sidebar to try it out.
    """)

# About Page
elif app_mode == "About":
    st.title("About Plant Disease Recognition System")
    st.markdown("""
    ### 🌿 Project Overview
    This project aims to leverage the power of **Transfer Learning** using pre-trained models like **ResNet50**, **VGG16**, and a custom **CNN** model to identify plant diseases accurately.
    
    ### 📊 Dataset Information
    The dataset includes over 87,000 RGB images categorized into 38 classes of healthy and diseased plants.

    ### 🤖 Models Used
    - **ResNet50**: A 50-layer deep convolutional neural network known for its residual connections.
    - **VGG16**: A 16-layer deep neural network focusing on simplicity with smaller convolutional filters.
    - **CNN**: A custom Convolutional Neural Network trained specifically for plant disease recognition.

    ### Why Use Transfer Learning?
    Transfer Learning allows us to use pre-trained models to:
    - Save computational resources.
    - Achieve high accuracy with a smaller dataset.
    - Benefit from models already trained on large datasets like ImageNet.

    ### Project Goals
    1. Efficiently detect plant diseases.
    2. Provide a user-friendly interface for farmers and researchers.
    3. Promote agricultural health and productivity.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("🌿 Disease Recognition")
    st.markdown("""
    Upload an image of the plant leaf, choose the model, and our system will predict the disease!
    """)
    
    # Model Selection Dropdown
    model_choice = st.selectbox("Select Model", ["ResNet50", "VGG16", "CNN"])
    
    # Image Upload Section
    test_image = st.file_uploader("Upload an Image of the Plant Leaf", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        # Predict Button
        if st.button("Predict Disease"):
            with st.spinner("Analyzing the image..."):
                result_index = model_prediction(test_image, model_choice)
            
            if result_index is not None:
                # Labels for the predictions
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                
                disease_name = class_names[result_index]
                st.success(f"Prediction: It's a {disease_name}")
            else:
                st.error("Prediction could not be completed.")