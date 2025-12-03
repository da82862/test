import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Set page title
st.title("Age Classification using ViT")

# Load the age classification pipeline
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a button to classify
    if st.button("Classify Age"):
        with st.spinner("Classifying..."):
            # Classify age
            age_predictions = age_classifier(image)
            age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
            
            # Display results
            st.subheader("Predicted Age Range:")
            st.success(f"**Age range: {age_predictions[0]['label']}**")
            st.write(f"**Confidence: {age_predictions[0]['score']:.2%}**")
            
            # Show all predictions
            st.subheader("All Predictions:")
            for pred in age_predictions:
                st.write(f"{pred['label']}: {pred['score']:.2%}")
else:
    st.info("Please upload an image to classify age.")
