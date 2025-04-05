import streamlit as st
import requests
from PIL import Image
import base64

def main():
    st.title("Lung Tumor Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict Tumor'):
            files = {'file': ('image.jpg', uploaded_file.getvalue(), 'image/jpeg')}
            
            try:
                response = requests.post('http://localhost:8000/predict-tumor/', files=files)
                response.raise_for_status()
                result = response.json()
                
                st.subheader("Prediction Result:")
                st.write(f"Tumor Type: {result['tumor_type']}")
                st.write(f"Predicted Label: {result['predicted_label']}")
                st.write(f"Tumor Likelihood Score: {result['tumor_likelihood_score']:.4f}")
                
                # Check if accuracy is present
                if 'accuracy' in result:
                    st.write(f"Model Accuracy: {result['accuracy']:.2f}%")
                else:
                    st.write("Model Accuracy: Not available")
                
                st.subheader("Image Analysis:")
                
                # Display plots
                for plot_name, plot_data in result['plots'].items():
                    st.image(base64.b64decode(plot_data), caption=plot_name.replace('_', ' ').title(), use_column_width=True)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while making the request: {e}")

if __name__ == "__main__":
    main()
