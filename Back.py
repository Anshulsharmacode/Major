import numpy as np
from PIL import Image
from skimage import color, filters
from skimage.feature import graycomatrix, graycoprops
from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
import io
import base64

# Initialize FastAPI
app = FastAPI()

# Tumor types mapping (including normal image type)
tumor_types = {
    'adenocarcinoma': 'Non-Small Cell Lung Cancer (NSCLC) - Adenocarcinoma',
    'squamous_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Squamous Cell Carcinoma',
    'large_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Large Cell Carcinoma',
    'small_cell': 'Small Cell Lung Cancer (SCLC)',
    'metastatic': 'Secondary Lung Tumors - Metastatic',
    'hamartoma': 'Benign Lung Tumors - Hamartomas',
    'pulmonary_adenoma': 'Benign Lung Tumors - Pulmonary Adenomas',
    'normal': 'Normal Lung Tissue'
}

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

def extract_image_features(image: np.array):
    """Extract features from a given image array."""
    if image.shape[-1] == 4:  # Handle RGBA images
        image = color.rgba2rgb(image)

    # Convert to grayscale and normalize
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))

    # Extract intensity features
    mean_intensity = np.mean(image_normalized)
    std_intensity = np.std(image_normalized)
    cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # Extract edge strength using Sobel filter
    edges_sobel = filters.sobel(image_normalized)
    avg_edge_strength = np.mean(edges_sobel)
    
    # Extract GLCM texture features
    glcm = graycomatrix((image_normalized * 255).astype(np.uint8), distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    return [mean_intensity, std_intensity, cv, avg_edge_strength, contrast, 
            dissimilarity, homogeneity, energy, correlation, asm]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tumor Prediction API! Use the /predict-tumor/ endpoint to upload images."}

@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Extract features from the image
        image_features = extract_image_features(image_array)
        
        # Reshape the features for prediction
        features = np.array(image_features).reshape(1, -1)
        
        # Predict the tumor type
        prediction = model.predict(features)
        
        # Get the predicted label and tumor type
        predicted_label = prediction[0]
        
        # Determine the tumor type, including normal
        tumor_type = tumor_types.get(predicted_label, "Unknown Tumor Type")
        
        # If it's not a known tumor type, classify it as "Normal"
        if predicted_label not in tumor_types:
            tumor_type = tumor_types['normal']
            predicted_label = 'normal'
        
        # Calculate tumor likelihood
        tumor_score = tumor_likelihood(image_features)
        
        # Generate plots
        plots = generate_plots(image_array, image_features)
        
        return {
            "tumor_type": tumor_type,
            "predicted_label": predicted_label,
            "tumor_likelihood_score": tumor_score,
            "plots": plots
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while processing the image: {str(e)}")

# Training the model with dummy data for demonstration
def train_model():
    # Generate dummy data for training (including a "normal" class)
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.choice(list(tumor_types.keys()), size=100)  # Random labels from tumor_types, including 'normal'

    # Split the dummy data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

    # Fit the model on dummy data
    model.fit(X_train, y_train)

# Call the training function when the application starts
train_model()

def tumor_likelihood(image_features):
    weights = {
        'mean_intensity': 0.2,
        'std_intensity': 0.2,
        'cv': 0.1,
        'avg_edge_strength': 0.2,
        'contrast': 0.1,
        'dissimilarity': 0.1,
        'homogeneity': 0.1,
        'energy': 0.1,
    }
    T = sum(w * f for w, f in zip(weights.values(), image_features[:8]))
    return T

def generate_plots(image_array, image_features):
    plots = {}
    
    # Original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off')
    plots['original'] = plot_to_base64(plt)
    
    # Grayscale image
    image_gray = color.rgb2gray(image_array)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plots['grayscale'] = plot_to_base64(plt)
    
    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(image_gray.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Intensity Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid()
    plots['histogram'] = plot_to_base64(plt)
    
    # Edge detection
    edges = filters.sobel(image_gray)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='hot')
    plt.title('Sobel Edge Detection Result')
    plt.axis('off')
    plots['edge_detection'] = plot_to_base64(plt)
    
    # GLCM features
    glcm = graycomatrix((image_gray * 255).astype(np.uint8), distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    features = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    values = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=features, y=values)
    plt.title('GLCM Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid()
    plots['glcm_features'] = plot_to_base64(plt)
    
    # 3D surface plot of image intensities
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, image_gray.shape[1], 1)
    y = np.arange(0, image_gray.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, image_gray, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Surface Plot of Image Intensities')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plots['3d_surface'] = plot_to_base64(plt)
    
    return plots

def plot_to_base64(plt):
    """Converts matplotlib plot to base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
