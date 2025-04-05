import numpy as np
from PIL import Image
from skimage import color, filters, feature, morphology, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
import io
import base64
import pickle
import os
import joblib

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
# Update model initialization with better hyperparameters
model = RandomForestClassifier(
    n_estimators=500,  # Increased from 200
    max_depth=20,      # Increased from 15
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

# Constants for feature extraction
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def extract_image_features(image: np.array):
    """Extract comprehensive features from a given image array and ensure dimensional consistency with trained model."""
    if image.shape[-1] == 4:  # Handle RGBA images
        image = color.rgba2rgb(image)

    # Convert to grayscale and normalize
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))
    
    # Basic statistical features
    mean_intensity = np.mean(image_normalized)
    std_intensity = np.std(image_normalized)
    cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
    skewness = np.mean(((image_normalized - mean_intensity) / std_intensity)**3) if std_intensity > 0 else 0
    kurtosis = np.mean(((image_normalized - mean_intensity) / std_intensity)**4) if std_intensity > 0 else 0
    
    # Percentiles
    percentile_25 = np.percentile(image_normalized, 25)
    percentile_50 = np.percentile(image_normalized, 50)
    percentile_75 = np.percentile(image_normalized, 75)
    
    # Edge features
    edges_sobel = filters.sobel(image_normalized)
    avg_edge_strength = np.mean(edges_sobel)
    std_edge_strength = np.std(edges_sobel)
    
    # Canny edge detection
    edges_canny = feature.canny(image_normalized, sigma=1.0)
    canny_edge_density = np.mean(edges_canny)
    
    # Morphological features
    # Binarize image using Otsu's thresholding
    thresh = filters.threshold_otsu(image_normalized)
    binary = image_normalized > thresh
    
    # Remove noise with morphological operations
    binary_clean = morphology.binary_opening(binary)
    binary_clean = morphology.binary_closing(binary_clean)
    
    # Label connected regions
    labeled_regions = measure.label(binary_clean)
    region_props = measure.regionprops(labeled_regions)
    
    # Get region properties
    if region_props:
        avg_area = np.mean([prop.area for prop in region_props])
        avg_perimeter = np.mean([prop.perimeter for prop in region_props])
        avg_eccentricity = np.mean([prop.eccentricity for prop in region_props])
        avg_solidity = np.mean([prop.solidity for prop in region_props])
        num_regions = len(region_props)
    else:
        avg_area = avg_perimeter = avg_eccentricity = avg_solidity = num_regions = 0
    
    # Texture features using Local Binary Patterns
    lbp = local_binary_pattern(image_normalized, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=LBP_N_POINTS + 2, range=(0, LBP_N_POINTS + 2), density=True)
    
    # GLCM texture features with multiple distances and angles
    glcm_features = []
    for distance in DISTANCES:
        for angle in ANGLES:
            glcm = graycomatrix((image_normalized * 255).astype(np.uint8), 
                                distances=[distance], 
                                angles=[angle], 
                                levels=256, 
                                symmetric=True, 
                                normed=True)
            
            glcm_features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0],
                graycoprops(glcm, 'ASM')[0, 0]
            ])
    
    # Combine all features
    features = [
        mean_intensity, std_intensity, cv, skewness, kurtosis,
        percentile_25, percentile_50, percentile_75,
        avg_edge_strength, std_edge_strength, canny_edge_density,
        avg_area, avg_perimeter, avg_eccentricity, avg_solidity, num_regions
    ]
    
    # Add LBP histogram features
    features.extend(lbp_hist)
    
    # Add GLCM features
    features.extend(glcm_features)
    
    # Ensure the feature vector has exactly 100 dimensions to match the trained model
    feature_dimension = 100
    if len(features) > feature_dimension:
        # If we have more features than expected, select the most important ones
        # Prioritize the first 16 main features, then select from remaining features to reach 100
        main_features = features[:16]
        remaining_needed = feature_dimension - 16
        # Select evenly from LBP and GLCM features
        selected_remaining = features[16:16+remaining_needed]
        features = main_features + selected_remaining
    elif len(features) < feature_dimension:
        # If we have fewer features than expected, pad with zeros
        padding = [0.0] * (feature_dimension - len(features))
        features.extend(padding)
    
    return features

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lung Cancer Classification API! Use the /predict-tumor/ endpoint to upload lung CT images for analysis."}

# Add new imports
from scipy import stats
from scipy import signal  # Add this import
from skimage.measure import shannon_entropy
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def validate_lung_ct(image_array: np.array) -> tuple[bool, str]:
    """
    Validates if the image is a proper lung CT scan using mathematical criteria.
    Returns (is_valid, message)
    """
    try:
        # Check image dimensions
        if len(image_array.shape) != 3:
            return False, "Invalid image dimensions. Expected 3 channels."
            
        # Convert to grayscale
        image_gray = color.rgb2gray(image_array)
        
        # Check image size (typical CT dimensions)
        min_size = 512  # Standard CT dimension
        if image_gray.shape[0] < min_size or image_gray.shape[1] < min_size:
            return False, f"Image too small. Minimum dimension required: {min_size}px"
        
        # Enhanced intensity distribution analysis
        hist, bins = np.histogram(image_gray, bins=256, density=True)
        
        # Smooth histogram for better peak detection
        smoothed_hist = signal.savgol_filter(hist, window_length=11, polyorder=3)
        
        # Find peaks with mathematical criteria
        peaks, properties = signal.find_peaks(
            smoothed_hist,
            height=0.001,  # Minimum height threshold
            distance=20,   # Minimum distance between peaks
            prominence=0.001  # Minimum prominence
        )
        
        # Validate bimodal distribution (air and tissue peaks)
        if len(peaks) < 2:
            return False, "Image lacks characteristic lung CT intensity distribution"
        
        # Calculate peak ratios for air-tissue contrast
        peak_heights = properties['peak_heights']
        peak_ratio = min(peak_heights) / max(peak_heights)
        if peak_ratio > 0.8:  # Peaks should have significant height difference
            return False, "Invalid tissue-air density ratio"
        
        # Enhanced contrast analysis using Michelson contrast
        max_val = np.percentile(image_gray, 99)  # 95th percentile for robustness
        min_val = np.percentile(image_gray, 1)   # 5th percentile for robustness
        michelson_contrast = (max_val - min_val) / (max_val + min_val)
        if michelson_contrast <= 0.4:  # Minimum expected contrast
            return False, f"Insufficient tissue contrast: {michelson_contrast:.2f}"
        
        # Enhanced entropy analysis
        entropy = shannon_entropy(image_gray)
        normalized_entropy = entropy / np.log2(256)  # Normalize by maximum possible entropy
        if normalized_entropy < 0.5:  # At least 50% of maximum possible entropy
            return False, f"Image lacks expected tissue pattern complexity: {normalized_entropy:.2f}"
        
        # Gradient magnitude analysis
        gradient_magnitude = filters.sobel(image_gray)
        gradient_mean = np.mean(gradient_magnitude)
        if gradient_mean < 0.02:  # Minimum expected edge strength
            return False, f"Insufficient structural detail: {gradient_mean:.2f}"
        
        # Statistical validation
        # Calculate Coefficient of Variation (CV)
        mean_intensity = np.mean(image_gray)
        std_intensity = np.std(image_gray)
        cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        if cv < 0.2:  # Minimum expected variation
            return False, f"Insufficient intensity variation: CV = {cv:.2f}"
        
        # Texture analysis using GLCM
        glcm = graycomatrix(
            (image_gray * 255).astype(np.uint8),
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Calculate homogeneity
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        if homogeneity > 0.95:  # Too homogeneous for a CT scan
            return False, f"Image too homogeneous: {homogeneity:.2f}"
        
        # All validations passed
        validation_score = {
            'contrast_score': michelson_contrast,
            'entropy_score': normalized_entropy,
            'gradient_score': gradient_mean,
            'variation_score': cv,
            'texture_score': 1 - homogeneity
        }
        
        # Calculate overall quality score (0-1)
        quality_score = np.mean([
            michelson_contrast,
            normalized_entropy,
            min(gradient_mean * 5, 1),  # Scale gradient score
            min(cv * 2, 1),            # Scale variation score
            1 - homogeneity
        ])
        
        return True, f"Valid lung CT scan (Quality Score: {quality_score:.2f})"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def analyze_tumor_characteristics(image_array: np.array) -> dict:
    """
    Analyzes tumor characteristics using established medical imaging formulas
    """
    image_gray = color.rgb2gray(image_array)
    
    # 1. Density Analysis
    density_score = calculate_density_score(image_gray)
    
    # 2. Shape Analysis
    shape_metrics = calculate_shape_metrics(image_gray)
    
    # 3. Texture Analysis
    texture_scores = calculate_texture_scores(image_gray)
    
    # 4. Growth Pattern Analysis
    growth_pattern = analyze_growth_pattern(image_gray)
    
    return {
        "density_score": density_score,
        "shape_metrics": shape_metrics,
        "texture_scores": texture_scores,
        "growth_pattern": growth_pattern
    }

def calculate_density_score(image_gray: np.array) -> float:
    """
    Calculates tumor density score using Hounsfield Unit approximation
    """
    # Normalize to approximate HU scale (-1000 to +1000)
    hu_approx = (image_gray * 2000) - 1000
    
    # Calculate mean density in tumor region
    thresh = filters.threshold_otsu(image_gray)
    tumor_mask = image_gray > thresh
    
    if np.sum(tumor_mask) > 0:
        mean_density = np.mean(hu_approx[tumor_mask])
        # Normalize to 0-1 scale
        return (mean_density + 1000) / 2000
    return 0.0

def calculate_shape_metrics(image_gray: np.array) -> dict:
    """
    Calculates shape-based metrics using established radiological formulas
    """
    # Segment tumor region
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Calculate region properties
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "sphericity": 0,
            "irregularity": 0,
            "spiculation": 0
        }
    
    # Get largest region (assumed to be tumor)
    tumor_region = max(regions, key=lambda r: r.area)
    
    # Calculate sphericity (1 = perfect sphere)
    surface_area = tumor_region.perimeter
    volume = tumor_region.area
    sphericity = (np.pi ** (1/3)) * (6 * volume) ** (2/3) / surface_area if surface_area > 0 else 0
    
    # Calculate irregularity
    irregularity = tumor_region.perimeter / (2 * np.sqrt(np.pi * tumor_region.area))
    
    # Calculate spiculation (using gradient analysis)
    gradient_mag = filters.sobel(image_gray)
    # Get boundary coordinates using the coords attribute and boolean mask
    coords = tumor_region.coords
    boundary_mask = np.zeros_like(image_gray, dtype=bool)
    boundary_mask[coords[:, 0], coords[:, 1]] = True
    # Erode to get the boundary
    boundary_mask = boundary_mask ^ morphology.binary_erosion(boundary_mask)
    boundary_coords = np.where(boundary_mask)
    spiculation = np.mean(gradient_mag[boundary_coords]) if len(boundary_coords[0]) > 0 else 0
    
    return {
        "sphericity": min(1.0, sphericity),
        "irregularity": min(1.0, irregularity / 2),  # Normalize to 0-1
        "spiculation": min(1.0, spiculation)
    }

def calculate_texture_scores(image_gray: np.array) -> dict:
    """
    Calculates texture-based scores using medical imaging formulas
    """
    # Calculate Hessian matrix-based features
    hessian_matrices = hessian_matrix(image_gray, sigma=1.0)
    eigenvalues = hessian_matrix_eigvals(hessian_matrices)
    
    # Calculate vesselness (Frangi filter response)
    lambda1, lambda2 = eigenvalues
    vesselness = np.zeros_like(image_gray)
    mask = (lambda2 != 0)
    rb = (lambda1[mask] / lambda2[mask]) ** 2
    s2 = lambda1[mask] ** 2 + lambda2[mask] ** 2
    vesselness[mask] = np.exp(-rb / (2 * 0.5)) * (1 - np.exp(-s2 / (2 * 0.2)))
    
    # GLCM-based texture scores
    glcm = graycomatrix(
        (image_gray * 255).astype(np.uint8),
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    return {
        "heterogeneity": float(graycoprops(glcm, 'contrast').mean()),
        "uniformity": float(graycoprops(glcm, 'energy').mean()),
        "vesselness": float(np.mean(vesselness))
    }

def analyze_growth_pattern(image_gray: np.array) -> dict:
    """
    Analyzes tumor growth pattern using radiological metrics
    """
    # Edge analysis for infiltrative vs. expansive growth
    edges = filters.sobel(image_gray)
    
    # Calculate edge sharpness
    edge_sharpness = np.mean(edges[edges > np.percentile(edges, 90)])
    
    # Calculate boundary gradient
    gradient_magnitude = filters.sobel(image_gray)
    boundary_strength = np.mean(gradient_magnitude)
    
    return {
        "edge_sharpness": min(1.0, edge_sharpness),
        "boundary_strength": min(1.0, boundary_strength),
        "infiltrative_score": 1.0 - min(1.0, edge_sharpness)  # Lower sharpness indicates infiltrative growth
    }

def predict_tumor_type(characteristics: dict) -> dict:
    """
    Predicts tumor type based on calculated characteristics using medical formulas
    """
    # Initialize scores for each tumor type
    scores = {
        'adenocarcinoma': 0.0,
        'squamous_cell': 0.0,
        'large_cell': 0.0,
        'small_cell': 0.0,
        'metastatic': 0.0
    }
    
    # Adenocarcinoma characteristics
    scores['adenocarcinoma'] = calculate_adenocarcinoma_score(characteristics)
    
    # Squamous cell characteristics
    scores['squamous_cell'] = calculate_squamous_cell_score(characteristics)
    
    # Large cell characteristics
    scores['large_cell'] = calculate_large_cell_score(characteristics)
    
    # Small cell characteristics
    scores['small_cell'] = calculate_small_cell_score(characteristics)
    
    # Metastatic characteristics
    scores['metastatic'] = calculate_metastatic_score(characteristics)
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}
    
    # Calculate confidence score
    confidence = max(scores.values())
    predicted_type = max(scores.items(), key=lambda x: x[1])[0]
    
    return {
        "predicted_type": predicted_type,
        "confidence": confidence,
        "scores": scores
    }

# Tumor type-specific scoring functions
def calculate_adenocarcinoma_score(chars: dict) -> float:
    """
    Ground-glass opacity, lepidic growth pattern, irregular margins
    Updated weights based on medical literature
    """
    return (
        (1 - chars['density_score']) * 0.35 +  # Ground-glass opacity (increased weight)
        chars['texture_scores']['heterogeneity'] * 0.25 +  # Texture heterogeneity
        chars['shape_metrics']['irregularity'] * 0.25 +    # Irregular margins
        (1 - chars['shape_metrics']['sphericity']) * 0.15  # Non-spherical shape
    )

def calculate_squamous_cell_score(chars: dict) -> float:
    """
    Central location, cavitation, thick walls
    """
    return (
        chars['density_score'] * 0.35 +                # Higher density
        chars['shape_metrics']['sphericity'] * 0.25 +  # More spherical
        chars['texture_scores']['uniformity'] * 0.25 + # More uniform
        chars['growth_pattern']['boundary_strength'] * 0.15  # Well-defined margins
    )

def calculate_large_cell_score(chars: dict) -> float:
    """
    Large size, necrotic center, well-defined margins
    """
    return (
        chars['shape_metrics']['sphericity'] * 0.25 +           # More spherical
        (1 - chars['texture_scores']['uniformity']) * 0.35 +    # Less uniform (necrotic)
        chars['growth_pattern']['boundary_strength'] * 0.25 +   # Well-defined margins
        chars['density_score'] * 0.15                          # Higher density
    )

def calculate_small_cell_score(chars: dict) -> float:
    """
    Central location, rapid growth, vascular invasion
    """
    return (
        chars['texture_scores']['vesselness'] * 0.35 +          # Vascular involvement
        chars['growth_pattern']['infiltrative_score'] * 0.25 +  # Infiltrative pattern
        chars['density_score'] * 0.25 +                        # Higher density
        (1 - chars['shape_metrics']['sphericity']) * 0.15      # Less spherical
    )

def calculate_metastatic_score(chars: dict) -> float:
    """
    Multiple nodules, well-defined margins, round shape
    """
    return (
        chars['shape_metrics']['sphericity'] * 0.35 +           # More spherical
        chars['growth_pattern']['boundary_strength'] * 0.25 +   # Well-defined margins
        chars['texture_scores']['uniformity'] * 0.25 +         # More uniform
        chars['density_score'] * 0.15                          # Variable density
    )

def generate_plots(image_array, image_features, class_probabilities):
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
    
    # Binary segmentation
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    binary_clean = morphology.binary_closing(binary_clean)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Thresholded')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(binary_clean, cmap='gray')
    plt.title('Morphological Processing')
    plt.axis('off')
    plots['segmentation'] = plot_to_base64(plt)
    
    # Classification probabilities
    plt.figure(figsize=(12, 6))
    classes = list(class_probabilities.keys())
    probs = list(class_probabilities.values())
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]
    
    # Create bar chart
    bars = plt.bar(classes, probs, color='skyblue')
    plt.xlabel('Tumor Type')
    plt.ylabel('Probability')
    plt.title('Classification Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.01, 
                 f'{prob:.2f}', 
                 ha='center', va='bottom')
    
    plots['classification_probabilities'] = plot_to_base64(plt)
    
    # 3D surface plot of image intensities - more efficient implementation
    # Downsample the image to make 3D plot more efficient
    factor = max(1, int(min(image_gray.shape) / 50))
    downsampled = image_gray[::factor, ::factor]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, downsampled.shape[1], 1)
    y = np.arange(0, downsampled.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, downsampled, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Surface Plot of Image Intensities (Downsampled)')
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

# Update your predict_tumor endpoint
@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Validate CT scan
        is_valid, validation_message = validate_lung_ct(image_array)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        # Analyze tumor characteristics
        characteristics = analyze_tumor_characteristics(image_array)
        
        # Predict tumor type
        prediction_results = predict_tumor_type(characteristics)
        
        # Generate visualization plots
        plots = generate_plots(image_array, characteristics, prediction_results['scores'])
        
        return {
            "tumor_type": tumor_types.get(prediction_results['predicted_type'], "Unknown"),
            "confidence": prediction_results['confidence'],
            "characteristics": characteristics,
            "class_probabilities": prediction_results['scores'],
            "plots": plots
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis error: {str(e)}"
        )
