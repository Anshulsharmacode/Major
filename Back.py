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

# Add new imports
from scipy.stats import entropy
from skimage.feature import peak_local_max
from skimage.filters import gaussian

# Enhanced model parameters for better accuracy
model = RandomForestClassifier(
    n_estimators=1000,    # Increased for better generalization
    max_depth=25,         # Slightly increased
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    class_weight='balanced_subsample',  # Better handling of imbalanced data
    n_jobs=-1             # Use all CPU cores
)

# Enhanced feature extraction constants
LBP_RADIUS = 10           # Increased for better texture capture
LBP_N_POINTS = 12 * LBP_RADIUS
DISTANCES = [1, 2, 3, 4, 5]  # More distances for better texture analysis
ANGLES = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]  # More angles

def extract_image_features(image: np.array):
    """Enhanced feature extraction with advanced image analysis"""
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)

    # Convert to grayscale and enhance contrast
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))
    
    # Apply Gaussian smoothing to reduce noise
    image_smooth = gaussian(image_normalized, sigma=1.0)
    
    # Enhanced statistical features
    features = []
    
    # Multi-scale analysis
    for sigma in [1, 2, 4]:
        img_scale = gaussian(image_normalized, sigma=sigma)
        
        # Basic statistics
        features.extend([
            np.mean(img_scale),
            np.std(img_scale),
            entropy(img_scale.ravel()),
            np.percentile(img_scale, [10, 25, 50, 75, 90]).tolist()
        ])
        
        # Edge features at multiple scales
        edges_sobel = filters.sobel(img_scale)
        edges_canny = feature.canny(img_scale, sigma=sigma)
        
        features.extend([
            np.mean(edges_sobel),
            np.std(edges_sobel),
            np.mean(edges_canny),
            np.sum(edges_canny) / edges_canny.size  # Edge density
        ])

    # Enhanced morphological analysis
    thresh = filters.threshold_otsu(image_normalized)
    binary = image_normalized > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Region analysis
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if regions:
        # Sort regions by area to focus on the main tumor region
        regions.sort(key=lambda x: x.area, reverse=True)
        main_region = regions[0]  # Largest region (assumed to be tumor)
        
        # Enhanced shape features
        features.extend([
            main_region.area,
            main_region.perimeter,
            main_region.eccentricity,
            main_region.solidity,
            main_region.extent,
            main_region.euler_number,
            main_region.orientation,
            main_region.major_axis_length,
            main_region.minor_axis_length,
            main_region.perimeter / (2 * np.sqrt(np.pi * main_region.area))  # Circularity
        ])
    else:
        features.extend([0] * 10)  # Padding for cases without regions

    # Enhanced texture analysis
    # Multi-scale LBP
    for radius in [2, 3, 4]:
        lbp = local_binary_pattern(image_normalized, 8 * radius, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
        features.extend(lbp_hist)

    # Enhanced GLCM features
    glcm_features = []
    for distance in DISTANCES:
        for angle in ANGLES:
            glcm = graycomatrix(
                (image_normalized * 255).astype(np.uint8),
                distances=[distance],
                angles=[angle],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Calculate additional GLCM properties
            glcm_features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0],
                graycoprops(glcm, 'ASM')[0, 0]
            ])
    
    features.extend(glcm_features)

    # Ensure consistent feature dimension
    feature_dimension = 100
    if len(features) > feature_dimension:
        features = features[:feature_dimension]
    elif len(features) < feature_dimension:
        features.extend([0] * (feature_dimension - len(features)))

    return np.array(features)

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

def extract_advanced_features(image: np.array) -> np.array:
    """Enhanced feature extraction with comprehensive image analysis"""
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    
    # Convert to grayscale and normalize
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))
    
    features = []
    
    # 1. Enhanced Statistical Features
    intensity_stats = [
        np.mean(image_normalized),
        np.std(image_normalized),
        skew(image_normalized.ravel()),
        kurtosis(image_normalized.ravel()),
        shannon_entropy(image_normalized)
    ]
    features.extend(intensity_stats)
    
    # 2. Multi-scale Texture Analysis
    for radius in [1, 2, 3]:
        # LBP features
        lbp = local_binary_pattern(image_normalized, 8 * radius, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
        features.extend(lbp_hist)
        
        # GLCM features at multiple angles
        glcm = graycomatrix(
            (image_normalized * 255).astype(np.uint8),
            distances=[radius],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        glcm_features = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'ASM').mean()
        ]
        features.extend(glcm_features)
    
    # 3. Enhanced Edge and Gradient Analysis
    # HOG features
    hog_features = hog(
        image_normalized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False
    )
    features.extend(hog_features[:20])  # Take first 20 HOG features
    
    # Multi-scale gradient analysis
    for sigma in [1, 2, 4]:
        grad_mag = filters.sobel(gaussian(image_normalized, sigma=sigma))
        grad_features = [
            np.mean(grad_mag),
            np.std(grad_mag),
            np.percentile(grad_mag, [25, 50, 75])
        ]
        features.extend(grad_features)
    
    # 4. Frequency Domain Features
    f_transform = fft2(image_normalized)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    freq_features = [
        np.mean(magnitude_spectrum),
        np.std(magnitude_spectrum),
        np.max(magnitude_spectrum),
        np.sum(magnitude_spectrum > np.mean(magnitude_spectrum))
    ]
    features.extend(freq_features)
    
    # 5. Shape and Region Properties
    thresh = filters.threshold_otsu(image_normalized)
    binary = image_normalized > thresh
    binary_clean = morphology.remove_small_objects(
        morphology.binary_opening(binary),
        min_size=50
    )
    
    regions = measure.regionprops(measure.label(binary_clean))
    if regions:
        main_region = max(regions, key=lambda r: r.area)
        shape_features = [
            main_region.area,
            main_region.perimeter,
            main_region.eccentricity,
            main_region.solidity,
            main_region.extent,
            main_region.euler_number,
            main_region.equivalent_diameter
        ]
    else:
        shape_features = [0] * 7
    features.extend(shape_features)
    
    return np.array(features)

# Add these imports at the top
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_local

# Add these constants after other constants
NORMAL_TISSUE_THRESHOLDS = {
    'entropy_min': 0.6,
    'contrast_min': 0.15,
    'homogeneity_max': 0.9,
    'vessel_density_min': 0.01,
    'texture_uniformity_max': 0.8
}

def analyze_normal_tissue(image_array: np.array) -> tuple[bool, float]:
    """
    Analyzes if the image shows normal lung tissue
    Returns (is_normal, confidence)
    """
    image_gray = color.rgb2gray(image_array)
    
    # 1. Tissue Pattern Analysis
    entropy_score = shannon_entropy(image_gray) / np.log2(256)
    
    # 2. Vessel Analysis
    hessian_matrices = hessian_matrix(image_gray, sigma=1.0)
    eigenvalues = hessian_matrix_eigvals(hessian_matrices)
    vessel_density = np.mean(np.abs(eigenvalues[0])) / np.mean(np.abs(eigenvalues[1]))
    
    # 3. Texture Analysis
    glcm = graycomatrix(
        (image_gray * 255).astype(np.uint8),
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    uniformity = graycoprops(glcm, 'energy').mean()
    
    # 4. Local Intensity Variation
    local_thresh = threshold_local(image_gray, block_size=35, method='gaussian')
    intensity_variation = np.mean(np.abs(image_gray - local_thresh))
    
    # Calculate normal tissue probability
    normal_scores = [
        1.0 if entropy_score > NORMAL_TISSUE_THRESHOLDS['entropy_min'] else entropy_score / NORMAL_TISSUE_THRESHOLDS['entropy_min'],
        1.0 if contrast > NORMAL_TISSUE_THRESHOLDS['contrast_min'] else contrast / NORMAL_TISSUE_THRESHOLDS['contrast_min'],
        1.0 if homogeneity < NORMAL_TISSUE_THRESHOLDS['homogeneity_max'] else NORMAL_TISSUE_THRESHOLDS['homogeneity_max'] / homogeneity,
        1.0 if vessel_density > NORMAL_TISSUE_THRESHOLDS['vessel_density_min'] else vessel_density / NORMAL_TISSUE_THRESHOLDS['vessel_density_min'],
        1.0 if uniformity < NORMAL_TISSUE_THRESHOLDS['texture_uniformity_max'] else NORMAL_TISSUE_THRESHOLDS['texture_uniformity_max'] / uniformity
    ]
    
    normal_probability = np.mean(normal_scores)
    is_normal = normal_probability > 0.7
    
    return is_normal, normal_probability

# Update the predict_tumor endpoint
@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # First, check if it's normal tissue
        is_normal, normal_confidence = analyze_normal_tissue(image_array)
        
        if is_normal and normal_confidence > 0.8:
            return {
                "tumor_type": "Normal Lung Tissue",
                "confidence": float(normal_confidence),
                "analysis": {
                    "is_normal": True,
                    "normal_confidence": float(normal_confidence)
                }
            }
        
        # If not clearly normal, proceed with tumor analysis
        features = extract_advanced_features(image_array)
        
        # Ensure model is trained
        if not hasattr(model, "classes_"):
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Please train the model first."
            )
        
        # Get prediction probabilities
        probabilities = model.predict_proba([features])[0]
        predicted_class = model.classes_[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        
        # Adjust confidence if there's significant normal tissue probability
        if normal_confidence > 0.5:
            confidence = confidence * (1 - normal_confidence)
        
        # Generate detailed analysis
        analysis_results = analyze_tumor_characteristics(image_array)
        
        return {
            "tumor_type": tumor_types.get(predicted_class, "Unknown"),
            "confidence": confidence,
            "normal_tissue_probability": float(normal_confidence),
            "analysis": {
                "is_normal": is_normal,
                "normal_confidence": float(normal_confidence),
                "tumor_characteristics": analysis_results,
                "feature_importance": dict(zip(
                    ["feature_" + str(i) for i in range(len(features))],
                    model.feature_importances_.tolist()
                ))
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis error: {str(e)}"
        )
