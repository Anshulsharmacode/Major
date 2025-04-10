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

# Add missing imports for advanced feature extraction
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import peak_local_max, hessian_matrix, hessian_matrix_eigvals, hog
from skimage.filters import gaussian, threshold_local
from skimage.measure import shannon_entropy
from scipy import signal, stats
from scipy.fft import fft2, fftshift
from skimage import transform
from datetime import datetime

# Initialize FastAPI
app = FastAPI()

# Tumor types mapping (including normal image type)
tumor_types = {
    'adenocarcinoma': 'Non-Small Cell Lung Cancer (NSCLC) - Adenocarcinoma',
    'squamous_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Squamous Cell Carcinoma',
    'large_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Large Cell Carcinoma',
    'small_cell': 'Small Cell Lung Cancer (SCLC)',
    'metastatic': 'Secondary Lung Tumors - Metastatic',
    'normal': 'Normal Lung Tissue'
}


model = RandomForestClassifier(
    n_estimators=2000,          # Increased number of trees
    max_depth=30,               # Increased depth
    min_samples_split=3,        # Reduced to allow more splits
    min_samples_leaf=1,         # Reduced to allow more detailed patterns
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)

# Enhanced feature extraction constants
LBP_RADIUS = 10
LBP_N_POINTS = 12 * LBP_RADIUS
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]

# Enhanced tissue classification thresholds based on your descriptions
NORMAL_TISSUE_THRESHOLDS = {
    'entropy_min': 0.6,
    'contrast_min': 0.15,
    'homogeneity_max': 0.9,
    'vessel_density_min': 0.01,
    'texture_uniformity_max': 0.8
}

# Tumor-specific characteristic thresholds
TUMOR_CHARACTERISTICS = {
    'adenocarcinoma': {
        'peripheral_location': True,
        'spiculation_min': 0.6,
        'ground_glass_opacity': True,
        'sphericity_max': 0.7
    },
    'squamous_cell': {
        'central_location': True,
        'cavitation': True,
        'airway_compression': True,
        'necrosis': True
    },
    'large_cell': {
        'peripheral_location': True,
        'large_size_min': 0.7,
        'irregular_shape': True,
        'infiltrative_growth_min': 0.7
    },
    'small_cell': {
        'central_location': True,
        'rapid_growth': True,
        'lymph_node_involvement': True,
        'vascular_compression': True
    },
    'metastatic': {
        'multiple_nodules': True,
        'well_circumscribed': True,
        'random_distribution': True,
        'variable_sizes': True
    },
    'hamartoma': {
        'peripheral_location': True,
        'popcorn_calcification': True,
        'well_circumscribed': True,
        'slow_growth': True
    },
    'pulmonary_adenoma': {
        'small_size_max': 0.3,
        'homogeneous_density': True,
        'smooth_margin': True,
        'no_invasion': True
    }
}

# Add functions to analyze specific tumor patterns
def detect_location_pattern(image_gray: np.array) -> dict:
    """
    Detects if the potential tumor is central or peripheral
    """
    # Calculate distance from center for potential tumor regions
    h, w = image_gray.shape
    center_y, center_x = h // 2, w // 2
    
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Label regions
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "is_central": False,
            "is_peripheral": False,
            "central_score": 0.0,
            "peripheral_score": 0.0
        }
    
    # Calculate distance of each region from the center
    region_distances = []
    for region in regions:
        y, x = region.centroid
        # Normalize distance to [0, 1] where 0 is center, 1 is corner
        dist_to_center = np.sqrt(((y - center_y) / (h/2))**2 + ((x - center_x) / (w/2))**2)
        region_distances.append((region, dist_to_center))
    
    # Sort by region size (descending)
    sorted_regions = sorted(region_distances, key=lambda x: x[0].area, reverse=True)
    
    # Take the largest regions that likely represent tumors
    main_regions = sorted_regions[:min(3, len(sorted_regions))]
    
    # Calculate location scores
    peripheral_score = np.mean([dist for _, dist in main_regions])
    central_score = 1.0 - peripheral_score
    
    return {
        "is_central": central_score > 0.6,
        "is_peripheral": peripheral_score > 0.6,
        "central_score": float(central_score),
        "peripheral_score": float(peripheral_score)
    }
    
def detect_multiple_nodules(image_gray: np.array) -> dict:
    """
    Detects if there are multiple nodules (important for metastatic tumors)
    """
    # Multi-scale segmentation for different nodule sizes
    nodule_masks = []
    for sigma in [1, 2, 3]:
        # Smooth image at different scales
        smoothed = gaussian(image_gray, sigma=sigma)
        thresh = filters.threshold_otsu(smoothed)
        binary = smoothed > thresh
        binary_clean = morphology.binary_opening(binary)
        binary_clean = morphology.remove_small_objects(binary_clean, min_size=25)
        nodule_masks.append(binary_clean)
    
    # Combine masks
    combined_mask = np.logical_or.reduce(nodule_masks)
    
    # Enhanced nodule detection
    labeled_regions = measure.label(combined_mask)
    regions = measure.regionprops(labeled_regions)
    
    # More sophisticated nodule filtering
    valid_nodules = []
    for region in regions:
        # Calculate shape features
        perimeter = region.perimeter
        area = region.area
        if area > 0 and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 999
            
            # Filter based on multiple criteria
            if (30 < area < image_gray.size * 0.1 and  # Size constraints
                circularity > 0.4 and                   # Roundness constraint
                aspect_ratio < 2.5):                    # Shape constraint
                valid_nodules.append(region)
    
    nodule_count = len(valid_nodules)
    
    # Enhanced metastatic pattern analysis
    if nodule_count >= 2:
        # Calculate spatial distribution
        centroids = [r.centroid for r in valid_nodules]
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                             (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        # Analyze spatial distribution
        spatial_randomness = np.std(distances) / np.mean(distances) if distances else 0
        
        # Analyze size distribution
        sizes = [r.area for r in valid_nodules]
        size_cv = np.std(sizes) / np.mean(sizes) if sizes else 0
        
        # Analyze intensity distribution
        intensities = [np.mean(image_gray[r.coords[:, 0], r.coords[:, 1]]) for r in valid_nodules]
        intensity_cv = np.std(intensities) / np.mean(intensities) if intensities else 0
        
        # Analyze shape consistency
        circularities = [4 * np.pi * r.area / (r.perimeter * r.perimeter) for r in valid_nodules]
        shape_consistency = 1 - (np.std(circularities) / np.mean(circularities) if circularities else 0)
    else:
        spatial_randomness = size_cv = intensity_cv = 0
        shape_consistency = 1
    
    # Enhanced metastatic scoring
    metastatic_score = (
        min(1.0, nodule_count / 4) * 0.3 +          # Number of nodules
        min(1.0, size_cv) * 0.2 +                   # Size variation
        min(1.0, spatial_randomness) * 0.2 +        # Spatial distribution
        min(1.0, intensity_cv) * 0.15 +             # Intensity variation
        shape_consistency * 0.15                     # Shape consistency
    )
    
    return {
        "nodule_count": nodule_count,
        "has_multiple_nodules": nodule_count > 1,
        "size_variation": float(size_cv),
        "spatial_randomness": float(spatial_randomness),
        "intensity_variation": float(intensity_cv),
        "shape_consistency": float(shape_consistency),
        "metastatic_pattern_score": float(metastatic_score)
    }
    
def detect_cavitation(image_gray: np.array) -> dict:
    """
    Detects cavitation pattern (hollow area within tumor, common in squamous cell carcinoma)
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Label regions
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "has_cavitation": False,
            "cavitation_score": 0.0
        }
    
    # Get largest region (assumed to be the main tumor)
    main_region = max(regions, key=lambda r: r.area)
    
    # Check for holes within the tumor (potential cavitation)
    y0, x0, y1, x1 = main_region.bbox
    tumor_mask = np.zeros_like(image_gray, dtype=bool)
    tumor_mask[y0:y1, x0:x1] = main_region.image
    
    # Invert the mask to get potential holes
    holes = morphology.remove_small_holes(tumor_mask) & ~tumor_mask
    
    # Calculate cavitation metrics
    hole_area = np.sum(holes)
    tumor_area = main_region.area
    cavitation_ratio = hole_area / tumor_area if tumor_area > 0 else 0
    
    return {
        "has_cavitation": cavitation_ratio > 0.05,
        "cavitation_score": float(min(1.0, cavitation_ratio * 5))  # Normalize to [0, 1]
    }

def detect_ground_glass_opacity(image_gray: np.array) -> dict:
    """
    Detects ground glass opacity pattern (hazy areas that don't obscure vessels, common in adenocarcinoma)
    """
    # Calculate local standard deviation to identify hazy areas
    local_std = np.zeros_like(image_gray)
    footprint = morphology.disk(3)
    
    # Pad image to avoid edge effects
    padded = np.pad(image_gray, 3, mode='reflect')
    
    # Calculate local standard deviation manually (using neighborhood operations)
    for i in range(3, padded.shape[0] - 3):
        for j in range(3, padded.shape[1] - 3):
            neighborhood = padded[i-3:i+4, j-3:j+4]
            local_std[i-3, j-3] = np.std(neighborhood)
    
    # Ground glass areas have low local standard deviation but higher than air
    air_mask = image_gray < np.percentile(image_gray, 10)
    tissue_mask = image_gray > np.percentile(image_gray, 25)
    
    # Potential GGO areas: moderate intensity but low texture variation
    ggo_mask = (
        (image_gray > np.percentile(image_gray, 20)) & 
        (image_gray < np.percentile(image_gray, 60)) & 
        (local_std < np.percentile(local_std[tissue_mask], 30))
    )
    
    # Calculate vessel overlay in potential GGO areas
    edges = filters.sobel(image_gray)
    vessel_mask = edges > np.percentile(edges, 80)
    vessels_in_ggo = np.logical_and(ggo_mask, vessel_mask)
    
    # Calculate GGO metrics
    ggo_area = np.sum(ggo_mask)
    image_area = image_gray.size
    ggo_ratio = ggo_area / image_area if image_area > 0 else 0
    
    # Calculate vessel preservation score
    vessel_preservation = np.sum(vessels_in_ggo) / np.sum(vessel_mask) if np.sum(vessel_mask) > 0 else 0
    
    return {
        "has_ground_glass_opacity": ggo_ratio > 0.05,
        "ggo_score": float(min(1.0, ggo_ratio * 10)),
        "vessel_preservation": float(vessel_preservation)
    }

def detect_spiculation(image_gray: np.array) -> dict:
    
    """
    Detects spiculated margins (spiky, sunburst pattern common in malignant tumors, especially adenocarcinoma)
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Label regions
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "has_spiculation": False,
            "spiculation_score": 0.0
        }
    
    # Get largest region (assumed to be the main tumor)
    main_region = max(regions, key=lambda r: r.area)
    
    # Calculate perimeter/area ratio (rough measure of boundary complexity)
    perimeter = main_region.perimeter
    area = main_region.area
    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
    
    # Calculate more detailed spiculation metrics using gradient analysis
    y0, x0, y1, x1 = main_region.bbox
    tumor_mask = np.zeros_like(image_gray, dtype=bool)
    tumor_mask[y0:y1, x0:x1] = main_region.image
    
    # Dilate and subtract to get the boundary region
    boundary = morphology.binary_dilation(tumor_mask, morphology.disk(2)) & ~tumor_mask
    
    # Calculate gradient in the boundary region
    gradient = filters.sobel(image_gray)
    boundary_gradient = gradient[boundary]
    
    # High gradient variance indicates spiculation
    gradient_std = np.std(boundary_gradient) if boundary_gradient.size > 0 else 0
    
    # Combine metrics for a spiculation score
    spiculation_score = (compactness - 1) * 0.5 + gradient_std * 2
    
    return {
        "has_spiculation": spiculation_score > 0.6,
        "spiculation_score": float(min(1.0, spiculation_score))
    }
    
def detect_lymph_node_pattern(image_gray: np.array) -> dict:
    """
    Detects patterns suggestive of lymph node involvement
    """
    # Segment potential lymph node regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Look for rounded structures in typical lymph node locations
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    lymph_node_candidates = []
    for region in regions:
        # Lymph nodes are typically round and of certain size
        if region.area > 100 and region.area < 1000:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            if circularity > 0.7:  # Relatively round
                lymph_node_candidates.append(region)
    
    involvement_score = min(1.0, len(lymph_node_candidates) / 5)
    
    return {
        "has_lymph_node_involvement": len(lymph_node_candidates) > 0,
        "involvement_score": float(involvement_score),
        "node_count": len(lymph_node_candidates)
    }

def detect_vascular_invasion(image_gray: np.array) -> dict:
    """
    Detects patterns suggestive of vascular invasion
    """
    # Enhance vessel-like structures
    vesselness_filter = filters.frangi(image_gray)
    
    # Look for tumor contact with vessels
    thresh = filters.threshold_otsu(image_gray)
    tumor_mask = image_gray > thresh
    tumor_mask = morphology.binary_opening(tumor_mask)
    
    # Dilate tumor mask to check for vessel contact
    contact_region = morphology.binary_dilation(tumor_mask, morphology.disk(3)) & ~tumor_mask
    
    # Check vessel presence in contact region
    vessel_contact = np.sum(vesselness_filter[contact_region]) / np.sum(contact_region) if np.any(contact_region) else 0
    
    invasion_score = min(1.0, vessel_contact * 5)
    
    return {
        "has_vascular_invasion": invasion_score > 0.5,
        "invasion_score": float(invasion_score)
    }
    
def analyze_tumor_characteristics(image_array: np.array) -> dict:
    """
    Comprehensively analyzes tumor characteristics using established medical imaging formulas
    """
    image_gray = color.rgb2gray(image_array)
    
    # Basic characteristics
    density_score = calculate_density_score(image_gray)
    shape_metrics = calculate_shape_metrics(image_gray)
    texture_scores = calculate_texture_scores(image_gray)
    growth_pattern = analyze_growth_pattern(image_gray)
    
    # Specific tumor pattern detection
    location = detect_location_pattern(image_gray)
    nodule_analysis = detect_multiple_nodules(image_gray)
    cavitation = detect_cavitation(image_gray)
    ground_glass = detect_ground_glass_opacity(image_gray)
    spiculation = detect_spiculation(image_gray)
    
    # Calculate tumor-specific scores based on characteristic patterns
    tumor_type_scores = {}
    
    # Adenocarcinoma score (peripheral, GGO, spiculated)
    tumor_type_scores["adenocarcinoma"] = (
        location["peripheral_score"] * 0.35 +  # Increase weight for peripheral location
        ground_glass["ggo_score"] * 0.35 +     # Maintain high weight for GGO
        spiculation["spiculation_score"] * 0.30 # Keep spiculation as important factor
    )
    
    # Squamous cell score (central, cavitation)
    tumor_type_scores["squamous_cell"] = (
        location["central_score"] * 0.40 +
        cavitation["cavitation_score"] * 0.25
    )
    
    # Large cell score (large, irregular, infiltrative)
    tumor_type_scores["large_cell"] = (
        shape_metrics["max_diameter"] * 0.35 +     # Increase emphasis on size
        (1.0 - shape_metrics["irregularity"]) * 0.35 + # Well-defined borders
        location["peripheral_score"] * 0.30        # Peripheral location
    )
    
    # Small cell score (central, high density)
    small_cell_density = density_score * 1.2  # Small cell typically has higher density
    lymph_node_involvement = detect_lymph_node_pattern(image_gray)
    vascular_invasion = detect_vascular_invasion(image_gray)
    
    # Updated small cell score calculation
    tumor_type_scores["small_cell"] = (
        location["central_score"] * 0.3 +
        small_cell_density * 0.3 +
        lymph_node_involvement["involvement_score"] * 0.2 +
        vascular_invasion["invasion_score"] * 0.2
    )
    
    # Metastatic score (multiple, well-circumscribed nodules)
    tumor_type_scores["metastatic"] = (
        nodule_analysis["metastatic_pattern_score"] * 0.7 +
        (1.0 - growth_pattern["infiltrative_score"]) * 0.3
    )
    
    # # Hamartoma score (well-circumscribed, calcification pattern)
    # tumor_type_scores["hamartoma"] = (
    #     (1.0 - spiculation["spiculation_score"]) * 0.5 +
    #     shape_metrics["sphericity"] * 0.5
    # )
    
    # # Pulmonary adenoma score (small, homogeneous, smooth margins)
    # tumor_type_scores["pulmonary_adenoma"] = (
    #     (1.0 - texture_scores["heterogeneity"]) * 0.5 +
    #     (1.0 - spiculation["spiculation_score"]) * 0.5
    # )
    
    return {
        "density_score": density_score,
        "shape_metrics": shape_metrics,
        "texture_scores": texture_scores,
        "growth_pattern": growth_pattern,
        "location": location,
        "nodule_analysis": nodule_analysis,
        "cavitation": cavitation,
        "ground_glass": ground_glass,
        "spiculation": spiculation,
        "tumor_type_scores": tumor_type_scores
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
        features.extend(lbp_hist.tolist())  # Convert to list before extending
        
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
    # HOG features - Ensure consistent size by resizing image if needed
    resized_image = transform.resize(image_normalized, (128, 128))
    hog_features = hog(
        resized_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True  # Ensure we get a 1D array
    )
    # Take first 20 HOG features or pad with zeros if less than 20
    hog_features = np.pad(hog_features[:20], (0, max(0, 20 - len(hog_features))))
    features.extend(hog_features.tolist())  # Convert to list before extending
    
    # 4. Multi-scale gradient analysis
    for sigma in [1, 2, 4]:
        grad_mag = filters.sobel(gaussian(image_normalized, sigma=sigma))
        grad_features = [
            np.mean(grad_mag),
            np.std(grad_mag),
        ]
        # Add percentiles as separate values
        grad_features.extend(np.percentile(grad_mag, [25, 50, 75]).tolist())
        features.extend(grad_features)
    
    # 5. Frequency Domain Features
    f_transform = fft2(image_normalized)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    freq_features = [
        np.mean(magnitude_spectrum),
        np.std(magnitude_spectrum),
        np.max(magnitude_spectrum),
        float(np.sum(magnitude_spectrum > np.mean(magnitude_spectrum)))  # Convert to float
    ]
    features.extend(freq_features)
    
    # 6. Shape and Region Properties
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
            float(main_region.area),
            float(main_region.perimeter),
            float(main_region.eccentricity),
            float(main_region.solidity),
            float(main_region.extent),
            float(main_region.euler_number),
            float(main_region.equivalent_diameter)
        ]
    else:
        shape_features = [0.0] * 7
    features.extend(shape_features)
    
    # Convert all features to float and ensure 1D array
    features = [float(f) if not isinstance(f, (int, float)) else f for f in features]
    return np.array(features, dtype=np.float64)

# Add these functions before the predict_tumor endpoint

def validate_lung_ct(image_array: np.array) -> tuple:
    """
    Validates if the image is likely a lung CT scan
    Returns (is_valid, message)
    """
    # Convert to grayscale if not already
    if len(image_array.shape) > 2 and image_array.shape[2] > 1:
        image_gray = color.rgb2gray(image_array)
    else:
        image_gray = image_array
    
    # Check image dimensions
    if image_array.shape[0] < 100 or image_array.shape[1] < 100:
        return False, "Image resolution too low for accurate analysis."
    
    # Check for extreme brightness/darkness
    if np.mean(image_gray) > 0.9 or np.mean(image_gray) < 0.1:
        return False, "Image is too bright or too dark to be a valid CT scan."
    
    # Check for histogram distribution (CT scans have distinctive distributions)
    hist, _ = np.histogram(image_gray, bins=50)
    hist_normalized = hist / np.sum(hist)
    
    # Check if image has the bimodal distribution typical of lung CTs
    # (air in lungs is dark, tissue/bone is bright)
    peaks = peak_local_max(hist_normalized, min_distance=5)
    if len(peaks) < 2:
        return False, "Image lacks the typical intensity distribution of a lung CT scan."
    
    # Check for presence of distinctive lung tissue patterns
    # CT scans have specific edge patterns from lung structures
    edges = filters.sobel(image_gray)
    if np.mean(edges) < 0.01:
        return False, "Image lacks the typical structural patterns of a lung CT scan."
    
    return True, "Valid lung CT scan detected."

def analyze_normal_tissue(image_array: np.array) -> dict:
    """
    Analyzes if the image shows normal lung tissue without tumors.
    Uses multiple criteria specific to normal lung CT appearance.
    """
    image_gray = color.rgb2gray(image_array) if len(image_array.shape) > 2 else image_array
    
    # 1. Normal Lung Density Distribution
    hist, bins = np.histogram(image_gray, bins=50, density=True)
    
    # Normal lungs should show clear bimodal distribution (air and tissue)
    peaks, _ = signal.find_peaks(hist, distance=5)
    peak_values = hist[peaks]
    if len(peaks) >= 2:
        bimodal_score = 1.0
    else:
        bimodal_score = 0.3

    # 2. Bronchial Tree Pattern
    # Normal lungs show branching vessel patterns
    edges = filters.sobel(image_gray)
    thin_edges = feature.canny(image_gray, sigma=2)
    
    # Calculate branching pattern score using skeletonize and label
    skeleton = morphology.skeletonize(thin_edges)
    # Create a structuring element for branch points
    branch_struct = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    # Use binary_erosion and binary_dilation instead of binary_hit_or_miss
    branch_points = np.sum(
        morphology.binary_dilation(skeleton, branch_struct) & 
        ~morphology.binary_erosion(skeleton, branch_struct)
    )
    branching_score = min(1.0, branch_points / 50)  # Normalize

    # 3. Vessel Distribution
    # Normal lungs have gradually diminishing vessels from hilum to periphery
    h, w = image_gray.shape
    center_y, center_x = h//2, w//2
    
    y_coords, x_coords = np.where(thin_edges)
    if len(y_coords) > 0:
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        dist_hist, _ = np.histogram(distances, bins=10)
        vessel_gradient = np.mean(np.diff(dist_hist))
        vessel_score = 1.0 if vessel_gradient < 0 else 0.5  # Should decrease from center
    else:
        vessel_score = 0.0

    # 4. Texture Uniformity
    glcm = graycomatrix(
        (image_gray * 255).astype(np.uint8),
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Normal tissue should have relatively uniform texture
    texture_score = (homogeneity + correlation) / 2

    # 5. Abnormal Dense Objects
    # Look for suspicious dense objects
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    suspicious_objects = 0
    for region in regions:
        # Check for characteristics of tumors
        if (region.area > 100 and  # Not too small
            region.eccentricity < 0.9 and  # Somewhat round
            region.solidity > 0.8):  # Solid mass
            suspicious_objects += 1
    
    abnormality_score = min(1.0, suspicious_objects / 3)
    clear_lung_score = 1.0 - abnormality_score

    # 6. Symmetry Analysis
    left_half = image_gray[:, :center_x]
    right_half = np.fliplr(image_gray[:, center_x:])
    # Adjust size if halves are unequal
    min_width = min(left_half.shape[1], right_half.shape[1])
    symmetry_score = 1 - np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))

    # Combine all scores with weighted importance
    weights = {
        'bimodal': 0.2,
        'branching': 0.15,
        'vessels': 0.15,
        'texture': 0.15,
        'clear_lung': 0.2,
        'symmetry': 0.15
    }

    normal_confidence = (
        weights['bimodal'] * bimodal_score +
        weights['branching'] * branching_score +
        weights['vessels'] * vessel_score +
        weights['texture'] * texture_score +
        weights['clear_lung'] * clear_lung_score +
        weights['symmetry'] * symmetry_score
    )

    # Decision thresholds
    is_normal = (normal_confidence > 0.7 and 
                suspicious_objects == 0 and 
                symmetry_score > 0.6 and 
                bimodal_score > 0.7)

    return {
        "is_normal": is_normal,
        "normal_confidence": float(normal_confidence),
        "characteristics": {
            "bimodal_score": bimodal_score,
            "vessel_score": vessel_score,
            "texture_score": texture_score,
            "symmetry_score": symmetry_score,
            "suspicious_objects": suspicious_objects
        }
    }

def calculate_density_score(image_gray: np.array) -> float:
    """
    Calculates the density score of potential tumor areas
    Higher values indicate denser tissue (more likely malignant)
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Calculate average intensity in potential tumor regions
    tumor_intensity = np.mean(image_gray[binary_clean]) if np.any(binary_clean) else 0
    
    # Normalize to [0, 1] scale
    return float(min(1.0, tumor_intensity * 2))

def calculate_shape_metrics(image_gray: np.array) -> dict:
    """
    Calculates shape metrics for potential tumor regions
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Label regions
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "sphericity": 0.0,
            "irregularity": 0.0,
            "max_diameter": 0.0,
            "area_ratio": 0.0
        }
    
    # Get largest region (assumed to be the main tumor)
    main_region = max(regions, key=lambda r: r.area)
    
    # Calculate shape metrics
    area = main_region.area
    perimeter = main_region.perimeter
    
    # Sphericity (1 = perfect circle, lower values = more irregular)
    sphericity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Irregularity (inverse of sphericity, normalized)
    irregularity = 1.0 - sphericity
    
    # Maximum diameter
    y0, x0, y1, x1 = main_region.bbox
    max_diameter = max(y1 - y0, x1 - x0) / max(image_gray.shape)
    
    # Area ratio (tumor area to total image area)
    area_ratio = area / (image_gray.shape[0] * image_gray.shape[1])
    
    return {
        "sphericity": float(sphericity),
        "irregularity": float(irregularity),
        "max_diameter": float(max_diameter),
        "area_ratio": float(area_ratio)
    }

def calculate_texture_scores(image_gray: np.array) -> dict:
    """
    Calculates texture scores for potential tumor regions
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Calculate texture features only in the tumor region
    if np.sum(binary_clean) > 100:  # Ensure there's a sufficient region
        # Create mask for tumor region
        y_indices, x_indices = np.where(binary_clean)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Extract tumor region
        tumor_region = image_gray[min_y:max_y+1, min_x:max_x+1]
        
        # Calculate GLCM features
        glcm = graycomatrix(
            (tumor_region * 255).astype(np.uint8),
            distances=[1, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Calculate local entropy as measure of heterogeneity
        local_entropy = shannon_entropy(tumor_region)
        
        # Normalize values to [0, 1]
        heterogeneity = min(1.0, local_entropy / 5.0)
        texture_contrast = min(1.0, contrast / 50.0)
    else:
        contrast = dissimilarity = homogeneity = energy = correlation = 0
        heterogeneity = texture_contrast = 0
    
    return {
        "contrast": float(contrast),
        "dissimilarity": float(dissimilarity),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "correlation": float(correlation),
        "heterogeneity": float(heterogeneity),
        "texture_contrast": float(texture_contrast)
    }

def analyze_growth_pattern(image_gray: np.array) -> dict:
    """
    Analyzes growth pattern of potential tumors (well-defined vs infiltrative)
    """
    # Segment potential tumor regions
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Label regions
    labeled_regions = measure.label(binary_clean)
    regions = measure.regionprops(labeled_regions)
    
    if not regions:
        return {
            "infiltrative_score": 0.0,
            "well_defined_score": 1.0,
            "border_gradient": 0.0
        }
    
    # Get largest region (assumed to be the main tumor)
    main_region = max(regions, key=lambda r: r.area)
    
    # Dilate and subtract to get the boundary region
    y0, x0, y1, x1 = main_region.bbox
    tumor_mask = np.zeros_like(image_gray, dtype=bool)
    tumor_mask[y0:y1, x0:x1] = main_region.image
    
    boundary = morphology.binary_dilation(tumor_mask, morphology.disk(2)) & ~tumor_mask
    
    # Calculate gradient at the boundary
    grad = filters.sobel(image_gray)
    boundary_grad = grad[boundary] if np.any(boundary) else np.array([0])
    
    # Calculate metrics
    # Lower gradient variance indicates well-defined border
    # Higher gradient variance indicates infiltrative growth
    border_gradient = np.std(boundary_grad)
    
    # Normalized scores
    infiltrative_score = min(1.0, border_gradient / 0.2)
    well_defined_score = 1.0 - infiltrative_score
    
    return {
        "infiltrative_score": float(infiltrative_score),
        "well_defined_score": float(well_defined_score),
        "border_gradient": float(border_gradient)
    }

# Update the predict_tumor endpoint
@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # First, validate if it's a proper lung CT scan
        is_valid, validation_message = validate_lung_ct(image_array)
        
        if not is_valid:
            return {
                "status": "invalid_image",
                "message": validation_message,
                "recommendation": "Please provide a valid lung CT scan image."
            }
        
        # Check if it's normal tissue
        # Get normal tissue analysis results
        analysis_result = analyze_normal_tissue(image_array)
        is_normal = analysis_result["is_normal"]
        normal_confidence = analysis_result["normal_confidence"]
        
        if is_normal and normal_confidence > 0.7:
            return {
                "tumor_type": "Normal",
                "predicted_label": "Normal Lung Tissue",
                "confidence": normal_confidence,
                "tumor_likelihood_score": 0.0,
                "accuracy": float(normal_confidence * 100),
                "normal_tissue_probability": normal_confidence,
                "analysis": {
                    "is_normal": True,
                    "normal_characteristics": analysis_result["characteristics"]
                }
            }
        
        # Detailed tumor analysis using our enhanced analytical methods
        analysis_results = analyze_tumor_characteristics(image_array)
        
        # Extract features for the machine learning model
        features = extract_advanced_features(image_array)
        
        # Get most likely tumor type from characteristic analysis
        tumor_type_scores = analysis_results["tumor_type_scores"]
        characteristic_prediction = max(tumor_type_scores.items(), key=lambda x: x[1])
        characteristic_type = characteristic_prediction[0]
        characteristic_confidence = characteristic_prediction[1]
        
        # If model is trained, use it to reinforce the characteristic-based prediction
        model_prediction = None
        model_confidence = 0.0
        
        try:
            # Safely attempt prediction with the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([features])[0]
                model_prediction = model.classes_[np.argmax(probabilities)]
                model_confidence = float(np.max(probabilities))
        except Exception as model_error:
            print(f"Model prediction error: {str(model_error)}")
            # Fall back to characteristic-based prediction
        
        # Adjust confidence if there's significant normal tissue probability
        if normal_confidence > 0.5:
            model_confidence = model_confidence * (1 - normal_confidence)
        
        # Determine final prediction, giving more weight to characteristic analysis
        # since that's based on your specific tumor criteria
        final_type = characteristic_type
        final_confidence = characteristic_confidence
        
        if model_prediction:
            # If model and characteristic analysis agree, boost confidence
            if model_prediction == characteristic_type:
                final_confidence = (characteristic_confidence * 0.7) + (model_confidence * 0.3)
            # If model disagrees but has high confidence, use weighted average
            elif model_confidence > 0.7:
                final_type = model_prediction
                final_confidence = model_confidence
        
        # Generate human-readable descriptions based on tumor type
        descriptions = {
            'adenocarcinoma': "Peripheral nodule with ground-glass opacity and spiculated (spiky) edges, characteristic of Adenocarcinoma.",
            'squamous_cell': "Central tumor near major airways with possible cavitation, typical of Squamous Cell Carcinoma.",
            'large_cell': "Large, poorly defined peripheral mass with irregular and infiltrative appearance, suggestive of Large Cell Carcinoma.",
            'small_cell': "Central mass near hilum with dense appearance, characteristic of Small Cell Lung Cancer.",
            'metastatic': "Multiple round nodules of varying size scattered throughout the lungs, typical of metastatic disease.",
            'hamartoma': "Well-circumscribed nodule with possible calcification and smooth edges, suggestive of benign Hamartoma.",
            'pulmonary_adenoma': "Small, homogeneous nodule with smooth edges, characteristic of benign Pulmonary Adenoma.",
            'normal': "Normal lung tissue pattern with no suspicious masses or opacities."
        }
        
        if is_normal and normal_confidence > 0.8:
            final_type = 'normal'
            final_confidence = normal_confidence
            
        # Create visualization data for the UI
        try:
            # Generate all plots with proper error handling
            from plot import plots
            all_plots = plots['generate_plots'](image_array, {
                'tumor_characteristics': analysis_results,
                'tumor_type_scores': tumor_type_scores,
                'ground_glass': analysis_results.get("ground_glass", {}),
                'cavitation': analysis_results.get("cavitation", {})
            })
        except Exception as plot_error:
            print(f"Plot generation error: {str(plot_error)}")
            all_plots = {}

        return {
            # Basic classification results
            "tumor_type": tumor_types.get(final_type, "Unknown"),
            "predicted_label": tumor_types.get(final_type, "Unknown"),
            "confidence": float(final_confidence),
            "tumor_likelihood_score": float(1.0 - normal_confidence),
            "accuracy": float(final_confidence * 100),
            "normal_tissue_probability": float(normal_confidence),
            
            # Detailed analysis results
            "analysis": {
                "is_normal": bool(is_normal),
                "normal_confidence": float(normal_confidence),
                "tumor_characteristics": {
                    "density_score": float(analysis_results["density_score"]),
                    "shape_metrics": {k: float(v) for k, v in analysis_results["shape_metrics"].items()},
                    "texture_scores": {k: float(v) for k, v in analysis_results["texture_scores"].items()},
                    "growth_pattern": {k: float(v) for k, v in analysis_results["growth_pattern"].items()},
                    "location": {
                        "is_central": bool(analysis_results["location"]["is_central"]),
                        "is_peripheral": bool(analysis_results["location"]["is_peripheral"]),
                        "central_score": float(analysis_results["location"]["central_score"]),
                        "peripheral_score": float(analysis_results["location"]["peripheral_score"])
                    },
                    "nodule_analysis": {
                        "nodule_count": int(analysis_results["nodule_analysis"]["nodule_count"]),
                    "has_multiple_nodules": bool(analysis_results["nodule_analysis"]["has_multiple_nodules"]),
                        "size_variation": float(analysis_results["nodule_analysis"]["size_variation"]),
                        "metastatic_pattern_score": float(analysis_results["nodule_analysis"]["metastatic_pattern_score"])
                    },
                    "cavitation": {
                        "has_cavitation": bool(analysis_results["cavitation"]["has_cavitation"]),
                        "cavitation_score": float(analysis_results["cavitation"]["cavitation_score"])
                    },
                    "ground_glass": {
                        "has_ground_glass_opacity": bool(analysis_results["ground_glass"]["has_ground_glass_opacity"]),
                    "ggo_score": float(analysis_results["ground_glass"]["ggo_score"]),
                        "vessel_preservation": float(analysis_results["ground_glass"]["vessel_preservation"])
                    },
                    "spiculation": {
                        "has_spiculation": bool(analysis_results["spiculation"]["has_spiculation"]),
                        "spiculation_score": float(analysis_results["spiculation"]["spiculation_score"])
                    },
                    "tumor_type_scores": {k: float(v) for k, v in analysis_results["tumor_type_scores"].items()}
                },
                "description": descriptions.get(final_type, "Unknown tumor pattern"),
                "tumor_type_scores": tumor_type_scores,
                "location_analysis": {
                    "is_central": bool(analysis_results["location"]["is_central"]),
                    "is_peripheral": bool(analysis_results["location"]["is_peripheral"]),
                    "multiple_nodules": bool(analysis_results["nodule_analysis"]["has_multiple_nodules"]),
                    "nodule_count": int(analysis_results["nodule_analysis"]["nodule_count"])
                },
                "feature_patterns": {
                    "has_ground_glass_opacity": bool(analysis_results["ground_glass"]["has_ground_glass_opacity"]),
                    "has_spiculation": bool(analysis_results["spiculation"]["has_spiculation"]),
                    "has_cavitation": bool(analysis_results["cavitation"]["has_cavitation"])
                }
            },
            
            # All available plots
            "plots": all_plots,
            
            # Suggested plot display order
            "plot_order": [
                "original_grayscale",        # Original and grayscale image
                "basic_segmentation",        # Basic tumor segmentation
                "segmentation",             # Advanced tumor segmentation
                "feature_analysis",         # Comprehensive feature analysis
                "texture",                  # Texture analysis
                "frequency",                # Frequency domain analysis
                "multiscale",              # Multi-scale feature analysis
                "tumor_type_scores",        # Tumor type probability scores
                "ground_glass_opacity",     # Ground glass opacity analysis
                "cavitation_analysis"       # Cavitation pattern analysis
            ],
            
            # Add metadata about the analysis
            "metadata": {
                "analysis_version": "1.0",
                "processing_timestamp": str(datetime.now()),
                "image_dimensions": {
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis error: {str(e)}"
        )
