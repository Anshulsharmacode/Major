import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage import color, filters, measure, morphology, feature, exposure, img_as_ubyte
import io
import base64
from typing import Dict, Any, Optional
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore

def encode_plot_to_base64() -> str:
    """Helper function to encode matplotlib plot to base64"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def set_style():
    """Set consistent style for all plots"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#666666',
        'grid.color': '#666666',
        'text.color': '#ffffff',
        'axes.labelcolor': '#ffffff',
        'xtick.color': '#ffffff',
        'ytick.color': '#ffffff',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })

def plot_original_and_grayscale(image_array: np.ndarray) -> str:
    """Plot original image and its grayscale version with enhanced styling"""
    set_style()
    fig = plt.figure(figsize=(15, 7))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Original image
    ax1 = plt.subplot(121)
    ax1.imshow(image_array)
    ax1.set_title('Original Image', pad=20, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Add a subtle border
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(2)
    
    # Grayscale image with custom colormap
    ax2 = plt.subplot(122)
    image_gray = color.rgb2gray(image_array)
    custom_cmap = plt.cm.get_cmap('gray')
    im = ax2.imshow(image_gray, cmap=custom_cmap)
    ax2.set_title('Enhanced Grayscale', pad=20, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add a subtle border
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout(pad=3.0)
    return encode_plot_to_base64()

def plot_tumor_segmentation(image_array: np.ndarray) -> str:
    """Plot tumor segmentation with enhanced overlays and styling"""
    set_style()
    image_gray = color.rgb2gray(image_array)
    
    # Multi-threshold segmentation
    thresh_otsu = filters.threshold_otsu(image_gray)
    thresh_local = filters.threshold_local(image_gray, block_size=35)
    
    binary_otsu = image_gray > thresh_otsu
    binary_local = image_gray > thresh_local
    binary_combined = binary_otsu & binary_local
    binary_clean = morphology.binary_opening(binary_combined)
    binary_clean = morphology.remove_small_objects(binary_clean, min_size=100)
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Tumor Segmentation Analysis', fontsize=16, fontweight='bold', y=1.05)
    
    # Original with tumor overlay
    ax1 = plt.subplot(131)
    ax1.imshow(image_gray, cmap='gray')
    overlay = ax1.imshow(binary_clean, cmap='magma', alpha=0.4)
    ax1.set_title('Tumor Overlay', pad=20, fontsize=12)
    ax1.axis('off')
    plt.colorbar(overlay, ax=ax1, fraction=0.046, pad=0.04, label='Tumor Probability')
    
    # Binary segmentation with custom colormap
    ax2 = plt.subplot(132)
    seg_map = ax2.imshow(binary_clean, cmap='RdYlBu_r')
    ax2.set_title('Segmentation Map', pad=20, fontsize=12)
    ax2.axis('off')
    plt.colorbar(seg_map, ax=ax2, fraction=0.046, pad=0.04, label='Binary Mask')
    
    # Contour overlay with enhanced styling
    ax3 = plt.subplot(133)
    ax3.imshow(image_gray, cmap='gray')
    contours = measure.find_contours(binary_clean)
    
    # Plot contours with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(contours)))
    for contour, color in zip(contours, colors):
        ax3.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)
        
    ax3.set_title('Boundary Detection', pad=20, fontsize=12)
    ax3.axis('off')
    
    # Add subtle borders
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(2)
    
    plt.tight_layout(pad=3.0)
    return encode_plot_to_base64()

def plot_feature_analysis(image_array: np.ndarray, tumor_characteristics: Dict[str, Any]) -> str:
    """Plot feature analyses with enhanced visualization and interactive elements"""
    set_style()
    # Convert to grayscale and uint8 properly
    image_gray = color.rgb2gray(image_array)
    image_uint8 = (image_gray * 255).astype(np.uint8)  # Convert to uint8 properly
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Advanced Feature Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    # Edge detection with custom colormap
    ax1 = plt.subplot(231)
    edges = filters.sobel(image_gray)
    edge_map = ax1.imshow(edges, cmap='magma')
    ax1.set_title('Edge Detection', pad=15, fontsize=12)
    ax1.axis('off')
    plt.colorbar(edge_map, ax=ax1, fraction=0.046, pad=0.04)
    
    # LBP with enhanced color mapping
    ax2 = plt.subplot(232)
    try:
        lbp = feature.local_binary_pattern(image_uint8, P=8, R=3, method='uniform')
    except Exception as e:
        print(f"LBP calculation error: {str(e)}")
        lbp = np.zeros_like(image_uint8)
    lbp_map = ax2.imshow(lbp, cmap='nipy_spectral')
    ax2.set_title('Texture Analysis (LBP)', pad=15, fontsize=12)
    ax2.axis('off')
    plt.colorbar(lbp_map, ax=ax2, fraction=0.046, pad=0.04)
    
    # Intensity histogram with style
    ax3 = plt.subplot(233)
    hist_values, bins, _ = ax3.hist(image_gray.ravel(), bins=50, 
                                   color='#00ff99', alpha=0.7, 
                                   edgecolor='white')
    ax3.set_title('Intensity Distribution', pad=15, fontsize=12)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # GLCM features with gradient colors
    ax4 = plt.subplot(234)
    texture_scores = tumor_characteristics.get('texture_scores', {})
    if texture_scores:
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        values = [texture_scores.get(f, 0) for f in features]
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = ax4.bar(features, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax4.set_title('Texture Features', pad=15, fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # Shape metrics with custom styling
    ax5 = plt.subplot(235)
    shape_metrics = tumor_characteristics.get('shape_metrics', {})
    if shape_metrics:
        metrics = list(shape_metrics.keys())
        values = list(shape_metrics.values())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(metrics)))
        bars = ax5.bar(metrics, values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax5.set_title('Shape Characteristics', pad=15, fontsize=12)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
    
    # Growth pattern with gradient colors
    ax6 = plt.subplot(236)
    growth_pattern = tumor_characteristics.get('growth_pattern', {})
    if growth_pattern:
        patterns = list(growth_pattern.keys())
        values = list(growth_pattern.values())
        colors = plt.cm.plasma(np.linspace(0, 1, len(patterns)))
        bars = ax6.bar(patterns, values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax6.set_title('Growth Pattern Analysis', pad=15, fontsize=12)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
    
    # Add subtle borders to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(2)
    
    plt.tight_layout(pad=3.0)
    return encode_plot_to_base64()

def plot_comprehensive_analysis(image_array: np.ndarray, analysis_results: Dict[str, Any]) -> Dict[str, str]:
    """Generate comprehensive analysis plots with enhanced visualization"""
    set_style()
    # Convert to grayscale and uint8 properly
    image_gray = color.rgb2gray(image_array)
    image_uint8 = (image_gray * 255).astype(np.uint8)  # Convert to uint8 properly
    plots = {}
    
    # Texture Analysis with enhanced styling
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Advanced Texture Analysis', fontsize=16, fontweight='bold', y=1.05)
    
    # LBP with custom colormap
    ax1 = plt.subplot(131)
    try:
        lbp = feature.local_binary_pattern(image_uint8, P=8, R=3, method='uniform')
    except Exception as e:
        print(f"LBP calculation error: {str(e)}")
        lbp = np.zeros_like(image_uint8)
    lbp_map = ax1.imshow(lbp, cmap='nipy_spectral')
    ax1.set_title('Local Binary Pattern', pad=15, fontsize=12)
    ax1.axis('off')
    plt.colorbar(lbp_map, ax=ax1, fraction=0.046, pad=0.04)
    
    # Edge detection with enhanced visualization
    ax2 = plt.subplot(132)
    edges = filters.sobel(image_gray)
    edge_map = ax2.imshow(edges, cmap='magma')
    ax2.set_title('Edge Enhancement', pad=15, fontsize=12)
    ax2.axis('off')
    plt.colorbar(edge_map, ax=ax2, fraction=0.046, pad=0.04)
    
    # Gradient magnitude with custom colormap
    ax3 = plt.subplot(133)
    gradient = filters.sobel(image_gray)
    grad_map = ax3.imshow(gradient, cmap='viridis')
    ax3.set_title('Gradient Analysis', pad=15, fontsize=12)
    ax3.axis('off')
    plt.colorbar(grad_map, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout(pad=3.0)
    plots['texture'] = encode_plot_to_base64()
    
    # Frequency Analysis with enhanced visualization
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold', y=1.05)
    
    # FFT with enhanced colormap
    f_transform = np.fft.fft2(image_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    ax1 = plt.subplot(121)
    fft_map = ax1.imshow(magnitude_spectrum, cmap='twilight')
    ax1.set_title('Frequency Spectrum (FFT)', pad=15, fontsize=12)
    ax1.axis('off')
    plt.colorbar(fft_map, ax=ax1, fraction=0.046, pad=0.04)
    
    # Enhanced histogram
    ax2 = plt.subplot(122)
    hist_values, bins, _ = ax2.hist(image_gray.ravel(), bins=50, 
                                   color='#00ff99', alpha=0.7,
                                   edgecolor='white')
    ax2.set_title('Intensity Distribution', pad=15, fontsize=12)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plots['frequency'] = encode_plot_to_base64()
    
    # Multi-scale Analysis with enhanced visualization
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Multi-scale Feature Analysis', fontsize=16, fontweight='bold', y=1.05)
    
    for i, sigma in enumerate([1, 2, 4]):
        ax = plt.subplot(131 + i)
        smoothed = filters.gaussian(image_gray, sigma=sigma)
        smooth_map = ax.imshow(smoothed, cmap='plasma')
        ax.set_title(f'Scale σ={sigma}', pad=15, fontsize=12)
        ax.axis('off')
        plt.colorbar(smooth_map, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout(pad=3.0)
    plots['multiscale'] = encode_plot_to_base64()
    
    # Tumor Type Scores with enhanced styling
    if 'tumor_type_scores' in analysis_results:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Tumor Classification Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        scores = analysis_results['tumor_type_scores']
        types = list(scores.keys())
        values = list(scores.values())
        colors = plt.cm.RdYlBu(np.linspace(0, 1, len(types)))
        
        ax = plt.gca()
        bars = ax.barh(types, values, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2%}',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Confidence Score')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plots['tumor_type_scores'] = encode_plot_to_base64()
    
    return plots

def plot_pattern_analysis(image_array: np.ndarray, analysis_results: Dict[str, Any]) -> Dict[str, str]:
    """Generate pattern analysis plots with enhanced visualization"""
    set_style()
    image_gray = color.rgb2gray(image_array)
    plots = {}
    
    # Ground Glass Opacity Analysis with enhanced styling
    if 'ground_glass' in analysis_results:
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle('Ground Glass Opacity Analysis', fontsize=16, fontweight='bold', y=1.05)
        
        # Original with enhanced contrast
        ax1 = plt.subplot(131)
        ax1.imshow(image_gray, cmap='gray')
        ax1.set_title('Original Image', pad=15, fontsize=12)
        ax1.axis('off')
        
        # Texture variance map with custom colormap
        ax2 = plt.subplot(132)
        local_var = filters.rank.variance(image_gray, morphology.disk(3))
        var_map = ax2.imshow(local_var, cmap='inferno')
        ax2.set_title('Texture Variance Map', pad=15, fontsize=12)
        ax2.axis('off')
        plt.colorbar(var_map, ax=ax2, fraction=0.046, pad=0.04)
        
        # GGO score visualization
        ax3 = plt.subplot(133)
        ggo_score = analysis_results['ground_glass']['ggo_score']
        bar = ax3.bar(['GGO Score'], [ggo_score], color='#00ff99')
        
        # Add value label
        height = bar[0].get_height()
        ax3.text(bar[0].get_x() + bar[0].get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=12)
        
        ax3.set_ylim(0, 1)
        ax3.set_title('Ground Glass Opacity Score', pad=15, fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plots['ground_glass_opacity'] = encode_plot_to_base64()
    
    # Cavitation Analysis with enhanced visualization
    if 'cavitation' in analysis_results:
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle('Cavitation Pattern Analysis', fontsize=16, fontweight='bold', y=1.05)
        
        binary = image_gray > filters.threshold_otsu(image_gray)
        binary_clean = morphology.binary_opening(binary)
        holes = morphology.remove_small_holes(binary_clean) & ~binary_clean
        
        # Original with cavitation overlay
        ax1 = plt.subplot(131)
        ax1.imshow(image_gray, cmap='gray')
        overlay = ax1.imshow(holes, cmap='autumn', alpha=0.5)
        ax1.set_title('Cavitation Detection', pad=15, fontsize=12)
        ax1.axis('off')
        plt.colorbar(overlay, ax=ax1, fraction=0.046, pad=0.04)
        
        # Binary holes with custom colormap
        ax2 = plt.subplot(132)
        hole_map = ax2.imshow(holes, cmap='RdYlBu_r')
        ax2.set_title('Cavity Regions', pad=15, fontsize=12)
        ax2.axis('off')
        plt.colorbar(hole_map, ax=ax2, fraction=0.046, pad=0.04)
        
        # Cavitation score with styled bar
        ax3 = plt.subplot(133)
        cav_score = analysis_results['cavitation']['cavitation_score']
        bar = ax3.bar(['Cavitation Score'], [cav_score], color='#ff3366')
        
        # Add value label
        height = bar[0].get_height()
        ax3.text(bar[0].get_x() + bar[0].get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=12)
        
        ax3.set_ylim(0, 1)
        ax3.set_title('Cavitation Intensity', pad=15, fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plots['cavitation_analysis'] = encode_plot_to_base64()
        
    return plots

def plot_basic_segmentation(image_array: np.ndarray) -> str:
    """Plot basic segmentation visualization"""
    set_style()
    
    plt.figure(figsize=(10, 10))
    # Create a visualization of the segmentation
    image_gray = color.rgb2gray(image_array)
    thresh = filters.threshold_otsu(image_gray)
    binary = image_gray > thresh
    binary_clean = morphology.binary_opening(binary)
    
    # Original image
    ax1 = plt.subplot(121)
    ax1.imshow(image_gray, cmap='gray')
    ax1.set_title('Original CT Scan', pad=20, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Add subtle border
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(2)
    
    # Segmentation overlay
    ax2 = plt.subplot(122)
    ax2.imshow(image_gray, cmap='gray')
    overlay = ax2.imshow(binary_clean, cmap='hot', alpha=0.3)
    ax2.set_title('Region Analysis', pad=20, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add subtle border
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(2)
    
    plt.tight_layout(pad=3.0)
    return encode_plot_to_base64()

def generate_plots(image_array: np.array, analysis_results: Dict[str, Any]) -> Dict[str, str]:
    """Main function to generate all plots"""
    plots = {}
    
    # Basic visualizations
    plots['original_grayscale'] = plot_original_and_grayscale(image_array)
    plots['basic_segmentation'] = plot_basic_segmentation(image_array)
    plots['segmentation'] = plot_tumor_segmentation(image_array)
    
    # Feature analysis
    if 'tumor_characteristics' in analysis_results:
        plots['feature_analysis'] = plot_feature_analysis(
            image_array, 
            analysis_results['tumor_characteristics']
        )
    
    # Comprehensive analysis
    comprehensive_plots = plot_comprehensive_analysis(image_array, analysis_results)
    plots.update(comprehensive_plots)
    
    # Pattern analysis
    pattern_plots = plot_pattern_analysis(image_array, analysis_results)
    plots.update(pattern_plots)
    
    return plots

# Export the main plotting function
plots = {
    'encode_plot_to_base64': encode_plot_to_base64,
    'plot_original_and_grayscale': plot_original_and_grayscale,
    'plot_tumor_segmentation': plot_tumor_segmentation,
    'plot_feature_analysis': plot_feature_analysis,
    'plot_comprehensive_analysis': plot_comprehensive_analysis,
    'plot_pattern_analysis': plot_pattern_analysis,
    'plot_basic_segmentation': plot_basic_segmentation,
    'generate_plots': generate_plots
}
