import xarray as xr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.decomposition import PCA
import datetime
import os
from scipy.ndimage import gaussian_filter

class ClimateFeatureDetector:
    """
    SCAFET-inspired feature detection for climate data
    """
    
    def __init__(self):
        self.features_detected = {}
        
    def calculate_shape_index(self, field, scale_km=500):
        """
        Calculate shape index (SI) - core of SCAFET methodology
        SI > 0.5: ridge/filament structures (good for atmospheric rivers)
        SI < -0.5: depression/cyclonic structures
        """
        # Apply Gaussian smoothing to suppress small-scale variability
        sigma = scale_km / 100  # Adjust based on grid resolution
        smoothed = gaussian_filter(field, sigma=sigma)
        
        # Calculate gradients
        gy, gx = np.gradient(smoothed)
        
        # Calculate Hessian matrix elements
        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        
        # Calculate eigenvalues of Hessian
        determinant = gxx * gyy - gxy * gyx
        trace = gxx + gyy
        
        # Shape index calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*determinant))
            lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*determinant))
            
            # Normalize shape index to [-1, 1]
            si = np.where(np.abs(lambda1) > np.abs(lambda2), 
                         (lambda2 / lambda1), 
                         (lambda1 / lambda2))
            
        si = np.nan_to_num(si, nan=0.0)
        return np.clip(si, -1, 1)
    
    def detect_atmospheric_rivers(self, precip_data, wind_u, wind_v, threshold=0.375):
        """
        Detect atmospheric river-like structures using precipitation and wind
        """
        # Calculate moisture flux proxy (precip * wind_speed)
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        moisture_flux = precip_data * wind_speed
        
        # Calculate shape index
        si = self.calculate_shape_index(moisture_flux, scale_km=1000)
        
        # Identify ridge-like structures (SI > threshold)
        ar_candidates = si > threshold
        
        # Filter by size and elongation
        labeled, num_features = ndimage.label(ar_candidates)
        ar_features = []
        
        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            if np.sum(feature_mask) > 50:  # Minimum size
                # Check elongation ratio
                coords = np.where(feature_mask)
                if len(coords[0]) > 0:
                    y_span = np.max(coords[0]) - np.min(coords[0])
                    x_span = np.max(coords[1]) - np.min(coords[1])
                    aspect_ratio = max(y_span, x_span) / (min(y_span, x_span) + 1e-6)
                    
                    if aspect_ratio > 2.0:  # Elongated structure
                        ar_features.append(feature_mask)
        
        return ar_features, si
    
    def detect_cyclones(self, temp_data, wind_u, wind_v, threshold=-0.3):
        """
        Detect cyclonic structures using temperature and wind vorticity
        """
        # Calculate relative vorticity
        gy_u, gx_u = np.gradient(wind_u)
        gy_v, gx_v = np.gradient(wind_v)
        vorticity = gx_v - gy_u
        
        # Calculate shape index for vorticity
        si = self.calculate_shape_index(vorticity, scale_km=300)
        
        # Identify depression-like structures (SI < threshold)
        cyclone_candidates = si < threshold
        
        # Additional filtering based on temperature gradient
        temp_gradient = np.sqrt(np.gradient(temp_data)[0]**2 + np.gradient(temp_data)[1]**2)
        strong_gradient = temp_gradient > np.percentile(temp_gradient, 70)
        
        # Combine conditions
        cyclone_features = cyclone_candidates & strong_gradient
        
        # Filter by size
        labeled, num_features = ndimage.label(cyclone_features)
        filtered_cyclones = []
        
        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            if 20 < np.sum(feature_mask) < 500:  # Size constraints
                filtered_cyclones.append(feature_mask)
        
        return filtered_cyclones, si
    
    def detect_fronts(self, temp_data, threshold_percentile=85):
        """
        Detect temperature fronts using gradient analysis
        """
        # Calculate temperature gradient magnitude
        gy, gx = np.gradient(temp_data)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Calculate shape index for temperature
        si = self.calculate_shape_index(temp_data, scale_km=200)
        
        # Identify strong gradients
        threshold = np.percentile(gradient_magnitude, threshold_percentile)
        strong_gradients = gradient_magnitude > threshold
        
        # Filter for linear structures
        ridge_like = np.abs(si) > 0.2
        front_candidates = strong_gradients & ridge_like
        
        # Clean up small features
        front_candidates = ndimage.binary_opening(front_candidates, structure=np.ones((3,3)))
        
        return front_candidates, gradient_magnitude

def enhanced_climate_processor(date_str):
    """
    Enhanced version of your climate processing with feature detection
    """
    # Load your existing data
    temp_ds = xr.open_dataset("temp.nc")
    precip_ds = xr.open_dataset("precipitation.nc")
    wind_ds = xr.open_dataset("windspeed.nc")
    
    # For this example, we'll assume you have U and V wind components
    # If you only have wind speed, you can estimate components
    date_np = np.datetime64(date_str)
    
    try:
        temp_data = temp_ds["T2M"].sel(time=date_np, method="nearest").values
        precip_data = precip_ds["PRECTOTCORR"].sel(time=date_np, method="nearest").values
        wind_speed = wind_ds["WS10M"].sel(time=date_np, method="nearest").values
        
        # Estimate wind components (this is simplified - ideally you'd have U/V components)
        wind_u = wind_speed * 0.7  # Simplified approximation
        wind_v = wind_speed * 0.7  # Simplified approximation
        
        # Initialize feature detector
        detector = ClimateFeatureDetector()
        
        # Detect features
        ar_features, ar_si = detector.detect_atmospheric_rivers(precip_data, wind_u, wind_v)
        cyclone_features, cyclone_si = detector.detect_cyclones(temp_data, wind_u, wind_v)
        front_features, temp_gradient = detector.detect_fronts(temp_data)
        
        # Create enhanced visualization
        create_feature_visualization(date_str, temp_data, precip_data, wind_speed,
                                   ar_features, cyclone_features, front_features,
                                   ar_si, cyclone_si, temp_gradient)
        
        # Create feature summary
        create_feature_summary(date_str, ar_features, cyclone_features, front_features)
        
        return {
            'atmospheric_rivers': len(ar_features),
            'cyclones': len(cyclone_features),
            'fronts': np.sum(front_features),
            'ar_si': ar_si,
            'cyclone_si': cyclone_si
        }
        
    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        return None

def create_feature_visualization(date_str, temp_data, precip_data, wind_speed,
                               ar_features, cyclone_features, front_features,
                               ar_si, cyclone_si, temp_gradient):
    """
    Create comprehensive visualization with detected features
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Climate Features for {date_str}', fontsize=16)
    
    # Original RGB image (your existing approach)
    R = normalize_data(temp_data)
    G = normalize_data(wind_speed)
    B = normalize_data(precip_data)
    rgb_image = np.stack([R, G, B], axis=-1)
    
    axes[0,0].imshow(rgb_image)
    axes[0,0].set_title('Original RGB (T,W,P)')
    axes[0,0].axis('off')
    
    # Temperature with detected fronts
    im1 = axes[0,1].imshow(temp_data, cmap='RdBu_r')
    axes[0,1].contour(front_features, levels=[0.5], colors='black', linewidths=2)
    axes[0,1].set_title('Temperature + Fronts')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Precipitation with atmospheric rivers
    im2 = axes[0,2].imshow(precip_data, cmap='Blues')
    for i, ar in enumerate(ar_features):
        axes[0,2].contour(ar, levels=[0.5], colors='red', linewidths=2)
    axes[0,2].set_title(f'Precipitation + ARs ({len(ar_features)})')
    axes[0,2].axis('off')
    plt.colorbar(im2, ax=axes[0,2])
    
    # Shape index for AR detection
    im3 = axes[1,0].imshow(ar_si, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1,0].set_title('Shape Index (AR Detection)')
    axes[1,0].axis('off')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Shape index for cyclone detection
    im4 = axes[1,1].imshow(cyclone_si, cmap='RdBu_r', vmin=-1, vmax=1)
    for i, cyclone in enumerate(cyclone_features):
        axes[1,1].contour(cyclone, levels=[0.5], colors='yellow', linewidths=2)
    axes[1,1].set_title(f'Shape Index + Cyclones ({len(cyclone_features)})')
    axes[1,1].axis('off')
    plt.colorbar(im4, ax=axes[1,1])
    
    # Temperature gradient
    im5 = axes[1,2].imshow(temp_gradient, cmap='hot')
    axes[1,2].set_title('Temperature Gradient')
    axes[1,2].axis('off')
    plt.colorbar(im5, ax=axes[1,2])
    
    # Save the enhanced visualization
    os.makedirs('enhanced_features', exist_ok=True)
    plt.savefig(f'enhanced_features/{date_str}_features_enhanced.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_feature_summary(date_str, ar_features, cyclone_features, front_features):
    """
    Generate English language summary of detected features
    """
    summary = f"\n=== Weather Feature Summary for {date_str} ===\n"
    
    # Atmospheric Rivers
    if len(ar_features) > 0:
        summary += f" {len(ar_features)} atmospheric river(s) detected - bringing moisture transport\n"
    else:
        summary += " No atmospheric rivers detected\n"
    
    # Cyclones
    if len(cyclone_features) > 0:
        summary += f"{len(cyclone_features)} cyclonic system(s) detected - low pressure areas\n"
    else:
        summary += "No significant cyclonic systems detected\n"
    
    # Fronts
    front_strength = np.sum(front_features)
    if front_strength > 100:
        summary += f"Strong temperature fronts detected - significant weather boundaries\n"
    elif front_strength > 50:
        summary += f"Moderate temperature fronts detected\n"
    else:
        summary += "Weak or no temperature fronts detected\n"
    
    print(summary)
    
    # Save summary to file
    os.makedirs('feature_summaries', exist_ok=True)
    with open(f'feature_summaries/{date_str}_summary.txt', 'w') as f:
        f.write(summary)

def normalize_data(data):
    """Normalize data to 0-255 range"""
    arr = np.nan_to_num(data, nan=0.0)
    min_val, max_val = np.percentile(arr, 2), np.percentile(arr, 98)
    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)

# Main execution
if __name__ == "__main__":
    # Process a single day
    date_str = "2024-06-17"
    results = enhanced_climate_processor(date_str)
    
    if results:
        print(f"Features detected for {date_str}:")
        print(f"- Atmospheric Rivers: {results['atmospheric_rivers']}")
        print(f"- Cyclones: {results['cyclones']}")
        print(f"- Front Activity: {results['fronts']}")
