"""
CT Head AI — Classical Radiomic Feature Extractor
Extracts domain-specific, physics-grounded radiomic features from CT Head volumes.
100% Classical Computer Vision and Mathematical Statistics (NO Deep Learning).
"""

import numpy as np
from scipy import stats
import cv2

# HU Windows (Standard Radiological Settings)
WINDOWS = {
    "brain":    {"center": 35,  "width": 80},    # Infarcts, edema, parenchyma
    "subdural": {"center": 75,  "width": 200},   # Subdural & epidural hemorrhages
    "bone":     {"center": 600, "width": 2800},  # Calvarium & skull fractures
}

def apply_hu_window(hu_array, center, width):
    """Apply Hounsfield Unit window and normalize to [0, 1]"""
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(hu_array, lo, hi)
    return (clipped - lo) / (hi - lo)

def extract_intensity_features(data_channel, prefix):
    """
    Extract first-order histogram and distribution statistics from a volume channel.
    Ground truth correlation:
      - Elevated 95th/99th percentiles in brain channel = acute hemorrhage (ICH)
      - Suppressed 5th/10th percentiles in brain channel = ischemic infarction
    """
    # Subsample voxels [::8] for ultra-fast, robust statistical distribution computation
    flat = data_channel.reshape(-1)[::8].astype(np.float64)
    
    # Quantiles and percentiles
    p5, p25, p50, p75, p90, p95, p99 = np.percentile(flat, [5, 25, 50, 75, 90, 95, 99])
    mean_val = np.mean(flat)
    std_val = np.std(flat)
    
    # Skewness & Kurtosis (Asymmetry of density distribution)
    # Hyperdense blood creates a strong positive tail (high positive skew)
    skewness = float(stats.skew(flat)) if std_val > 1e-5 else 0.0
    kurt = float(stats.kurtosis(flat)) if std_val > 1e-5 else 0.0

    # Hyperdense ratio (fraction of voxels near top intensity)
    hyperdense_ratio = np.mean(flat > 0.75)
    hypodense_ratio = np.mean(flat < 0.20)

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_p5": p5,
        f"{prefix}_p25": p25,
        f"{prefix}_median": p50,
        f"{prefix}_p75": p75,
        f"{prefix}_p90": p90,
        f"{prefix}_p95": p95,
        f"{prefix}_p99": p99,
        f"{prefix}_skewness": skewness,
        f"{prefix}_kurtosis": kurt,
        f"{prefix}_hyperdense_ratio": hyperdense_ratio,
        f"{prefix}_hypodense_ratio": hypodense_ratio,
    }

def extract_bilateral_symmetry(brain_volume):
    """
    Bilateral Hemispheric Asymmetry Index:
    Computes absolute difference between left and right hemispheres of the brain.
    Clinical rationale:
      - Midline shift and mass effect displace the ventricular system and parenchyma,
        creating strong left-right asymmetry.
    """
    # brain_volume shape: (Slices, H, W)
    mid_x = brain_volume.shape[2] // 2
    left_hemisphere = brain_volume[:, :, :mid_x]
    
    # Mirror right hemisphere to match left coordinates
    right_flipped = np.flip(brain_volume[:, :, mid_x:], axis=2)
    
    # Align shapes in case W is odd
    min_w = min(left_hemisphere.shape[2], right_flipped.shape[2])
    left_hemisphere = left_hemisphere[:, :, :min_w]
    right_flipped = right_flipped[:, :, :min_w]
    
    diff = np.abs(left_hemisphere - right_flipped)
    
    asymmetry_mean = float(np.mean(diff))
    asymmetry_max = float(np.max(diff))
    asymmetry_p95 = float(np.percentile(diff, 95))
    
    return {
        "hemispheric_asymmetry_mean": asymmetry_mean,
        "hemispheric_asymmetry_max": asymmetry_max,
        "hemispheric_asymmetry_p95": asymmetry_p95,
    }

def extract_bone_edge_features(bone_volume):
    """
    Bone Contour and Gradient Analysis:
    Applies Sobel/Canny edge operators on the bone window.
    Clinical rationale:
      - High edge discontinuities or fragmented cortical lines indicate skull fractures.
    """
    # Sample axial slices across the calvarium
    num_slices = bone_volume.shape[0]
    sample_indices = np.linspace(num_slices * 0.2, num_slices * 0.8, 8, dtype=int)
    
    edge_counts = []
    gradient_magnitudes = []
    
    for idx in sample_indices:
        slc = (bone_volume[idx] * 255).astype(np.uint8)
        # Canny edge detection
        edges = cv2.Canny(slc, 100, 200)
        edge_counts.append(np.sum(edges > 0))
        
        # Sobel gradients
        gx = cv2.Sobel(slc, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(slc, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        gradient_magnitudes.append(np.mean(mag))
        
    return {
        "bone_edge_density_mean": float(np.mean(edge_counts)),
        "bone_edge_density_max": float(np.max(edge_counts)),
        "bone_gradient_mag_mean": float(np.mean(gradient_magnitudes)),
    }

def extract_texture_glcm_features(brain_volume):
    """
    Second-Order Texture Features (Gray-Level Co-occurrence Matrix approximation):
    Measures spatial arrangement and homogeneity of tissue.
    Clinical rationale:
      - Edema, ischemic change, and parenchymal contusions cause localized texture coarseness.
    """
    # Analyze the central slice where ventricular/parenchymal anatomy is clearest
    mid_idx = brain_volume.shape[0] // 2
    slc = brain_volume[mid_idx]
    
    # Discretize into 16 bins for fast, robust co-occurrence
    quantized = np.clip((slc * 15).astype(np.int32), 0, 15)
    
    # Horizontal neighboring pairs (x, x+1)
    row_pairs_i = quantized[:, :-1].flatten()
    row_pairs_j = quantized[:, 1:].flatten()
    
    # 16x16 GLCM
    glcm = np.zeros((16, 16), dtype=np.float64)
    np.add.at(glcm, (row_pairs_i, row_pairs_j), 1.0)
    total_pairs = np.sum(glcm)
    if total_pairs > 0:
        glcm /= total_pairs

    # Contrast: sum(P_ij * (i - j)^2)
    i_indices, j_indices = np.indices((16, 16))
    diff_matrix = (i_indices - j_indices) ** 2
    contrast = float(np.sum(glcm * diff_matrix))
    
    # Homogeneity: sum(P_ij / (1 + (i - j)^2))
    homogeneity = float(np.sum(glcm / (1.0 + diff_matrix)))
    
    # Energy / ASM: sum(P_ij^2)
    energy = float(np.sum(glcm ** 2))
    
    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
    }

def extract_volume_radiomics(volume_array):
    """
    Master Radiomic Feature Extractor.
    Input:
      volume_array: Shape (Slices, H, W, 3) where:
        Ch 0 = Brain window
        Ch 1 = Subdural window
        Ch 2 = Bone window
      Values normalized to [0, 1] or uint8 [0, 255].
    Output:
      Dictionary of ~35 interpretable, classical radiomic features.
    """
    vol = volume_array.astype(np.float32)
    if vol.max() > 1.0:
        vol = vol / 255.0

    brain_ch    = vol[:, :, :, 0]
    subdural_ch = vol[:, :, :, 1]
    bone_ch     = vol[:, :, :, 2]

    features = {}

    # 1. First-Order Density Features per Window
    features.update(extract_intensity_features(brain_ch, prefix="brain"))
    features.update(extract_intensity_features(subdural_ch, prefix="subdural"))
    features.update(extract_intensity_features(bone_ch, prefix="bone"))

    # 2. Bilateral Hemispheric Asymmetry (Midline Shift / Mass Effect)
    features.update(extract_bilateral_symmetry(brain_ch))

    # 3. Bone Contour Integrity (Skull Fractures)
    features.update(extract_bone_edge_features(bone_ch))

    # 4. Tissue Texture Homogeneity (Infarctions / Ischemia)
    features.update(extract_texture_glcm_features(brain_ch))

    return features

if __name__ == "__main__":
    print("Testing Classical Radiomic Feature Extractor...")
    fake_vol = np.random.rand(64, 224, 224, 3).astype(np.float32)
    feats = extract_volume_radiomics(fake_vol)
    print(f"[OK] Extracted {len(feats)} classical radiomic features:")
    for k, v in list(feats.items())[:10]:
        print(f"  * {k:30s}: {v:.4f}")
    print("  ... and more.")
