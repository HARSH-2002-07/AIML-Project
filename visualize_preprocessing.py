"""
CT Head AI — Preprocessing Pipeline Visualizer & Academic Demonstration
Generates a multi-panel scientific diagnostic figure and terminal demonstration
illustrating the exact mathematical and physical steps of the preprocessing pipeline.

Usage:
    python visualize_preprocessing.py
    python visualize_preprocessing.py --case data/sample_cases/case_hemorrhage.npy
    python visualize_preprocessing.py --case data/sample_cases/case_fracture.npy
    python visualize_preprocessing.py --case data/sample_cases/case_midline_shift.npy
    python visualize_preprocessing.py --case data/sample_cases/case_infarct.npy
    python visualize_preprocessing.py --case data/sample_cases/case_normal.npy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/terminal execution
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.radiomics import extract_volume_radiomics, WINDOWS
from src.sample_generator import generate_ct_case

def run_preprocessing_demonstration(case_path=None, output_fig="reports/preprocessing_pipeline_visualization.png"):
    print("\n" + "=" * 78)
    print("      CT HEAD AI -- CLINICAL RADIOMICS PREPROCESSING PIPELINE")
    print("      [Physical Normalization, Windowing, Symmetry, Edges & Texture]")
    print("=" * 78)

    # 1. Load or Generate Case
    if case_path and os.path.exists(case_path):
        print(f"\n[STAGE 1] Ingesting Volumetric CT Scan: {case_path}")
        vol = np.load(case_path)
    else:
        print("\n[STAGE 1] Generating Volumetric Benchmark CT Scan: [HEMORRHAGE + MASS EFFECT]")
        vol = generate_ct_case(condition="hemorrhage")
        case_path = "benchmark_hemorrhage.npy"

    print(f"  * Raw Volume Shape: {vol.shape}")
    print(f"  * Spatial Matrix:    64 axial slices x 224 x 224 pixels")

    # Normalize to [0.0, 1.0] if uint8
    if vol.max() > 1.0:
        vol = vol.astype(np.float32) / 255.0
    print(f"  * Radiodensity Range: [{vol.min():.2f}, {vol.max():.2f}] (Normalized Hounsfield Units)")

    # 2. Channel Deconstruction
    brain_vol    = vol[:, :, :, 0]
    subdural_vol = vol[:, :, :, 1]
    bone_vol     = vol[:, :, :, 2]

    # Select representative mid-axial slice for visualization
    mid_idx = vol.shape[0] // 2
    slc_brain    = brain_vol[mid_idx]
    slc_subdural = subdural_vol[mid_idx]
    slc_bone     = bone_vol[mid_idx]

    print("\n[STAGE 2] Multi-Window Attenuation Mapping (Clinical Hounsfield Windows):")
    print("  * Channel 0: Brain Window    (Center:  35 HU, Width:   80 HU) -> Parenchyma, Grey/White matter")
    print("  * Channel 1: Subdural Window (Center:  75 HU, Width:  200 HU) -> Acute Extravasated Blood Pooling")
    print("  * Channel 2: Bone Window     (Center: 600 HU, Width: 2800 HU) -> Cortical Calvarium & Fractures")

    # 3. Bilateral Hemispheric Asymmetry Preprocessing
    print("\n[STAGE 3] Bilateral Hemispheric Symmetry Preprocessing:")
    mid_x = slc_brain.shape[1] // 2
    left_hemi = slc_brain[:, :mid_x]
    right_hemi_flipped = np.flip(slc_brain[:, mid_x:], axis=1)
    min_w = min(left_hemi.shape[1], right_hemi_flipped.shape[1])
    diff_hemi = np.abs(left_hemi[:, :min_w] - right_hemi_flipped[:, :min_w])

    asym_mean = float(np.mean(diff_hemi))
    asym_max  = float(np.max(diff_hemi))
    asym_p95  = float(np.percentile(diff_hemi, 95))
    print(f"  * Sagittal Midline Alignment: mid_x = {mid_x} px")
    print(f"  * Left vs Right Reflection Difference: Mean={asym_mean:.4f}, Max={asym_max:.4f}, P95={asym_p95:.4f}")
    print(f"  * Clinical Interpretation: {'Marked parenchymal shift detected' if asym_p95 > 0.20 else 'Bilateral symmetry preserved'}")

    # 4. Bone Spatial Gradient & Canny Edge Preprocessing
    print("\n[STAGE 4] Cortical Bone Morphological & Gradient Preprocessing:")
    slc_bone_uint8 = (slc_bone * 255).astype(np.uint8)
    canny_edges = cv2.Canny(slc_bone_uint8, 100, 200)
    gx = cv2.Sobel(slc_bone_uint8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(slc_bone_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    edge_count = np.sum(canny_edges > 0)
    mean_grad  = np.mean(grad_mag)
    print(f"  * High-Pass Spatial Operator: Sobel 3x3 derivative kernel (Mean Gradient Mag: {mean_grad:.2f})")
    print(f"  * Dual-Threshold Hysteresis: Canny [100, 200] (Edge Pixel Count: {edge_count})")
    print(f"  * Clinical Interpretation: {'Cortical step-off / fracture discontinuity' if edge_count > 1215 else 'Intact continuous calvarium'}")

    # 5. Second-Order GLCM Texture Preprocessing
    print("\n[STAGE 5] Second-Order GLCM Tissue Texture Preprocessing:")
    quantized = np.clip((slc_brain * 15).astype(np.int32), 0, 15)
    row_pairs_i = quantized[:, :-1].flatten()
    row_pairs_j = quantized[:, 1:].flatten()
    glcm = np.zeros((16, 16), dtype=np.float64)
    np.add.at(glcm, (row_pairs_i, row_pairs_j), 1.0)
    total_pairs = np.sum(glcm)
    if total_pairs > 0:
        glcm /= total_pairs

    i_indices, j_indices = np.indices((16, 16))
    diff_matrix = (i_indices - j_indices) ** 2
    contrast = float(np.sum(glcm * diff_matrix))
    homogeneity = float(np.sum(glcm / (1.0 + diff_matrix)))
    energy = float(np.sum(glcm ** 2))
    print(f"  * Gray-Level Quantization: 16 discrete attenuation bins")
    print(f"  * Spatial Co-occurrence Matrix: 16x16 joint transition probabilities")
    print(f"  * Computed Texture Invariants: Contrast={contrast:.3f}, Homogeneity={homogeneity:.3f}, Energy={energy:.3f}")

    # 6. Extract Full 48-Dimensional Quantitative Radiomic Feature Vector
    print("\n[STAGE 6] Compiling 48 Handcrafted Quantitative Biomarkers:")
    feats = extract_volume_radiomics(vol)
    print(f"  * Total Handcrafted Physical Features Extracted: {len(feats)}")

    # 7. Generate Multi-Panel Scientific Figure
    print(f"\n[STAGE 7] Rendering Scientific Diagnostic Visualization...")
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)

    fig = plt.figure(figsize=(16, 11), facecolor="#0f172a")
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

    # Custom styling
    title_kwargs = {"fontsize": 11, "fontweight": "bold", "color": "#f8fafc", "pad": 8}
    label_kwargs = {"fontsize": 9, "color": "#94a3b8"}

    # Top Row: The 3 Radiological Windows + Composite RGB
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(slc_brain, cmap="bone")
    ax1.set_title("1. Brain Window (35/80 HU)", **title_kwargs)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(slc_subdural, cmap="magma")
    ax2.set_title("2. Subdural Window (75/200 HU)", **title_kwargs)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(slc_bone, cmap="gray")
    ax3.set_title("3. Bone Window (600/2800 HU)", **title_kwargs)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    ax4 = fig.add_subplot(gs[0, 3])
    rgb_composite = np.stack([slc_subdural, slc_brain, slc_bone], axis=-1)
    ax4.imshow(rgb_composite)
    ax4.set_title("4. Multi-Window RGB Fusion", **title_kwargs)
    ax4.axis("off")

    # Middle Row: Symmetry & Bone Edge Operators
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(left_hemi, cmap="bone")
    ax5.set_title("Left Hemisphere", **title_kwargs)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(right_hemi_flipped, cmap="bone")
    ax6.set_title("Right Hemisphere (Flipped)", **title_kwargs)
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(diff_hemi, cmap="turbo")
    ax7.set_title(f"Asymmetry Diff (|L - R_flip|)\nMean: {asym_mean:.3f} | P95: {asym_p95:.3f}", **title_kwargs)
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(canny_edges, cmap="copper")
    ax8.set_title(f"Canny Cortical Edges\nCount: {edge_count} | Mag: {mean_grad:.1f}", **title_kwargs)
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    # Bottom Row: Density Histogram, GLCM Matrix, Feature Table
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.set_facecolor("#1e293b")
    flat_brain = slc_brain.flatten()
    flat_sub    = slc_subdural.flatten()
    ax9.hist(flat_brain[flat_brain > 0.05], bins=40, alpha=0.6, color="#38bdf8", label="Brain Window (>0.05)")
    ax9.hist(flat_sub[flat_sub > 0.05], bins=40, alpha=0.6, color="#f43f5e", label="Subdural Window (>0.05)")
    p95_val = np.percentile(flat_brain, 95)
    ax9.axvline(p95_val, color="#facc15", linestyle="--", linewidth=2, label=f"Brain P95 ({p95_val:.2f})")
    ax9.set_title("Voxel Density Distribution & Quantiles", **title_kwargs)
    ax9.tick_params(colors="#94a3b8")
    ax9.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    for spine in ax9.spines.values():
        spine.set_color("#334155")

    ax10 = fig.add_subplot(gs[2, 2])
    im10 = ax10.imshow(np.log1p(glcm), cmap="plasma")
    ax10.set_title(f"16x16 GLCM Texture\nCont:{contrast:.2f} Hom:{homogeneity:.2f}", **title_kwargs)
    ax10.tick_params(colors="#94a3b8")
    ax10.set_xlabel("Voxel i", **label_kwargs)
    ax10.set_ylabel("Voxel j", **label_kwargs)
    for spine in ax10.spines.values():
        spine.set_color("#334155")

    ax11 = fig.add_subplot(gs[2, 3])
    ax11.set_facecolor("#1e293b")
    ax11.axis("off")
    summary_text = (
        "PREPROCESSING SUMMARY\n"
        "-------------------------------\n"
        f"Input Slice:    Mid-Axial (z={mid_idx})\n"
        f"Brain Mean:     {feats['brain_mean']:.3f}\n"
        f"Brain P95/P99:  {feats['brain_p95']:.3f} / {feats['brain_p99']:.3f}\n"
        f"Hyperdense:     {feats['brain_hyperdense_ratio']*100:.2f}%\n"
        f"Hypodense:      {feats['brain_hypodense_ratio']*100:.2f}%\n"
        f"Asymmetry Mean: {feats['hemispheric_asymmetry_mean']:.4f}\n"
        f"Bone Edge Mean: {feats['bone_edge_density_mean']:.1f}\n"
        f"GLCM Contrast:  {feats['glcm_contrast']:.3f}\n"
        "-------------------------------\n"
        "STATUS: 48 Physical Features\n"
        "Ready for ML Classifier Ingestion"
    )
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
              fontsize=9, family="monospace", color="#38bdf8",
              verticalalignment="top")

    fig.suptitle("CT HEAD RADIOMICS: MULTI-STAGE PREPROCESSING & FEATURE EXTRACTION PIPELINE",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)

    plt.savefig(output_fig, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n[OK] High-resolution visualization saved to: {output_fig}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Head Preprocessing Visualizer")
    parser.add_argument("--case", type=str, default="data/sample_cases/case_hemorrhage.npy",
                        help="Path to .npy CT volume")
    parser.add_argument("--output", type=str, default="reports/preprocessing_pipeline_visualization.png",
                        help="Path to output PNG image")
    args = parser.parse_args()

    run_preprocessing_demonstration(case_path=args.case, output_fig=args.output)
