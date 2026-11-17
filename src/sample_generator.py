"""
CT Head AI — Synthetic & Sample CT Case Generator
Generates realistic CT Head volumes with radiological pathology signatures.
Used for testing, benchmarking, and live demonstrations without requiring 500GB DICOM files.
"""

import os
import numpy as np
import cv2

def create_base_head_slice(img_size=224):
    """Generates an anatomically structured 3-channel axial CT slice base."""
    # Work on independent 2D contiguous channels (uint8)
    ch_brain    = np.zeros((img_size, img_size), dtype=np.uint8)
    ch_subdural = np.zeros((img_size, img_size), dtype=np.uint8)
    ch_bone     = np.zeros((img_size, img_size), dtype=np.uint8)

    center = (img_size // 2, img_size // 2)
    
    # 1. Skull Ring (Bone window = high intensity in Ch_bone, moderate in Ch_subdural)
    cv2.ellipse(ch_bone, center, (85, 100), 0, 0, 360, 245, thickness=6)
    cv2.ellipse(ch_subdural, center, (85, 100), 0, 0, 360, 180, thickness=6)

    # 2. Brain Parenchyma (Brain window ~115, Subdural ~55)
    cv2.ellipse(ch_brain, center, (78, 93), 0, 0, 360, 115, thickness=-1)
    cv2.ellipse(ch_subdural, center, (78, 93), 0, 0, 360, 55, thickness=-1)

    # 3. Ventricles (CSF is hypodense: ~30 in brain window)
    cv2.ellipse(ch_brain, (center[0] - 12, center[1] - 5), (6, 22), -10, 0, 360, 30, thickness=-1)
    cv2.ellipse(ch_brain, (center[0] + 12, center[1] - 5), (6, 22), 10, 0, 360, 30, thickness=-1)

    # Add soft anatomical texture noise
    noise = np.random.normal(0, 5, (img_size, img_size))
    ch_brain = np.clip(ch_brain.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return ch_brain, ch_subdural, ch_bone

def generate_ct_case(condition="normal", num_slices=64, img_size=224):
    """
    Generates a full 3D CT study (64, 224, 224, 3) in uint8.
    Conditions supported: 'normal', 'hemorrhage', 'infarct', 'fracture', 'midline_shift'
    """
    vol = np.zeros((num_slices, img_size, img_size, 3), dtype=np.uint8)
    
    for s in range(num_slices):
        z_ratio = np.sin((s / num_slices) * np.pi)
        if z_ratio > 0.15:
            b, sub, bone = create_base_head_slice(img_size)
            vol[s, :, :, 0] = b
            vol[s, :, :, 1] = sub
            vol[s, :, :, 2] = bone

    center = (img_size // 2, img_size // 2)

    if condition == "hemorrhage":
        # Acute blood: High attenuation intraparenchymal + subdural pool
        for s in range(20, 50):
            b_slc   = np.ascontiguousarray(vol[s, :, :, 0])
            sub_slc = np.ascontiguousarray(vol[s, :, :, 1])
            # Focal parenchymal hematoma + subdural crescent
            cv2.circle(b_slc, (center[0] + 24, center[1] - 10), 18, 240, thickness=-1)
            cv2.ellipse(sub_slc, (center[0] + 45, center[1]), (14, 38), 20, 0, 360, 235, thickness=-1)
            vol[s, :, :, 0] = b_slc
            vol[s, :, :, 1] = sub_slc

    elif condition == "infarct":
        # Ischemic stroke / edema: Large hypodense MCA territory wedge
        for s in range(18, 48):
            b_slc = np.ascontiguousarray(vol[s, :, :, 0])
            cv2.ellipse(b_slc, (center[0] - 38, center[1] + 8), (35, 24), 35, 0, 360, 20, thickness=-1)
            vol[s, :, :, 0] = b_slc

    elif condition == "fracture":
        # Depressed comminuted skull fracture with cortical discontinuity and bone fragments
        for s in range(18, 44):
            bone_slc = np.ascontiguousarray(vol[s, :, :, 2])
            # Cortical break gap
            cv2.line(bone_slc, (center[0] + 65, center[1] - 65), (center[0] + 95, center[1] - 35), 0, thickness=14)
            # Displaced bone fragments
            for offset in [-12, -6, 0, 6, 12]:
                cv2.circle(bone_slc, (center[0] + 75 + offset, center[1] - 45 - offset // 2), 3, 255, thickness=-1)
            vol[s, :, :, 2] = bone_slc

    elif condition == "midline_shift":
        # Large hemispheric mass effect displacing parenchyma and ventricles across midline
        for s in range(18, 50):
            b_slc, _, _ = create_base_head_slice(img_size)
            # Re-draw shifted, compressed ventricles pushed 22 pixels right
            cv2.ellipse(b_slc, (center[0] + 16, center[1] - 5), (4, 16), 18, 0, 360, 25, thickness=-1)
            # Large mass on the left hemisphere pushing midline
            cv2.circle(b_slc, (center[0] - 42, center[1] - 10), 32, 210, thickness=-1)
            vol[s, :, :, 0] = b_slc

    return vol

def save_sample_cases(output_dir="data/sample_cases"):
    """Creates a suite of benchmark cases for testing and live evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    conditions = ["normal", "hemorrhage", "infarct", "fracture", "midline_shift"]
    
    saved_paths = {}
    for cond in conditions:
        vol = generate_ct_case(condition=cond)
        path = os.path.join(output_dir, f"case_{cond}.npy")
        np.save(path, vol)
        saved_paths[cond] = path
        print(f"  * Generated: {path} (Shape: {vol.shape})")
        
    return saved_paths

if __name__ == "__main__":
    print("Generating sample CT cases for classical ML...")
    save_sample_cases()
    print("[OK] Sample cases ready.")
