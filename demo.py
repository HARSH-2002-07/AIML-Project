"""
CT Head AI — Interactive Classical Machine Learning Demo
Runs end-to-end non-deep learning inference on CT head volumes.
Usage:
    python demo.py
    python demo.py --case data/sample_cases/case_hemorrhage.npy
    python demo.py --condition fracture
    python demo.py --condition infarct
    python demo.py --condition normal
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.radiomics import extract_volume_radiomics
from src.sample_generator import generate_ct_case, save_sample_cases
from src.train_classical import train_and_benchmark
from src.evaluate import BIOMARKER_EXPLANATIONS

def run_demo(case_path=None, condition="hemorrhage"):
    print("\n" + "=" * 76)
    print("      CT HEAD AI -- CLASSICAL RADIOMICS CLINICAL TRIAGE DEMO")
    print("      [100% Traditional Machine Learning -- Random Forest & Scikit-Learn]")
    print("=" * 76)

    # 1. Load or Generate CT Volume
    if case_path and os.path.exists(case_path):
        print(f"\n[1/4] Loading Patient CT Study: {case_path}")
        vol = np.load(case_path)
    else:
        print(f"\n[1/4] Generating Benchmark Clinical Case with signature: [{condition.upper()}]")
        vol = generate_ct_case(condition=condition)
        case_path = f"virtual_case_{condition}.npy"

    print(f"      * Volume Shape: {vol.shape} (64 axial slices x 224x224 px x 3 HU Windows)")
    print(f"      * Channels: Ch0=Brain (35/80 HU) | Ch1=Subdural (75/200 HU) | Ch2=Bone (600/2800 HU)")

    # 2. Extract Classical Handcrafted Radiomic Features
    print("\n[2/4] Extracting Domain-Specific Mathematical & Physical Radiomics...")
    feats = extract_volume_radiomics(vol)
    print(f"      * Successfully computed {len(feats)} mathematical biomarkers:")
    print(f"        - Brain Window 95th Percentile HU:    {feats['brain_p95']:.3f}")
    print(f"        - Hyperdense Blood Volume Ratio:       {feats['brain_hyperdense_ratio']*100:.2f}%")
    print(f"        - Bilateral Hemispheric Asymmetry:     {feats['hemispheric_asymmetry_mean']:.4f}")
    print(f"        - Skull Cortical Edge Density (Canny): {feats['bone_edge_density_mean']:.1f}")
    print(f"        - GLCM Tissue Texture Contrast:        {feats['glcm_contrast']:.3f}")

    # 3. Load Classical Model Pipeline
    model_path = BASE_DIR / "models" / "classical_pipeline.joblib"
    if not model_path.exists():
        print("\n[!] Trained classical model not found. Initializing quick benchmark training...")
        train_and_benchmark()

    pipeline = joblib.load(model_path)
    clf          = pipeline["classifier"]
    scaler       = pipeline["scaler"]
    feature_cols = pipeline["feature_cols"]
    target_cols  = pipeline["target_cols"]
    model_name   = pipeline["model_name"]

    # Align feature vector
    feature_vector = np.array([[feats.get(col, 0.0) for col in feature_cols]])
    feature_scaled = scaler.transform(feature_vector)

    # 4. Classical ML Inference
    print(f"\n[3/4] Running Inference via {model_name}...")
    prob_list = clf.predict_proba(feature_scaled)
    probabilities = {
        target: float(prob_list[i][0, 1]) if prob_list[i].shape[1] > 1 else 0.0
        for i, target in enumerate(target_cols)
    }

    # 5. Display Clinical Triage Results
    print("\n[4/4] Automated Radiological Triage Findings:")
    print("-" * 76)
    print(f"{'Finding / Pathology':<26} | {'Confidence':<12} | {'Triage Status':<22} | {'Flag'}")
    print("-" * 76)

    positive_findings = []
    for target in target_cols:
        prob = probabilities[target]
        if target == "normal":
            status = "Normal Anatomy" if prob >= 0.5 else "Abnormalities Present"
            flag = "[NORMAL]" if prob >= 0.5 else " [--]   "
        else:
            if prob >= 0.70:
                status = "High Likelihood"
                flag = "[ALERT] "
                positive_findings.append((target, prob))
            elif prob >= 0.40:
                status = "Moderate Suspicion"
                flag = "[WARN]  "
                positive_findings.append((target, prob))
            else:
                status = "Negative / Unremarkable"
                flag = " [--]   "

        print(f"{target:<26} | {prob:<12.1%} | {status:<22} | {flag}")

    print("-" * 76)

    # 6. White-Box Physical Explainability
    print("\n" + "=" * 76)
    print("      WHITE-BOX RADIOMICS EXPLANATION (CLINICAL REASONING)")
    print("=" * 76)

    if positive_findings:
        print("\nThe classical model identified the following key pathological drivers:")
        for target, prob in positive_findings:
            print(f"\n[+] Diagnostic Rationale for [{target.upper()}] ({prob:.1%} confidence):")
            top_features = pipeline.get("feature_importances", {}).get(target, [])
            for f_name, f_imp in top_features[:3]:
                measured_val = feats.get(f_name, 0.0)
                explanation = BIOMARKER_EXPLANATIONS.get(f_name, "Quantitative mathematical distribution marker")
                print(f"   * {f_name:<28} = {measured_val:.3f} (Feature Importance: {f_imp:.3f})")
                print(f"     -> {explanation}")
    else:
        print("\n[OK] Study exhibits uniform bilateral density symmetry, intact skull contour, and absence")
        print("   of hyperdense hemorrhagic pooling or hypodense ischemic edema.")

    print("\n" + "=" * 76)
    print("      DEMO COMPLETE -- 100% Classical Scikit-Learn Model")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Head AI Classical ML Demo")
    parser.add_argument("--case", type=str, default=None, help="Path to .npy CT volume")
    parser.add_argument("--condition", type=str, default="hemorrhage",
                        choices=["normal", "hemorrhage", "infarct", "fracture", "midline_shift"],
                        help="Benchmark clinical condition to simulate if no case provided")
    args = parser.parse_args()

    # Pre-generate sample cases if folder is empty
    sample_dir = BASE_DIR / "data" / "sample_cases"
    if not sample_dir.exists() or len(list(sample_dir.glob("*.npy"))) == 0:
        save_sample_cases(str(sample_dir))

    run_demo(case_path=args.case, condition=args.condition)
