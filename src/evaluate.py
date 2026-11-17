"""
CT Head AI — Classical Model Evaluator & Clinical Interpretability Report
Generates academic benchmarking tables, ROC analysis, and physical feature correlation reports.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Comprehensive radiological explanation map for physical features
BIOMARKER_EXPLANATIONS = {
    # Brain Window
    "brain_mean": "Mean baseline attenuation of brain parenchyma (~20-40 HU)",
    "brain_std": "Attenuation variance reflecting tissue heterogeneity",
    "brain_p90": "Upper quantile attenuation capturing high-density parenchymal components",
    "brain_p95": "Elevated 95th percentile HU signals acute extravasated blood pooling (50-90 HU)",
    "brain_p99": "Severe peak attenuation confirming focal hyperdense hematoma core",
    "brain_skewness": "Positive distribution skew confirms hyperdense hemorrhagic component",
    "brain_kurtosis": "Peakedness of density distribution reflecting focal density outliers",
    "brain_hyperdense_ratio": "Percentage of brain parenchyma exceeding 60 HU acute blood threshold",
    "brain_hypodense_ratio": "Percentage of tissue below 20 HU reflecting cytotoxic edema or ischemic infarct",

    # Subdural Window
    "subdural_mean": "Mean attenuation in subdural space capturing extra-axial fluid collections",
    "subdural_std": "Density variance confirming mixed-age or active extravasation hematoma",
    "subdural_p90": "Upper 90th percentile intensity indicating subdural pooling",
    "subdural_p95": "High subdural window intensity capturing extra-axial crescentic/lens hematomas",
    "subdural_p99": "Extremely high 99th percentile HU confirms acute extravasated hyperdense blood pooling (50-90 HU)",
    "subdural_skewness": "Positive distribution skew confirming hyperdense hemorrhagic component",
    "subdural_kurtosis": "Heavy-tailed density profile confirming focal extra-axial hematoma",
    "subdural_hyperdense_ratio": "Percentage of subdural volume occupied by acute extravasated blood (>60 HU)",
    "subdural_hypodense_ratio": "Hypodense subdural fluid collection reflecting chronic hematoma / hygroma",

    # Bone Window
    "bone_mean": "Baseline cortical bone attenuation density",
    "bone_std": "Bone attenuation variance",
    "bone_p95": "Cortical calvarium density profile",
    "bone_p99": "Dense cortical bone threshold (>1000 HU)",
    "bone_edge_density_mean": "High Canny edge discontinuity flags cortical skull fracture breaks",
    "bone_edge_density_max": "Peak cortical edge discontinuity confirming comminuted fracture",
    "bone_gradient_mag_mean": "High Sobel gradient magnitude confirms displaced bone fragments",

    # Symmetry (Midline Shift)
    "hemispheric_asymmetry_mean": "Left-right density disparity quantifies midline shift & mass effect",
    "hemispheric_asymmetry_max": "Peak lateral hemispheric density disparity indicating severe mass displacement",
    "hemispheric_asymmetry_p95": "95th-percentile bilateral disparity confirming ventricular compression and shift",

    # Texture (GLCM)
    "glcm_contrast": "Disrupted local tissue texture reflects ischemic edema or parenchymal damage",
    "glcm_homogeneity": "Reduced homogeneity indicates loss of normal grey-white matter interface",
    "glcm_energy": "Reduced GLCM energy indicates parenchymal tissue disruption from acute infarction",
}

def evaluate_pipeline():
    model_path = BASE_DIR / "models" / "classical_pipeline.joblib"
    val_path   = BASE_DIR / "data" / "features_val.csv"

    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run train_classical.py first.")
        return

    pipeline = joblib.load(model_path)
    val_df   = pd.read_csv(val_path)

    clf          = pipeline["classifier"]
    scaler       = pipeline["scaler"]
    feature_cols = pipeline["feature_cols"]
    target_cols  = pipeline["target_cols"]
    model_name   = pipeline["model_name"]

    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df[target_cols].values

    # Predict probabilities
    prob_list = clf.predict_proba(X_val)
    preds = np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_val)) for p in prob_list])
    pred_binary = (preds >= 0.5).astype(int)

    print("\n" + "=" * 78)
    print(f"   ACADEMIC EVALUATION REPORT: {model_name.upper()} (CLASSICAL ML)")
    print("=" * 78)
    print(f"{'Finding / Pathology':<20} | {'ROC-AUC':<8} | {'Accuracy':<9} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Score':<8}")
    print("-" * 78)

    records = []
    for i, target in enumerate(target_cols):
        y_true = y_val[:, i]
        y_pred = pred_binary[:, i]
        y_score = preds[:, i]

        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
        acc = accuracy_score(y_true, y_pred)

        # Confusion Matrix: tn, fp, fn, tp
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (len(y_true), 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        records.append({
            "Pathology": target,
            "AUC": auc,
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1_Score": f1
        })

        print(f"{target:<20} | {auc:<8.3f} | {acc:<9.1%} | {sensitivity:<11.1%} | {specificity:<11.1%} | {f1:<8.3f}")

    mean_auc = np.mean([r['AUC'] for r in records])
    mean_acc = np.mean([r['Accuracy'] for r in records])
    print("-" * 78)
    print(f"{'OVERALL AVERAGE':<20} | {mean_auc:<8.3f} | {mean_acc:<9.1%} |")
    print("=" * 78)

    # Print Feature Importance Explanations
    importances = pipeline.get("feature_importances", {})
    if importances:
        print("\n" + "=" * 78)
        print("   CLINICAL FEATURE IMPORTANCE & PHYSICAL RADIOLOGICAL INTERPRETABILITY")
        print("=" * 78)
        for target, top_feats in importances.items():
            print(f"\n[+] Diagnostic Basis for [{target.upper()}]:")
            for f_name, score in top_feats[:3]:
                explanation = BIOMARKER_EXPLANATIONS.get(f_name, "Quantitative mathematical distribution marker")
                print(f"   * {f_name:<28} (Weight: {score:.3f})")
                print(f"     -> Clinical Meaning: {explanation}")

    # Save to CSV report
    reports_dir = BASE_DIR / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_df = pd.DataFrame(records)
    report_df.to_csv(reports_dir / "evaluation_metrics.csv", index=False)
    print(f"\n[OK] Formatted metrics saved to {reports_dir / 'evaluation_metrics.csv'}")

if __name__ == "__main__":
    evaluate_pipeline()
