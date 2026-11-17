"""
CT Head AI — Classical Machine Learning Trainer
Trains and benchmarks traditional Scikit-Learn classifiers on handcrafted radiomic features.
Outputs:
  - Per-finding ROC-AUC, Accuracy, Precision, Recall, and F1-Scores.
  - White-box feature importance tables explaining which radiomic markers drive predictions.
  - Serialized pipeline saved as models/classical_pipeline.joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.dataset_generator import LABEL_COLS, build_tabular_dataset

# Focus on primary clinical triage targets
KEY_TARGETS = [
    'ICH', 'subdural', 'epidural', 'infarct',
    'fracture', 'midline_shift', 'normal'
]

def load_or_generate_datasets():
    """Loads features CSVs or builds them automatically from split files."""
    train_path = BASE_DIR / "data" / "features_train.csv"
    val_path   = BASE_DIR / "data" / "features_val.csv"
    
    if not train_path.exists():
        print("features_train.csv not found. Building dataset...")
        split_train = BASE_DIR / "data" / "splits" / "indian_train.csv"
        build_tabular_dataset(split_train, train_path, max_samples=2500)

    if not val_path.exists():
        print("features_val.csv not found. Building dataset...")
        split_val = BASE_DIR / "data" / "splits" / "indian_val.csv"
        build_tabular_dataset(split_val, val_path, max_samples=600)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # Separate feature columns from labels and metadata
    metadata_cols = ['case_id'] + LABEL_COLS
    feature_cols = [c for c in train_df.columns if c not in metadata_cols]

    return train_df, val_df, feature_cols

def train_and_benchmark():
    print("=" * 70)
    print("  CT Head AI — Classical Machine Learning Benchmark")
    print("=" * 70)

    train_df, val_df, feature_cols = load_or_generate_datasets()
    print(f"Features: {len(feature_cols)} handcrafted radiomic metrics")
    print(f"Train Cases: {len(train_df)} | Validation Cases: {len(val_df)}")

    # Target labels to train (use KEY_TARGETS or all 17)
    target_cols = [c for c in KEY_TARGETS if c in train_df.columns]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_cols].values

    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # ── Define Classical Classifiers ──
    models = {
        "Random Forest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),
        "L2-Logistic Regression": MultiOutputClassifier(
            LogisticRegression(
                C=1.0,
                penalty="l2",
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    }

    results = {}
    best_model_name = None
    best_mean_auc = -1.0
    best_pipeline = None

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)

        # Predict probabilities
        # MultiOutputClassifier gives a list of (N, 2) arrays
        prob_list = clf.predict_proba(X_val_scaled)
        # Extract positive class probability for each target
        preds = np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_val)) for p in prob_list])
        pred_binary = (preds >= 0.5).astype(int)

        # Compute Metrics
        aucs = {}
        accuracies = {}
        f1s = {}

        for i, col in enumerate(target_cols):
            try:
                auc = roc_auc_score(y_val[:, i], preds[:, i])
            except Exception:
                auc = 0.5
            aucs[col] = auc
            accuracies[col] = accuracy_score(y_val[:, i], pred_binary[:, i])
            p, r, f1, _ = precision_recall_fscore_support(y_val[:, i], pred_binary[:, i], average="binary", zero_division=0)
            f1s[col] = f1

        mean_auc = np.mean(list(aucs.values()))
        results[name] = {
            "mean_auc": mean_auc,
            "per_target_auc": aucs,
            "accuracies": accuracies,
            "f1_scores": f1s
        }

        print(f"  --- {name} Results ---")
        print(f"  Mean ROC-AUC: {mean_auc:.4f}")
        for col in target_cols:
            print(f"    * {col:18s} | AUC: {aucs[col]:.3f} | Acc: {accuracies[col]:.1%} | F1: {f1s[col]:.3f}")

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_model_name = name
            best_pipeline = {
                "model_name": name,
                "classifier": clf,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "metrics": results[name]
            }

    # ── Feature Importance Analysis (White-Box Interpretability) ──
    print(f"\nExtracting Clinical Feature Importances from {best_model_name}...")
    rf_clf = best_pipeline["classifier"]
    feature_importances = {}
    
    if hasattr(rf_clf, "estimators_"):
        for i, target in enumerate(target_cols):
            est = rf_clf.estimators_[i]
            if hasattr(est, "feature_importances_"):
                imps = est.feature_importances_
                top_idx = np.argsort(imps)[::-1][:5]
                top_features = [(feature_cols[idx], float(imps[idx])) for idx in top_idx]
                feature_importances[target] = top_features
                print(f"\n  Top Biomarkers for [{target.upper()}]:")
                for f_name, f_imp in top_features:
                    print(f"    - {f_name:30s} (Importance: {f_imp:.3f})")

    best_pipeline["feature_importances"] = feature_importances

    # ── Save Trained Model ──
    models_dir = BASE_DIR / "models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = models_dir / "classical_pipeline.joblib"
    joblib.dump(best_pipeline, save_path)
    print(f"\n[OK] Successfully saved classical model pipeline to {save_path}")

    # Also save summary metrics JSON
    metrics_path = models_dir / "benchmark_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved benchmark metrics to {metrics_path}")

    return best_pipeline

if __name__ == "__main__":
    train_and_benchmark()
