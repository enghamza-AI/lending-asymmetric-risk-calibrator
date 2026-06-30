
# train_model.py


import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import lightgbm as lgb
import joblib
import yaml
import os
import matplotlib.pyplot as plt



def load_config(config_path="config/business_rules.yaml"):
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_params(y_train):


    
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    ratio = n_negative / n_positive

    print(f"Class imbalance ratio: {ratio:.2f}:1")
    print(f"Using class_weight='balanced' to compensate")

    params = {
        "objective":         "binary",
        "metric":            "auc",
        "n_estimators":      1000,
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 20,
        "class_weight":      "balanced",
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "random_state":      42,
        "verbose":           -1,   
        "n_jobs":            -1,    
    }

    return params




def train_model(X_train, y_train, X_test, y_test):

    print("=" * 50)
    print("STARTING MODEL TRAINING")
    print("=" * 50)

    params = get_model_params(y_train)

    model = lgb.LGBMClassifier(**params)
 

   
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  
        callbacks=callbacks
    )

    print(f"\nTraining complete.")
    print(f"Best iteration: {model.best_iteration_} trees")
    print(f"(Out of maximum {params['n_estimators']})")

    return model




def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names=None):

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]

    
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc  = roc_auc_score(y_test,  test_proba)

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")

    

    gap = train_auc - test_auc
    if gap > 0.05:
        print(f"WARNING: AUC gap = {gap:.4f} — possible overfitting")
        print("Consider: reduce num_leaves, increase min_child_samples")
    else:
        print(f"AUC gap = {gap:.4f} — model generalizes well")

    
    test_preds = model.predict(X_test)  
    cm = confusion_matrix(y_test, test_preds)

    print("\nConfusion Matrix (at threshold=0.5):")
    print(f"  True Negatives  (approved, repaid):   {cm[0][0]:,}")
    print(f"  False Positives (rejected, would repay): {cm[0][1]:,}")
    print(f"  False Negatives (approved, defaulted): {cm[1][0]:,}")
    print(f"  True Positives  (rejected, defaulted): {cm[1][1]:,}")

    print("\nClassification Report (at threshold=0.5):")
    print(classification_report(y_test, test_preds, target_names=["Fully Paid", "Charged Off"]))
    print("NOTE: This report uses threshold=0.5.")
    print("threshold_optimizer.py will find the REAL optimal threshold.")

    return test_proba


def plot_roc_curve(y_test, test_proba, save_path="outputs/roc_curve.png"):

    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    auc = roc_auc_score(y_test, test_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"LightGBM (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate (Wrongly Rejected Good Loans)")
    plt.ylabel("True Positive Rate (Correctly Caught Defaults)")
    plt.title("ROC Curve — Lending Club Default Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def plot_feature_importance(model, feature_names, save_path="outputs/feature_importance.png"):

    if feature_names is None:
        print("No feature names provided — skipping importance plot")
        return

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]  

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in reversed(indices)],
        [importance[i] for i in reversed(indices)],
        color="steelblue"
    )
    plt.xlabel("Feature Importance (split count)")
    plt.title("Top 15 Features — LightGBM Default Prediction")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")




def save_model(model, path="outputs/lgbm_model.joblib"):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def save_test_probabilities(y_test, test_proba, path="outputs/test_probabilities.csv"):

    results = pd.DataFrame({
        "true_label": y_test,
        "prob_default": test_proba
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=False)
    print(f"Test probabilities saved to: {path}")
    print(f"Shape: {results.shape}")
    print(f"Default rate in test set: {results['true_label'].mean():.3f}")


def load_model(path="outputs/lgbm_model.joblib"):
    """Loads saved model from disk. Use in app.py for inference."""
    return joblib.load(path)




def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names=None):

    model = train_model(X_train, y_train, X_test, y_test)

  
    test_proba = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)

   
    plot_roc_curve(y_test, test_proba)
    plot_feature_importance(model, feature_names)

   
    save_model(model)
    save_test_probabilities(y_test, test_proba)

    print("\nTrain model pipeline complete.")
    print("Next step: run threshold_optimizer.py")

    return model, test_proba




if __name__ == "__main__":
    
    from sklearn.datasets import make_classification

    print("Running train_model smoke test with synthetic data...")

   
    X_fake, y_fake = make_classification(
        n_samples=1000,
        n_features=20,
        weights=[0.82, 0.18],   
        random_state=42
    )

    split = 800
    X_tr, X_te = X_fake[:split], X_fake[split:]
    y_tr, y_te = y_fake[:split], y_fake[split:]

    model, proba = train_and_evaluate(X_tr, X_te, y_tr, y_te)

    print(f"\nTest probabilities sample: {proba[:5]}")
    print("Smoke test passed.")