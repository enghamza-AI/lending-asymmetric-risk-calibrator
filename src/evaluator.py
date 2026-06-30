
# evaluator.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import yaml
import os



def load_config(config_path="config/business_rules.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_all_metrics(y_true, y_proba, threshold, config, group_col=None):

    cost = config["cost_matrix"]

    
    y_pred = (y_proba > threshold).astype(int)

    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)

  

    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

   
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

   
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

   
    auc = roc_auc_score(y_true, y_proba)

    
    avg_precision = average_precision_score(y_true, y_proba)

   

  
    approval_rate = (tn + fn) / total

   
    approved_total = tn + fn
    default_rate_approved = fn / approved_total if approved_total > 0 else 0.0


    profit = (tn  * cost["revenue_per_good_loan"]) \
           - (fn  * cost["loss_given_default"]) \
           - (fp  * cost["opportunity_cost_per_rejection"])


    profit_per_loan = profit / total

  
    fairness_gap = 0.0
    group_approval_rates = {}

    if group_col is not None:
        approved = (y_pred == 0).astype(int)
        for group in group_col.unique():
            mask = (group_col == group).values
            if mask.sum() > 0:
                group_approval_rates[group] = approved[mask].mean()

        if len(group_approval_rates) >= 2:
            rates = list(group_approval_rates.values())
            fairness_gap = max(rates) - min(rates)

    return {
        
        "threshold":              round(threshold, 4),

        
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),

       
        "recall":                 round(recall, 4),
        "precision":              round(precision, 4),
        "f1_score":               round(f1, 4),
        "specificity":            round(specificity, 4),
        "false_positive_rate":    round(fpr, 4),
        "auc_roc":                round(auc, 4),
        "avg_precision_score":    round(avg_precision, 4),

        
        "approval_rate":          round(approval_rate, 4),
        "default_rate_approved":  round(default_rate_approved, 4),
        "expected_profit":        round(profit, 2),
        "profit_per_loan":        round(profit_per_loan, 4),

        
        "fairness_gap":           round(fairness_gap, 4),
        "group_approval_rates":   group_approval_rates,
    }




def compare_thresholds(y_true, y_proba, thresholds_to_compare, config, group_col=None):

    rows = []
    for name, threshold in thresholds_to_compare.items():
        metrics = compute_all_metrics(y_true, y_proba, threshold, config, group_col)
        metrics["name"] = name
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)

   
    display_cols = [
        "name", "threshold",
        "expected_profit", "profit_per_loan",
        "recall", "precision", "f1_score",
        "approval_rate", "default_rate_approved",
        "fairness_gap", "auc_roc"
    ]

   
    display_cols = [c for c in display_cols if c in comparison_df.columns]

    return comparison_df[display_cols]




def compute_breakeven_threshold(config):


    cost = config["cost_matrix"]
    revenue = cost["revenue_per_good_loan"]
    loss    = cost["loss_given_default"]

    breakeven = revenue / (loss + revenue)

    print(f"\nBreak-even analysis:")
    print(f"  Revenue per good loan:   ${revenue:,}")
    print(f"  Loss per default:        ${loss:,}")
    print(f"  Break-even threshold:    {breakeven:.4f}")
    print(f"  Interpretation: approve loans with P(default) < {breakeven:.4f}")
    print(f"  Above this probability, expected loss exceeds expected revenue.")

    return breakeven




def plot_precision_recall_curve(y_true, y_proba,
                                 save_path="outputs/precision_recall_curve.png"):
  

    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    aps = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()  

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="steelblue", lw=2,
             label=f"LightGBM (APS = {aps:.4f})")
    plt.axhline(y=baseline, color="gray", linestyle="--",
                label=f"Random classifier (precision = {baseline:.3f})")
    plt.xlabel("Recall (Fraction of Defaults Caught)")
    plt.ylabel("Precision (Fraction of Rejections That Were Defaults)")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Precision-recall curve saved to: {save_path}")



def generate_business_report(y_true, y_proba, optimal_threshold,
                              config, group_col=None):

    metrics = compute_all_metrics(y_true, y_proba, optimal_threshold, config, group_col)
    breakeven = compute_breakeven_threshold(config)

    total_loans = len(y_true)
    approved = int(metrics["approval_rate"] * total_loans)
    defaults_caught = metrics["TP"]
    defaults_missed = metrics["FN"]
    good_rejected   = metrics["FP"]

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           LOAN RISK CALIBRATOR — BUSINESS REPORT             ║
╠══════════════════════════════════════════════════════════════╣
║ RECOMMENDED THRESHOLD: {optimal_threshold:.4f}                           ║
╠══════════════════════════════════════════════════════════════╣
║ DECISION OUTCOMES (on {total_loans:,} test loans)                   ║
║   Approved:                 {approved:>8,} ({metrics['approval_rate']*100:.1f}%)           ║
║   Rejected:                 {total_loans - approved:>8,} ({(1-metrics['approval_rate'])*100:.1f}%)           ║
║                                                              ║
║   Defaults correctly caught: {defaults_caught:>7,} ({metrics['recall']*100:.1f}% recall)    ║
║   Defaults missed (approved): {defaults_missed:>6,} (cost: ${defaults_missed * config['cost_matrix']['loss_given_default']:,.0f})   ║
║   Good customers rejected:   {good_rejected:>7,} (cost: ${good_rejected * config['cost_matrix']['opportunity_cost_per_rejection']:,.0f})   ║
╠══════════════════════════════════════════════════════════════╣
║ FINANCIAL IMPACT                                             ║
║   Expected profit:          ${metrics['expected_profit']:>12,.2f}              ║
║   Profit per loan:          ${metrics['profit_per_loan']:>12.2f}              ║
╠══════════════════════════════════════════════════════════════╣
║ FAIRNESS                                                     ║
║   Approval rate gap:        {metrics['fairness_gap']*100:>7.1f}%                       ║
║   (Constraint: <= {config['constraints']['max_fairness_gap']*100:.0f}%)                            ║
╠══════════════════════════════════════════════════════════════╣
║ REFERENCE                                                    ║
║   Break-even threshold:     {breakeven:.4f}                           ║
║   Optimal threshold:        {optimal_threshold:.4f}                           ║
║   Model AUC-ROC:            {metrics['auc_roc']:.4f}                           ║
╚══════════════════════════════════════════════════════════════╝
"""

    print(report)

    # Save report to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/business_report.txt", "w") as f:
        f.write(report)
    print("Business report saved to outputs/business_report.txt")

    return report




def run_evaluation(optimal_threshold, config_path="config/business_rules.yaml",
                   probabilities_path="outputs/test_probabilities.csv",
                   X_test_raw=None):

    print("=" * 50)
    print("STARTING EVALUATOR")
    print("=" * 50)

    config = load_config(config_path)
    proba_data = pd.read_csv(probabilities_path)
    y_true  = proba_data["true_label"].values
    y_proba = proba_data["prob_default"].values

    group_col = None
    if X_test_raw is not None:
        fairness_attr = config["optimization"]["fairness_attribute"]
        if fairness_attr in X_test_raw.columns:
            group_col = X_test_raw[fairness_attr].reset_index(drop=True)

    # Compare key thresholds
    thresholds_to_compare = {
        "Default (0.50)":   0.50,
        "Optimal":          optimal_threshold,
        "Conservative":     max(0.01, optimal_threshold - 0.10),
        "Aggressive":       min(0.99, optimal_threshold + 0.10),
    }

    comparison_df = compare_thresholds(y_true, y_proba, thresholds_to_compare, config, group_col)

    print("\nThreshold Comparison Table:")
    print(comparison_df.to_string(index=False))

  
    plot_precision_recall_curve(y_true, y_proba)

   
    generate_business_report(y_true, y_proba, optimal_threshold, config, group_col)

   
    metrics = compute_all_metrics(y_true, y_proba, optimal_threshold, config, group_col)

    print("\n" + "=" * 50)
    print("EVALUATOR COMPLETE")
    print("=" * 50)

    return metrics, comparison_df


if __name__ == "__main__":
    print("Run this file after threshold_optimizer.py has completed.")
    print("It requires outputs/test_probabilities.csv to exist.")