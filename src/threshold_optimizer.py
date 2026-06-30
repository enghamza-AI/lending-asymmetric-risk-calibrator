
# threshold_optimizer.py


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   




def load_config(config_path="config/business_rules.yaml"):
    """Loads business rules from YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_profit_at_threshold(y_true, y_proba, threshold, config):

    cost = config["cost_matrix"]
    loss_given_default     = cost["loss_given_default"]
    revenue_per_good_loan  = cost["revenue_per_good_loan"]
    opportunity_cost       = cost["opportunity_cost_per_rejection"]

    
    y_pred = (y_proba > threshold).astype(int)

    
    TP = ((y_pred == 1) & (y_true == 1)).sum()  
    TN = ((y_pred == 0) & (y_true == 0)).sum()  
    FP = ((y_pred == 1) & (y_true == 0)).sum()  
    FN = ((y_pred == 0) & (y_true == 1)).sum()  

    
    profit = (TN * revenue_per_good_loan) \
           - (FN * loss_given_default) \
           - (FP * opportunity_cost)
    

   
    total = len(y_true)
    approval_rate = (TN + FN) / total   
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0   
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    metrics = {
        "threshold":    threshold,
        "profit":       profit,
        "recall":       recall,
        "precision":    precision,
        "approval_rate": approval_rate,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

    return profit, metrics




def compute_fairness_gap(y_true, y_proba, X_test_raw, threshold, config):


    fairness_attr = config["optimization"]["fairness_attribute"]

   
    if X_test_raw is None or fairness_attr not in X_test_raw.columns:
        return 0.0, {}

    y_pred = (y_proba > threshold).astype(int)
    approved = (y_pred == 0).astype(int)   

   
    group_col = X_test_raw[fairness_attr].reset_index(drop=True)

    group_rates = {}
    for group in group_col.unique():
        mask = (group_col == group).values
        if mask.sum() > 0:
            group_rates[group] = approved[mask].mean()

    if len(group_rates) < 2:
        return 0.0, group_rates

    rates = list(group_rates.values())
    fairness_gap = max(rates) - min(rates)

    return fairness_gap, group_rates




def sweep_thresholds(y_true, y_proba, X_test_raw, config):
   

    resolution = config["optimization"]["pareto_resolution"]
    thresholds = np.arange(resolution, 1.0, resolution)

    print(f"Sweeping {len(thresholds)} thresholds...")

    results = []
    for t in thresholds:
        profit, metrics = compute_profit_at_threshold(y_true, y_proba, t, config)
        fairness_gap, group_rates = compute_fairness_gap(
            y_true, y_proba, X_test_raw, t, config
        )
        metrics["fairness_gap"] = fairness_gap
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print(f"Threshold sweep complete. Shape: {results_df.shape}")

    return results_df



def is_dominated(row, all_rows):

    for _, other in all_rows.iterrows():
        if (other["profit"]       >= row["profit"] and
            other["recall"]       >= row["recall"] and
            other["fairness_gap"] <= row["fairness_gap"] and
            (other["profit"]       > row["profit"] or
             other["recall"]       > row["recall"] or
             other["fairness_gap"] < row["fairness_gap"])):
            return True
    return False


def compute_pareto_frontier(results_df):

    print("Computing Pareto frontier...")

    pareto_mask = []
    for idx, row in results_df.iterrows():
        dominated = is_dominated(row, results_df)
        pareto_mask.append(not dominated)

    pareto_df = results_df[pareto_mask].reset_index(drop=True)

    print(f"Total thresholds evaluated: {len(results_df)}")
    print(f"Pareto-optimal thresholds:  {len(pareto_df)}")
    print(f"Dominated (discarded):      {len(results_df) - len(pareto_df)}")

    return pareto_df




def optimize_threshold_scipy(y_true, y_proba, X_test_raw, config):
   
    constraints_config = config["constraints"]
    min_recall         = constraints_config["min_recall"]
    max_fairness_gap   = constraints_config["max_fairness_gap"]
    min_approval_rate  = constraints_config["min_approval_rate"]

    def objective(t):
    
        threshold = t[0]
        profit, _ = compute_profit_at_threshold(y_true, y_proba, threshold, config)
        return -profit   

    def constraint_recall(t):
       
        threshold = t[0]
        _, metrics = compute_profit_at_threshold(y_true, y_proba, threshold, config)
        return metrics["recall"] - min_recall

    def constraint_fairness(t):
       
        threshold = t[0]
        fairness_gap, _ = compute_fairness_gap(
            y_true, y_proba, X_test_raw, threshold, config
        )
        return max_fairness_gap - fairness_gap

    def constraint_approval(t):
      
        threshold = t[0]
        _, metrics = compute_profit_at_threshold(y_true, y_proba, threshold, config)
        return metrics["approval_rate"] - min_approval_rate

 
    bounds = [(0.01, 0.99)]

  
    scipy_constraints = [
        {"type": "ineq", "fun": constraint_recall},
        {"type": "ineq", "fun": constraint_fairness},
        {"type": "ineq", "fun": constraint_approval},
    ]

    
    x0 = [0.5]

    print("\nRunning constrained optimization (scipy SLSQP)...")

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
        
    )

    if result.success:
        optimal_threshold = result.x[0]
        optimal_profit = -result.fun   
        print(f"Optimization succeeded!")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Expected profit:   ${optimal_profit:,.2f}")
    else:
        print(f"WARNING: Optimization did not fully converge: {result.message}")
        print("Falling back to best threshold from Pareto sweep.")
        optimal_threshold = None

    return optimal_threshold, result




def sensitivity_analysis(y_true, y_proba, X_test_raw, config):


    perturbation = config["sensitivity"]["perturbation"]
    params_to_test = config["sensitivity"]["parameters_to_test"]

    print("\nRunning sensitivity analysis...")

   
    baseline_threshold, _ = optimize_threshold_scipy(y_true, y_proba, X_test_raw, config)
    if baseline_threshold is None:
        baseline_threshold = 0.5
    baseline_profit, _ = compute_profit_at_threshold(
        y_true, y_proba, baseline_threshold, config
    )

    results = []

    for param in params_to_test:
        for direction, multiplier in [("+10%", 1 + perturbation), ("-10%", 1 - perturbation)]:

           
            import copy
            modified_config = copy.deepcopy(config)

           
            if param in modified_config["cost_matrix"]:
                modified_config["cost_matrix"][param] *= multiplier
            elif param in modified_config["constraints"]:
                modified_config["constraints"][param] *= multiplier

           
            new_threshold, _ = optimize_threshold_scipy(
                y_true, y_proba, X_test_raw, modified_config
            )
            if new_threshold is None:
                new_threshold = baseline_threshold

            new_profit, _ = compute_profit_at_threshold(
                y_true, y_proba, new_threshold, config  
            )

            results.append({
                "parameter":        param,
                "direction":        direction,
                "new_threshold":    round(new_threshold, 4),
                "baseline_threshold": round(baseline_threshold, 4),
                "threshold_change": round(new_threshold - baseline_threshold, 4),
                "profit_change":    round(new_profit - baseline_profit, 2)
            })

    sensitivity_df = pd.DataFrame(results)
    return sensitivity_df, baseline_threshold




def plot_pareto_frontier(pareto_df, optimal_threshold,
                         save_path="outputs/pareto_frontier.png"):
  

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        pareto_df["recall"],
        pareto_df["profit"],
        c=pareto_df["fairness_gap"],
        cmap="RdYlGn_r",   
        s=80,
        alpha=0.8,
        zorder=3
    )

    plt.colorbar(scatter, ax=ax, label="Fairness Gap (lower = fairer)")

    
    if optimal_threshold is not None:
        opt_row = pareto_df.iloc[
            (pareto_df["threshold"] - optimal_threshold).abs().argsort()[:1]
        ]
        ax.scatter(
            opt_row["recall"],
            opt_row["profit"],
            marker="*",
            s=300,
            color="blue",
            zorder=5,
            label=f"Recommended threshold = {optimal_threshold:.3f}"
        )

    ax.set_xlabel("Recall (Fraction of Defaults Caught)", fontsize=12)
    ax.set_ylabel("Expected Profit ($)", fontsize=12)
    ax.set_title("Pareto Frontier — Profit vs Recall vs Fairness", fontsize=14)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Pareto frontier chart saved to: {save_path}")


def plot_sensitivity(sensitivity_df, save_path="outputs/sensitivity_analysis.png"):
   

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue" if d == "+10%" else "tomato"
              for d in sensitivity_df["direction"]]

    ax.barh(
        sensitivity_df["parameter"] + " " + sensitivity_df["direction"],
        sensitivity_df["threshold_change"],
        color=colors
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in Optimal Threshold")
    ax.set_title("Sensitivity Analysis — How Business Parameters Affect Optimal Threshold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sensitivity analysis chart saved to: {save_path}")




def run_optimization(config_path="config/business_rules.yaml",
                     probabilities_path="outputs/test_probabilities.csv",
                     X_test_raw=None):


    print("=" * 50)
    print("STARTING THRESHOLD OPTIMIZER")
    print("=" * 50)

    config = load_config(config_path)

   
    proba_data = pd.read_csv(probabilities_path)
    y_true  = proba_data["true_label"].values
    y_proba = proba_data["prob_default"].values

    print(f"Loaded {len(y_true):,} test predictions")

  
    results_df = sweep_thresholds(y_true, y_proba, X_test_raw, config)

  
    pareto_df = compute_pareto_frontier(results_df)

   
    optimal_threshold, _ = optimize_threshold_scipy(y_true, y_proba, X_test_raw, config)

  
    sensitivity_df, _ = sensitivity_analysis(y_true, y_proba, X_test_raw, config)

   
    plot_pareto_frontier(pareto_df, optimal_threshold)
    plot_sensitivity(sensitivity_df)

   
    pareto_df.to_csv("outputs/pareto_frontier.csv", index=False)
    sensitivity_df.to_csv("outputs/sensitivity_analysis.csv", index=False)

    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print(f"Recommended threshold: {optimal_threshold:.4f}")
    print("Pareto frontier saved to outputs/pareto_frontier.csv")
    print("Sensitivity analysis saved to outputs/sensitivity_analysis.csv")
    print("=" * 50)

    return optimal_threshold, pareto_df, sensitivity_df


if __name__ == "__main__":
    optimal_threshold, pareto_df, sensitivity_df = run_optimization()