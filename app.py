
# app.py


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from threshold_optimizer import (
    compute_profit_at_threshold,
    compute_fairness_gap,
    sweep_thresholds,
    compute_pareto_frontier,
    optimize_threshold_scipy,
)
from evaluator import compute_all_metrics, compute_breakeven_threshold




st.set_page_config(
    page_title="Lending Risk Calibrator",
    page_icon="📊",
    layout="wide",         
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4f8bf9;
    }
    .warning-card {
        background-color: #2d1b1b;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
    }
    .success-card {
        background-color: #1b2d1b;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00c853;
    }
    </style>
""", unsafe_allow_html=True)




@st.cache_data
def load_probabilities(path="outputs/test_probabilities.csv"):
  
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_saved_config(path="config/business_rules.yaml"):

    with open(path, "r") as f:
        return yaml.safe_load(f)




def render_sidebar():


    st.sidebar.title("⚙️ Business Rules")
    st.sidebar.markdown("Adjust parameters and see the Pareto frontier update live.")

    st.sidebar.subheader("💰 Cost Matrix")

    

    loss_given_default = st.sidebar.slider(
        "Loss Given Default ($)",
        min_value=5000,
        max_value=50000,
        value=15000,
        step=1000,
        help="How much the bank loses when an approved loan defaults. "
             "Includes principal loss minus recovered collateral."
    )

    revenue_per_good_loan = st.sidebar.slider(
        "Revenue per Good Loan ($)",
        min_value=500,
        max_value=5000,
        value=1200,
        step=100,
        help="Net interest income earned when an approved loan is fully repaid."
    )

    opportunity_cost = st.sidebar.slider(
        "Opportunity Cost per Rejection ($)",
        min_value=0,
        max_value=2000,
        value=300,
        step=50,
        help="Revenue foregone when a good customer is incorrectly rejected."
    )

   
    st.sidebar.subheader("📋 Business Constraints")

    min_approval_rate = st.sidebar.slider(
        "Minimum Approval Rate",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        format="%.0f%%",
        help="Minimum fraction of applicants that must be approved. "
             "Regulators and business teams set this floor."
    )

    max_fairness_gap = st.sidebar.slider(
        "Max Fairness Gap",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.0f%%",
        help="Maximum allowed approval rate difference between demographic groups."
    )

    min_recall = st.sidebar.slider(
        "Minimum Recall",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        format="%.0f%%",
        help="Minimum fraction of actual defaults that must be caught."
    )

   
    config = {
        "cost_matrix": {
            "loss_given_default":          loss_given_default,
            "revenue_per_good_loan":       revenue_per_good_loan,
            "opportunity_cost_per_rejection": opportunity_cost,
        },
        "constraints": {
            "min_approval_rate": min_approval_rate,
            "max_fairness_gap":  max_fairness_gap,
            "min_recall":        min_recall,
        },
        "optimization": {
            "fairness_attribute": "home_ownership",
            "pareto_resolution":  0.01,
            "scipy_method":       "SLSQP",
        },
        "sensitivity": {
            "perturbation": 0.10,
            "parameters_to_test": [
                "loss_given_default",
                "revenue_per_good_loan",
                "min_approval_rate",
                "max_fairness_gap"
            ]
        }
    }

    return config




def render_pareto_chart(pareto_df, optimal_threshold):

    if pareto_df is None or len(pareto_df) == 0:
        st.warning("No Pareto frontier data available. Run the pipeline first.")
        return

    fig = go.Figure()

   
    fig.add_trace(go.Scatter(
        x=pareto_df["recall"],
        y=pareto_df["profit"],
        mode="markers",
        marker=dict(
            size=10,
            color=pareto_df["fairness_gap"],
            colorscale="RdYlGn_r",     
            showscale=True,
            colorbar=dict(title="Fairness Gap"),
            line=dict(width=1, color="white")
        ),
        text=[
            f"Threshold: {row['threshold']:.3f}<br>"
            f"Profit: ${row['profit']:,.0f}<br>"
            f"Recall: {row['recall']:.3f}<br>"
            f"Fairness Gap: {row['fairness_gap']:.3f}<br>"
            f"Approval Rate: {row['approval_rate']:.3f}"
            for _, row in pareto_df.iterrows()
        ],
        hoverinfo="text",
        name="Pareto Frontier"
    ))

   
    if optimal_threshold is not None:
        opt_idx = (pareto_df["threshold"] - optimal_threshold).abs().idxmin()
        opt_row = pareto_df.loc[opt_idx]

        fig.add_trace(go.Scatter(
            x=[opt_row["recall"]],
            y=[opt_row["profit"]],
            mode="markers",
            marker=dict(symbol="star", size=20, color="royalblue",
                        line=dict(width=2, color="white")),
            name=f"Recommended (t={optimal_threshold:.3f})",
            hoverinfo="text",
            text=[f"RECOMMENDED<br>Threshold: {optimal_threshold:.3f}"]
        ))

    fig.update_layout(
        title="Pareto Frontier — Profit vs Recall vs Fairness",
        xaxis_title="Recall (Fraction of Defaults Caught)",
        yaxis_title="Expected Profit ($)",
        yaxis_tickformat="$,.0f",
        height=500,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)



def render_metrics(metrics, config, optimal_threshold):

    st.subheader("📈 Metrics at Recommended Threshold")

   
    constraints = config["constraints"]
    recall_ok   = metrics["recall"]        >= constraints["min_recall"]
    approval_ok = metrics["approval_rate"] >= constraints["min_approval_rate"]
    fairness_ok = metrics["fairness_gap"]  <= constraints["max_fairness_gap"]
    all_ok = recall_ok and approval_ok and fairness_ok

    if all_ok:
        st.success("✅ All business constraints satisfied at recommended threshold")
    else:
        violations = []
        if not recall_ok:
            violations.append(f"Recall {metrics['recall']:.3f} < minimum {constraints['min_recall']}")
        if not approval_ok:
            violations.append(f"Approval rate {metrics['approval_rate']:.3f} < minimum {constraints['min_approval_rate']}")
        if not fairness_ok:
            violations.append(f"Fairness gap {metrics['fairness_gap']:.3f} > maximum {constraints['max_fairness_gap']}")
        st.warning(f"⚠️ Constraint violations: {' | '.join(violations)}")

    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Expected Profit",
            f"${metrics['expected_profit']:,.0f}",
            help="Total expected profit across all test loans at this threshold"
        )
    with col2:
        st.metric(
            "Approval Rate",
            f"{metrics['approval_rate']*100:.1f}%",
            delta=f"Min: {constraints['min_approval_rate']*100:.0f}%",
            delta_color="off"
        )
    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']*100:.1f}%",
            delta=f"Min: {constraints['min_recall']*100:.0f}%",
            delta_color="off"
        )
    with col4:
        st.metric(
            "Fairness Gap",
            f"{metrics['fairness_gap']*100:.1f}%",
            delta=f"Max: {constraints['max_fairness_gap']*100:.0f}%",
            delta_color="off"
        )

   
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}",
                  help="Model's ranking ability — threshold independent")
    with col6:
        st.metric("Precision", f"{metrics['precision']*100:.1f}%",
                  help="Of loans rejected, fraction that were actual defaults")
    with col7:
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}",
                  help="Harmonic mean of precision and recall (reference only)")
    with col8:
        st.metric("Default Rate (Approved)", f"{metrics['default_rate_approved']*100:.1f}%",
                  help="Fraction of approved loans that actually defaulted")




def render_confusion_matrix(metrics):

    st.subheader("🔲 Confusion Matrix")

    tp, tn, fp, fn = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]

    z = [[tn, fp], [fn, tp]]
    text = [
        [f"TN: {tn:,}<br>Correctly Approved<br>(Good Revenue)",
         f"FP: {fp:,}<br>Wrongly Rejected<br>(Lost Revenue)"],
        [f"FN: {fn:,}<br>Missed Defaults<br>(HIGH COST)",
         f"TP: {tp:,}<br>Correctly Rejected<br>(Loss Avoided)"]
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        xgap=3, ygap=3
    ))

    fig.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=["Predicted: Approve", "Predicted: Reject"]),
        yaxis=dict(tickvals=[0, 1], ticktext=["Actual: Good", "Actual: Default"]),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)



def main():

    # Header
    st.title("📊 Lending Asymmetric Risk Calibrator")
    st.markdown(
        "Multi-objective loan threshold optimizer · "
        "Balances **profit**, **recall**, and **fairness** on 2.3M Lending Club loans · "
        "[GitHub](https://github.com/enghamza-AI/lending-asymmetric-risk-calibrator) · "
        "[HuggingFace](https://huggingface.co/spaces/enghamza-AI/loan-risk-calibrator)"
    )

    st.divider()

   
    config = render_sidebar()

   
    proba_data = load_probabilities()

    if proba_data is None:
      
        st.info(
            "⚙️ **Model output not found.** "
            "Run the pipeline first:\n\n"
            "```bash\n"
            "python src/data_loader.py\n"
            "python src/feature_engineering.py  \n"
            "python src/train_model.py\n"
            "python src/threshold_optimizer.py\n"
            "```\n\n"
            "Then refresh this page."
        )
        st.stop()   

    y_true  = proba_data["true_label"].values
    y_proba = proba_data["prob_default"].values

    
    with st.spinner("Computing Pareto frontier with your business rules..."):
        results_df  = sweep_thresholds(y_true, y_proba, None, config)
        pareto_df   = compute_pareto_frontier(results_df)
        optimal_threshold, opt_result = optimize_threshold_scipy(y_true, y_proba, None, config)

    if optimal_threshold is None:
        st.warning(
            "⚠️ Optimization could not find a threshold satisfying all constraints. "
            "Try relaxing the minimum recall or maximum fairness gap in the sidebar."
        )
        optimal_threshold = 0.5   

   
    breakeven = compute_breakeven_threshold(config)

    metrics = compute_all_metrics(y_true, y_proba, optimal_threshold, config)

  
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🎯 Pareto Frontier")
        st.caption(
            "Each point is a Pareto-optimal threshold. "
            "Hover for details. "
            "Color = fairness gap (greener = fairer). "
            "★ = recommended threshold."
        )
        render_pareto_chart(pareto_df, optimal_threshold)

    with col_right:
        st.subheader("💡 Recommendation")
        st.markdown(f"""
        **Recommended threshold:** `{optimal_threshold:.4f}`

        **Plain English:**
        Reject any loan application where the model's
        predicted default probability exceeds **{optimal_threshold*100:.1f}%**.

        **Break-even reference:** `{breakeven:.4f}`
        *(theoretical optimum with no constraints)*

        **Why this threshold?**
        It maximizes expected profit while satisfying:
        - Recall ≥ {config['constraints']['min_recall']*100:.0f}%
        - Approval rate ≥ {config['constraints']['min_approval_rate']*100:.0f}%
        - Fairness gap ≤ {config['constraints']['max_fairness_gap']*100:.0f}%
        """)

    st.divider()

    # Metrics
    render_metrics(metrics, config, optimal_threshold)

    st.divider()

    # Confusion matrix
    render_confusion_matrix(metrics)

    st.divider()

    # Threshold sweep table
    with st.expander("📋 Full Threshold Sweep Data", expanded=False):
        st.caption(
            "All thresholds evaluated during the sweep. "
            "Highlighted rows are Pareto-optimal."
        )
        pareto_thresholds = set(pareto_df["threshold"].values)
        display_df = results_df[[
            "threshold", "profit", "recall", "precision",
            "approval_rate", "fairness_gap"
        ]].copy()
        display_df["pareto_optimal"] = display_df["threshold"].isin(pareto_thresholds)
        st.dataframe(display_df, use_container_width=True, height=300)

    # Footer
    st.divider()
    st.caption(
        "Built by [Hamza](https://github.com/enghamza-AI) · "
        "Stage 2 Week 4 · AI Systems Engineering Track · "
        "Lending Club Dataset (2.3M loans, 2007–2018)"
    )




if __name__ == "__main__":
    main()