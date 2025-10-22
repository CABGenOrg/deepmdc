"""
Optimized script for INOVA project data analysis.

This script consolidates and refines multiple sections from the original
Jupyter notebook into a single, modular, and efficient pipeline for data
processing, frequency analysis, attention coefficient analysis, and performance
evaluation.

Key features:
- Unified data loading and preprocessing.
- Automated creation of output directories for better file organization.
- Dedicated functions for saving CSVs and plots with specified parameters
  (300dpi, PNG).
- Refined data transformation from wide to long format.
- Consolidated generation of all heatmaps, boxplots, and bar charts.
- All comments and outputs are in English for consistency.
"""

import hydra
import warnings
import pandas as pd
import seaborn as sns
from os import path, makedirs
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import src.utils.handle_files as hf
import src.utils.handle_charts as hc
import src.utils.data_analysis_utils as dau
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, confusion_matrix

# Suppress the UserWarning from seaborn/matplotlib due to tight_layout
warnings.filterwarnings("ignore", category=UserWarning,
                        message="This figure includes Axes that are not "
                        "compatible with tight_layout, so tight_layout can't "
                        "be applied.")
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    input_table = path.abspath(cfg.analysis.input_table)
    analysis_output = path.abspath(cfg.analysis.analysis_output_path)
    analysis_output_csv = path.join(analysis_output, "csv")
    analysis_output_plots = path.join(analysis_output, "plots")

    if not path.exists(input_table):
        raise FileNotFoundError(f"{input_table} not found.")

    makedirs(analysis_output_csv)
    makedirs(analysis_output_plots)

    df = dau.preprocess_dataframe(hf.read_tsv(input_table))
    df = dau.create_unique_product_names(df)

    dau.save_df_to_csv(df, path.join(analysis_output_csv,
                                     "general_model_unique.csv"))

    # Define groups
    groups_info = {
        "Group1_S-S": (df["prediction"] == "S") & (df["label"] == "S"),
        "Group2_R-R": (df["prediction"] == "R") & (df["label"] == "R"),
        "Group3_R-S": (df["prediction"] == "R") & (df["label"] == "S"),
        "Group4_S-R": (df["prediction"] == "S") & (df["label"] == "R"),
    }

    # Unpivot the dataframe for easier analysis
    unpivoted_df = dau.unpivot_dataframe(df)

    # ==============================================================================
    # SECTION 1: HEATMAPS AND TOP 20 FREQUENCY ANALYSIS (TABLES ONLY)
    # ==============================================================================
    print("\n--- Generating Heatmaps and Top 20 Frequency Tables ---")

    for name, condition in groups_info.items():
        group_df = df[condition].copy()
        if not group_df.empty:
            dau.save_df_to_csv(group_df,
                               path.join(analysis_output_csv, f"{name}.csv"))
            hc.generate_heatmap(group_df, name, analysis_output_plots)

    # --- Combined Heatmap ---
    df_combined = pd.DataFrame()
    for name, condition in groups_info.items():
        group_df = df[condition].copy()
        group_df["Group"] = name
        df_combined = pd.concat([df_combined, group_df], ignore_index=True)

    df_combined["model_group"] = df_combined["Group"] + \
        "_" + df_combined["model"]
    product_cols_combined = [
        col for col in df_combined.columns if col.startswith("product")]
    df_long_combined = df_combined.melt(
        id_vars=["model_group"], value_vars=product_cols_combined,
        var_name="product_col",
        value_name="product").dropna(subset=["product"])
    count_combined = df_long_combined.groupby(
        ["model_group", "product"]).size().reset_index(name="count")
    heatmap_data_combined = count_combined.pivot(
        index="model_group", columns="product", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(200, 15))
    sns.heatmap(heatmap_data_combined, cmap="inferno_r",
                linewidths=.5, linecolor="white", ax=ax)
    ax.set_title("Combined Heatmap", fontsize=18, fontweight="bold")
    ax.set_xlabel("Products")
    ax.set_ylabel("Model and Group")
    fig.tight_layout()
    dau.save_plot_to_png(
        fig,
        path.join(analysis_output_plots, "heatmap_overview_by_groups.png")
    )

    # --- Top 20 Frequency Tables (Original implementation) ---
    print("--- Top 20 Frequency Tables (Original) ---")
    # Global Top 20
    global_top_20 = unpivoted_df['product'].value_counts().nlargest(
        20).reset_index()
    global_top_20.columns = ['product', 'count']
    dau.save_df_to_csv(
        global_top_20,
        path.join(analysis_output_csv, 'global_top20_count_summary.csv')
    )

    # Top 20 for each model and group
    for model_name in unpivoted_df['model'].unique():
        model_df = unpivoted_df[unpivoted_df['model'] == model_name]
        model_top_20 = model_df['product'].value_counts().nlargest(
            20).reset_index()
        model_top_20.columns = ['product', 'count']
        dau.save_df_to_csv(
            model_top_20,
            path.join(analysis_output_csv,
                      f'{model_name}_top20_count_summary.csv')
        )

    for group_name, condition in groups_info.items():
        group_df = \
            unpivoted_df[unpivoted_df.set_index(['id', 'model']).index.isin(
                df[condition].set_index(['id', 'model']).index)]
        group_top_20 = group_df['product'].value_counts().nlargest(
            20).reset_index()
        group_top_20.columns = ['product', 'count']
        dau.save_df_to_csv(
            group_top_20,
            path.join(analysis_output_csv,
                      f'{group_name}_top20_count_summary.csv')
        )

    # Top 20 for each Group x Model combination
    unpivoted_df.loc[:, 'group'] = 'Other'
    for name, condition in groups_info.items():
        unpivoted_df.loc[unpivoted_df.set_index(['id', 'model']).index.isin(
            df[condition].set_index(['id', 'model']).index), 'group'] = name

    for model_name in unpivoted_df['model'].unique():
        for group_name in unpivoted_df['group'].unique():
            combo_df = unpivoted_df[(unpivoted_df['model'] == model_name) & (
                unpivoted_df['group'] == group_name)]
            if not combo_df.empty:
                combo_top_20 = combo_df['product'].value_counts().nlargest(
                    20).reset_index()
                combo_top_20.columns = ['product', 'count']
                dau.save_df_to_csv(
                    combo_top_20,
                    path.join(
                        analysis_output_csv,
                        f'{model_name}_{group_name}_top20_count_summary.csv'))

    # ==============================================================================
    # SECTION 2: TOP 20 FREQUENCY ANALYSIS (BOXPLOTS AND CSVS)
    # ==============================================================================
    print("\n--- Generating Top 20 Products by Frequency (Box Plots) ---")

    # Global analysis
    hc.plot_top_frequent_products(
        df, 'Global', analysis_output, num_products=20
    )

    # Per Model analysis
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        hc.plot_top_frequent_products(
            model_df, f'Model_{model_name}',
            analysis_output, num_products=20
        )

    # Per Group analysis
    for group_name, condition in groups_info.items():
        group_df = df[condition]
        hc.plot_top_frequent_products(
            group_df, group_name, analysis_output, num_products=20)

    # Per Model within each Group analysis
    for group_name, condition in groups_info.items():
        group_df = df[condition]
        for model_name in group_df['model'].unique():
            model_group_df = group_df[group_df['model'] == model_name]
            hc.plot_top_frequent_products(
                model_group_df, f'Model_{model_name}_in_{group_name}',
                analysis_output, num_products=20
            )

    # ==============================================================================
    # SECTION 3: PRODUCT COUNT CHARTS
    # ==============================================================================
    print("\n--- Generating Product Count Charts ---")

    # Generate charts for Global, Models, and Groups
    hc.generate_count_charts(df, 'Global', analysis_output_plots)
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        hc.generate_count_charts(
            model_df, f'Model_{model_name}', analysis_output_plots)

    for group_name, condition in groups_info.items():
        group_df = df[condition]
        hc.generate_count_charts(group_df, group_name, analysis_output_plots)

    # ==============================================================================
    # SECTION 4: PERFORMANCE METRICS
    # ==============================================================================
    print("\n--- Generating Performance Metrics ---")

    performance_metrics = []
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        y_true = model_df["label"]
        y_pred = model_df["prediction"]

        accuracy = accuracy_score(y_true, y_pred) or 0
        precision = precision_score(y_true, y_pred, pos_label="R") or 0
        recall = recall_score(y_true, y_pred, pos_label="R") or 0
        f1 = f1_score(y_true, y_pred, pos_label="R") or 0

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=["S", "R"]).ravel()

        performance_metrics.append({
            "Model": model_name,
            "True Positives (S/S)": tp,
            "True Negatives (R/R)": tn,
            "False Positives (S/R)": fp,
            "False Negatives (R/S)": fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    performance_df = pd.DataFrame(performance_metrics)
    dau.save_df_to_csv(
        performance_df,
        path.join(analysis_output_csv, "model_performance_summary.csv")
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    performance_df.set_index("Model")[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_title("Comparison of Model Performance Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    fig.tight_layout()
    dau.save_plot_to_png(
        fig, path.join(analysis_output_plots, "model_performance_metrics.png")
    )

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
