from os import path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.utils.data_analysis_utils as dau


def generate_heatmap(df_subset: pd.DataFrame, group_name: str,
                     output_dir: str):
    """
    Generates and saves a heatmap showing the frequency of products per model.

    This function transforms a subset of a DataFrame from wide to long format
    for columns starting with 'product', counts the occurrences of each product
    per model, and visualizes the result as a heatmap. The heatmap is saved as
    a PNG file in the specified directory.

    Args:
        df_subset (pd.DataFrame): Input DataFrame containing at least one
            'product' column and a 'model' column.
        group_name (str): Name of the group used for the heatmap title and
            filename.
        output_dir (str): Directory path where the heatmap PNG file will be
        saved.

    Returns:
        None: The function saves the heatmap as a PNG file. If no product
              columns are found or the DataFrame is empty, the function exits
              without saving.
    """
    product_cols = [
        col for col in df_subset.columns if col.startswith("product")]
    if not product_cols or df_subset.empty:
        print(f"Skipping heatmap for {group_name}: No product columns "
              "or empty data.")
        return

    df_long = df_subset.melt(
        id_vars=["model"], value_vars=product_cols,
        var_name="product_col", value_name="product"
    ).dropna(subset=["product"])

    count_data = df_long.groupby(
        ["model", "product"]).size().reset_index(name="count")
    heatmap_data = count_data.pivot(
        index="model", columns="product", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(50, 12))
    sns.heatmap(heatmap_data, cmap="inferno_r",
                linewidths=.5, linecolor="white", ax=ax)
    ax.set_title(f"Heatmap {group_name}", fontsize=18)
    ax.set_xlabel("Products")
    ax.set_ylabel("Model")
    fig.tight_layout()

    png_path = path.join(output_dir, f"heatmap_{group_name}.png")
    dau.save_plot_to_png(fig, png_path)


def plot_top_frequent_products(df_subset: pd.DataFrame, name: str,
                               output_dir: str, num_products: int = 20):
    """
    Analyzes the top N most frequent products, saves their weight distribution,
    and generates a box plot of weights.

    This function identifies the top N most frequent products in the input
    DataFrame, extracts their associated weight data, saves this data as a CSV,
    and creates a box plot visualizing the distribution of weights per product.
    The CSV and the plot are saved in the specified directory.

    Args:
        df_subset (pd.DataFrame): Input DataFrame containing product columns
            ('product1' to 'product5') and corresponding weight columns
            ('weight1' to 'weight5').
        name (str): Name of the group or dataset, used in file names and plot
        title.
        output_dir (str): Directory where the CSV and box plot PNG will be
        saved.
        num_products (int, optional): Number of top products to analyze.
        Defaults to 20.

    Returns:
        None: The function saves a CSV and a PNG box plot file. If no top
        products are found, it prints a message and exits.
    """
    top_products_data, top_products_list = dau.get_top_products_and_weights_df(
        df_subset, num_products=num_products)

    if top_products_data.empty:
        print(f"No top frequent products found for {name}.")
        return

    # Save CSV (includes weight data for the top N most frequent products)
    top_products_table_path = path.join(
        output_dir, "csv",
        f"top{num_products}_frequent_products_weights_{name}.csv")
    dau.save_df_to_csv(
        top_products_data,
        top_products_table_path
    )

    # Generate and save box plot of Attention Coefficients
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=top_products_data, x='weight', y='product',
                order=top_products_list, palette='Spectral', ax=ax)
    ax.set_title(f'Attention Coeff. Distribution for Top {num_products} '
                 'Frequent Products for {name}')
    ax.set_xlabel('Attention Coefficient (Weight)')
    ax.set_ylabel('Product')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    png_path = path.join(
        output_dir, "plots",
        f"top{num_products}_frequent_products_boxplot_{name}.png")
    dau.save_plot_to_png(
        fig,
        png_path
    )


def generate_count_charts(data_subset: pd.DataFrame, name: str,
                          output_dir: str):
    """
    Generates and saves horizontal bar charts of product counts.

    This function computes the frequency of each product in the input DataFrame
    using `get_top_products_and_counts`, then creates a horizontal bar chart
    showing the counts of each product. The chart is saved as a PNG file in the
    specified output directory. If no product count data is available, the
    function prints a message and exits without generating a chart.

    Args:
        data_subset (pd.DataFrame): Input DataFrame containing product columns
            ('product1' to 'product5').
        name (str): Name of the group or dataset, used in the chart title and
            output file name.
        output_dir (str): Directory where the chart PNG will be saved.

    Returns:
        None: The function saves a PNG file. If no product data is found, it
        prints a message and exits.
    """
    product_counts = dau.get_top_products_and_counts(data_subset)
    if not product_counts:
        print(f"No product count data to plot for {name}.")
        return

    products = list(product_counts.keys())
    counts = list(product_counts.values())

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(products, counts, color='darkblue')
    ax.set_title(f'Product Counts for {name}')
    ax.set_xlabel('Count')
    ax.set_ylabel('Product')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    png_path = path.join(output_dir, f"product_counts_{name}.png")
    dau.save_plot_to_png(fig, png_path)
