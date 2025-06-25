import matplotlib.pyplot as plt
import pandas as pd

def save_model_comparison_table_as_png(filename):
    """
    Creates and saves a comparison table for ViT, DenseNet, and Ensemble models.
    Bolds:
    - Ensemble's Accuracy, F1 Score, Precision, Recall
    - DenseNet's Inference Speed (2500 ms)
    """
    # Define model performance metrics
    data = {
        "Accuracy": [0.742, 0.755, 0.828],
        "F1 Score": [0.742, 0.753, 0.829],
        "Precision": [0.745, 0.752, 0.830],
        "Recall": [0.742, 0.755, 0.828],
        "Inference Speed (ms)": [3900, 2500, 2900]  # Lower is better
    }

    model_names = ["ViT", "DenseNet", "Ensemble"]

    # Convert to DataFrame and round values
    df = pd.DataFrame(data, index=model_names).round(3)

    # Adjust figure size based on number of rows
    fig_height = len(df) * 0.5 + 2
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=model_names,
                     loc='center',
                     cellLoc='center',
                     colColours=["#f0f0f0"] * len(df.columns),
                     rowColours=["#f5f5f5"] * len(df))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Apply cell styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('lightgray')
        cell.set_linewidth(0.5)

        # Header row
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_fontsize(11)
            cell.set_text_props(weight='bold')

        # Bold Ensemble's Accuracy, F1 Score, Precision, Recall
        if row == 3 and 0 <= col <= 3:  # Row 3 = Ensemble, Cols 1-4 = Metrics
            cell.set_text_props(weight='bold')

        # Bold DenseNet's inference speed (2500 ms)
        if row == 2 and col == 4:  # Row 2 = DenseNet, Col 5 = Inference
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Call function to save the table
save_model_comparison_table_as_png("model_comparison_fixed.png")
