import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Define a classifier head (must match what was used during training)
def classifier_head(in_features, num_classes, dropout_rate):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, num_classes)
    )

# Function to load a model from a saved checkpoint
def load_model(model_name, num_classes, dropout_rate, device):
    if model_name == "EfficientNetB4":
        model = models.efficientnet_b4(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = classifier_head(in_features, num_classes, dropout_rate)
    elif model_name == "DenseNet121":
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = classifier_head(in_features, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = os.path.join("best_models", f"{model_name}_best.pth")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Function to get a dataloader and class names for a given dataset split
def get_dataloader(split, data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset.classes

# Function to perform ensemble predictions (averaging logits) on a dataloader
def ensemble_predict(models, dataloader, device):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Collect outputs from all models
            outputs_list = [model(inputs) for model in models]
            # Average the outputs (logits)
            avg_outputs = sum(outputs_list) / len(outputs_list)
            _, preds = torch.max(avg_outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


def save_classification_report_as_png(report, filename):
    """
    Converts a classification report (dict) into a pandas DataFrame,
    rounds numeric values to 2 decimals, replaces class keys 0–4 with custom labels,
    and renders it as a nicely formatted matplotlib table with a soft background.
    The figure is saved as a PNG.
    """
    # Mapping from numeric keys to desired class labels
    mapping = {
        "0": "No DR (Healthy)",
        "1": "Mild DR",
        "2": "Moderate DR",
        "3": "Severe DR",
        "4": "Proliferative DR"
    }
    
    # Create DataFrame from report and round to 2 decimals
    df = pd.DataFrame(report).transpose().round(2)
    # Replace row index labels for classes 0–4 if applicable
    df.index = [mapping.get(str(idx), idx) for idx in df.index]
    
    # Adjust figure height based on number of rows
    fig_height = len(df) * 0.5 + 2
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table from the DataFrame
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='center',
                     colColours=["#f0f0f0"] * df.shape[1],  # Soft background for headers
                     rowColours=["#f5f5f5"] + ["#fafafa"] * (len(df) - 1)  # First row slightly different
                     )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style each cell: make edges softer
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('lightgray')  # Soft edges
        cell.set_linewidth(0.5)  # Thin borders
        # Header row
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_fontsize(11)
            cell.set_text_props(weight='bold')
        # First data row
        elif row == 1:
            cell.set_facecolor("#fafafa")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  

    

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define dataset directory and hyperparameters
    data_dir = "./Diabetic_Balanced_Data"  # Assumes train/val/test subfolders exist here.
    dropout_rate = 0.5
    batch_size = 32

    # Load train loader to determine classes
    train_loader, classes = get_dataloader("train", data_dir, batch_size)
    num_classes = len(classes)
    print("Detected classes:", classes)

    # List of model names to load into the ensemble
    model_names = ["EfficientNetB4", "DenseNet121"]
    models_list = [load_model(mn, num_classes, dropout_rate, device) for mn in model_names]

    # Evaluate ensemble on each dataset split and print classification reports
    for split in ["train", "val", "test"]:
        print(f"\nEnsemble Classification Report on {split.upper()} set:")
        loader, _ = get_dataloader(split, data_dir, batch_size)
        labels, preds = ensemble_predict(models_list, loader, device)
        report = classification_report(labels, preds, target_names=classes, output_dict=True)
        save_classification_report_as_png(report, f"{split}_classification_report.png")
        print(report)



if __name__ == "__main__":
    main()
