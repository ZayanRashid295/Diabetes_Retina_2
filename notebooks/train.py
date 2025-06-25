import os
import copy
import time
import argparse
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from tqdm import tqdm
from itertools import product
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import seaborn as sns


def plot_confusion_matrix(cm, class_names):
    """Plot and return a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=45)
    plt.tight_layout()
    return fig


def classifier_head(in_features, num_classes, dropout_rate):
    """Creates a classifier head with dropout and batch normalization."""
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, num_classes)
    )


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, model_name, save_dir, early_stop_patience):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0

    # For overlaying train/val metrics, accumulate per-epoch data.
    epoch_data = []  # Columns: Epoch, Train Loss, Val Loss, Train Accuracy, Val Accuracy

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_metrics = {}
        val_metrics = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            # Iterate over the data with TQDM progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
                  f"Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}")

            # Log metrics to wandb with model name in the keys
            wandb.log({
                f"{model_name}_{phase}_loss": epoch_loss,
                f"{model_name}_{phase}_accuracy": epoch_acc,
                f"{model_name}_{phase}_precision": epoch_precision,
                f"{model_name}_{phase}_recall": epoch_recall,
                f"{model_name}_{phase}_f1": epoch_f1,
                "epoch": epoch + 1
            })

            # Store metrics for overlay charts
            if phase == 'train':
                train_metrics = {"loss": epoch_loss, "acc": epoch_acc}
            else:
                val_metrics = {"loss": epoch_loss, "acc": epoch_acc}

            # If validation phase, check for early stopping and save best model
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # Save best model based on validation accuracy
                if epoch_acc > best_val_accuracy:
                    best_val_accuracy = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log confusion matrix and classification report
                    cm = confusion_matrix(all_labels, all_preds)
                    class_names = dataloaders['train'].dataset.classes
                    fig = plot_confusion_matrix(cm, class_names)
                    wandb.log({f"{model_name}_confusion_matrix": wandb.Image(fig)})
                    plt.close(fig)

                    # Create a table for the classification report instead of plotting its values as graph points.
                    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
                    report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-score", "Support"])
                    for key, values in report.items():
                        if isinstance(values, dict):
                            report_table.add_data(key, values.get("precision"), values.get("recall"), values.get("f1-score"), values.get("support"))
                        else:
                            # For accuracy (a single float), add a row with blanks for the other columns.
                            report_table.add_data(key, values, None, None, None)
                    wandb.log({f"{model_name}_classification_report_table": report_table})

                    # Save model checkpoint
                    os.makedirs(save_dir, exist_ok=True)
                    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best {model_name} model to {checkpoint_path}")

        # Append epoch-level train/val metrics for overlay charts
        if train_metrics and val_metrics:
            epoch_data.append([epoch + 1, train_metrics["loss"], val_metrics["loss"],
                               train_metrics["acc"], val_metrics["acc"]])
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # Log overlay charts if any epoch data was collected
    if epoch_data:
        epoch_table = wandb.Table(data=epoch_data, columns=["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])
        loss_chart = wandb.plot.line(
            epoch_table, "Epoch", ["Train Loss", "Val Loss"], title=f"{model_name} Loss (Train vs Val)"
        )
        acc_chart = wandb.plot.line(
            epoch_table, "Epoch", ["Train Accuracy", "Val Accuracy"], title=f"{model_name} Accuracy (Train vs Val)"
        )
        wandb.log({f"{model_name}_Loss_Chart": loss_chart, f"{model_name}_Accuracy_Chart": acc_chart})

    print(f"\nBest validation accuracy for {model_name}: {best_val_accuracy:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, criterion, device, model_name):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\nTest Metrics for {model_name} -- Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
          f"Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1: {test_f1:.4f}")

    # Instead of individual logs, return a metrics dict for table logging
    metrics = {
        "Model": model_name,
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1
    }
    return metrics, all_labels, all_preds


def test_ensemble(models, dataloader, criterion, device, ensemble_name):
    """
    Test ensemble performance by averaging logits from all models.
    Also logs the combined confusion matrix and classification report table.
    """
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Ensure all models are in evaluation mode
    for model in models.values():
        model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Ensemble Test"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_list = []
            for model in models.values():
                outputs = model(inputs)
                outputs_list.append(outputs)
            # Average logits from all models
            avg_outputs = sum(outputs_list) / len(outputs_list)
            loss = criterion(avg_outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(avg_outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    ensemble_loss = running_loss / len(dataloader.dataset)
    ensemble_acc = accuracy_score(all_labels, all_preds)
    ensemble_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    ensemble_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    ensemble_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\nEnsemble Test Metrics -- Loss: {ensemble_loss:.4f} Acc: {ensemble_acc:.4f} "
          f"Precision: {ensemble_precision:.4f} Recall: {ensemble_recall:.4f} F1: {ensemble_f1:.4f}")
    
    wandb.log({
        f"{ensemble_name}_test_loss": ensemble_loss,
        f"{ensemble_name}_test_accuracy": ensemble_acc,
        f"{ensemble_name}_test_precision": ensemble_precision,
        f"{ensemble_name}_test_recall": ensemble_recall,
        f"{ensemble_name}_test_f1": ensemble_f1
    })
    
    # Log combined confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig = plot_confusion_matrix(cm, dataloader.dataset.classes)
    wandb.log({f"{ensemble_name}_combined_confusion_matrix": wandb.Image(fig)})
    
    # Create and log a classification report table for the ensemble
    report = classification_report(all_labels, all_preds, target_names=dataloader.dataset.classes, output_dict=True, zero_division=0)
    report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-score", "Support"])
    for key, values in report.items():
        if isinstance(values, dict):
            report_table.add_data(key, values.get("precision"), values.get("recall"), values.get("f1-score"), values.get("support"))
        else:
            report_table.add_data(key, values, None, None, None)
    wandb.log({f"{ensemble_name}_classification_report_table": report_table})
    
    return all_labels, all_preds


def main(hyperparams=None):
    run_name = f"ensemble_run_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}" if hyperparams else "default_run"
    if hyperparams:
        wandb.init(project="ensemble-cnn", config=hyperparams, name=run_name)
    else:
        wandb.init(project="ensemble-cnn", config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 20,
            "optimizer": "adam",
            "dropout_rate": 0.5,
            "early_stop_patience": 4
        }, name=run_name)
    config = wandb.config

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define dataset directories (update these paths as necessary)
    data_dir = "./Diabetic_Balanced_Data"  # Should contain 'train', 'val', 'test' subfolders
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Data transforms: resize, normalize and (for train) augment.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets and dataloaders
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=4)
        for x in ['train', 'val', 'test']
    }

    num_classes = len(image_datasets['train'].classes)
    print("Detected classes:", image_datasets['train'].classes)
    save_dir = "./best_models"

    # Initialize models dictionary for the ensemble
    models_dict = {}

    # 1. EfficientNet-B4
    model_eff = models.efficientnet_b4(pretrained=True)
    in_features_eff = model_eff.classifier[1].in_features
    model_eff.classifier[1] = classifier_head(in_features_eff, num_classes, config.dropout_rate)
    model_eff = model_eff.to(device)
    models_dict["EfficientNetB4"] = model_eff

    # 2. DenseNet121
    model_dense = models.densenet121(pretrained=True)
    in_features_dense = model_dense.classifier.in_features
    model_dense.classifier = classifier_head(in_features_dense, num_classes, config.dropout_rate)
    model_dense = model_dense.to(device)
    models_dict["DenseNet121"] = model_dense

    # You can add more models here (e.g., ResNet50) for a larger ensemble.
    # For example:
    # model_resnet = models.resnet50(pretrained=True)
    # in_features_resnet = model_resnet.fc.in_features
    # model_resnet.fc = classifier_head(in_features_resnet, num_classes, config.dropout_rate)
    # model_resnet = model_resnet.to(device)
    # models_dict["ResNet50"] = model_resnet

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # To accumulate test metrics for table logging:
    test_metrics_rows = []

    # Loop through each model for training and testing individually
    for model_name, model in models_dict.items():
        print(f"\n--- Training {model_name} ---")
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Train the model (logs metrics, confusion matrix, classification report table and saves best checkpoint)
        model = train_model(model, dataloaders, criterion, optimizer, device,
                            config.num_epochs, model_name, save_dir, config.early_stop_patience)

        # Test the trained model (on TEST loader) and capture its test metrics in a dict
        print(f"\n--- Testing {model_name} on TEST set ---")
        metrics, _, _ = test_model(model, dataloaders['test'], criterion, device, model_name)
        test_metrics_rows.append([metrics["Model"], metrics["Test Loss"], metrics["Test Accuracy"],
                                  metrics["Test Precision"], metrics["Test Recall"], metrics["Test F1"]])

    # Log table for test metrics
    if test_metrics_rows:
        test_metrics_table = wandb.Table(data=test_metrics_rows, columns=["Model", "Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"])
        wandb.log({"Test Metrics Table": test_metrics_table})

    # Final evaluation: log confusion matrices on both VAL and TEST sets individually and for the ensemble

    # For individual models:
    for loader_key in ['val', 'test']:
        for model_name, model in models_dict.items():
            print(f"\nFinal evaluation for {model_name} on {loader_key.upper()} set:")
            metrics_ind, labels_ind, preds_ind = test_model(model, dataloaders[loader_key], criterion, device, model_name + "_" + loader_key.upper())
            cm_ind = confusion_matrix(labels_ind, preds_ind)
            fig_ind = plot_confusion_matrix(cm_ind, image_datasets['train'].classes)
            wandb.log({f"{model_name}_{loader_key.upper()}_confusion_matrix": wandb.Image(fig_ind)})

    # For ensemble:
    for loader_key in ['val', 'test']:
        print(f"\nFinal ensemble evaluation on {loader_key.upper()} set:")
        _, _ = test_ensemble(models_dict, dataloaders[loader_key], criterion, device, "Ensemble_" + loader_key.upper())

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_search", action="store_true", help="Run grid search over hyperparameters")
    args = parser.parse_args()

    if args.grid_search:
        # Define parameter grid for grid search (can be modified for randomized search)
        param_grid = {
            "learning_rate": [1e-4, 1e-3],
            "batch_size": [16],
            "dropout_rate": [0.3, 0.5],
            "num_epochs": [20],
            "early_stop_patience": [4]
        }
        keys, values = zip(*param_grid.items())
        # Loop over all combinations of hyperparameters
        for v in product(*values):
            hyperparams = dict(zip(keys, v))
            print("\nRunning experiment with hyperparameters:", hyperparams)
            main(hyperparams)
    else:
        # Default run with wandb's config (or defaults specified in main)
        main()
