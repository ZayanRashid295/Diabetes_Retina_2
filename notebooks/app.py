import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import wandb
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE='cpu'

hyperparams=None
run_name = f"ensemble_run_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}" if hyperparams else "default_run"
if hyperparams:
    wandb.init(project="ensemble-cnn", config=hyperparams, name=run_name)
else:
    wandb.init(project="ensemble-cnn",mode='offline', config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 10,
        "optimizer": "adam",
        "dropout_rate": 0.5,
        "early_stop_patience": 5
    }, name=run_name)
config = wandb.config

# Class labels
labels_dict = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}

# Image transforms
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classifier_head(in_features, num_classes, dropout_rate):
    """Creates a classifier head with dropout and batch normalization."""
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, num_classes)
    )

def generate_gradcam(image, model, model_name, pred_class):
    img_np = np.array(image)
    
    if model_name == "ViT":
        with torch.no_grad():
            model, processor = get_vit()
            inputs = processor(images=image, return_tensors="pt")
            img_tensor = inputs["pixel_values"].to(DEVICE)
            outputs = model(img_tensor, output_attentions=True)
            attentions = outputs.attentions[-1]
            cls_attention = attentions[:, :, 0, 1:].mean(dim=1).squeeze(0)
  
        num_patches = int(cls_attention.shape[-1] ** 0.5)
        heatmap = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap = np.uint8(255 * (heatmap / np.max(heatmap)))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed_img = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        
        return Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))

    elif model_name == "DenseNet":
        input_tensor = transform_gray(image.convert('L')).unsqueeze(0).to(DEVICE)
    else:
        input_tensor = transform_gray(image.convert('RGB')).unsqueeze(0).to(DEVICE)
        
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Perform inference to get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)  
    
    print(f"Predicted class: {pred_class}")
    targets = [ClassifierOutputTarget(pred_class)]  
    # target_layers = [model.features.denseblock4.denselayer32.conv1]

    target_layers = [model.features[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first image in batch
    
        # Overlay heatmap on the original image
        norm_img = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
        
        # Resize Grad-CAM heatmap to match the input image size
        grayscale_cam_resized = cv2.resize(grayscale_cam, (norm_img.shape[1], norm_img.shape[0]))

        # Overlay heatmap on the original image
        # print(norm_img.shape)
        print(grayscale_cam_resized.shape)
        visualization = show_cam_on_image(norm_img, grayscale_cam_resized, use_rgb=True)

        return visualization

# Load ViT model
def get_vit():
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./models/best_model_v1.pth', map_location=DEVICE))
    return model, processor

# Load DenseNet model
def get_densenet():
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./models/best_densenet201.pth', map_location=DEVICE))
    return model

# Load ensemble models
def get_ensemble():
    models_dict = {}
    model_eff = models.efficientnet_b4(pretrained=True)
    in_features_eff = model_eff.classifier[1].in_features
    num_classes=5
    model_eff.classifier[1] = classifier_head(in_features_eff, num_classes, config.dropout_rate)
    model_eff = model_eff.to(DEVICE)
    model_eff.load_state_dict(torch.load('./models/EfficientNetB4_best.pth', map_location=DEVICE, weights_only=True))
    models_dict["EfficientNetB4"] = model_eff

    model_dense = models.densenet121(pretrained=True)
    in_features_dense = model_dense.classifier.in_features
    model_dense.classifier = classifier_head(in_features_dense, num_classes, config.dropout_rate)
    model_dense = model_dense.to(DEVICE)
    model_dense.load_state_dict(torch.load('./models/DenseNet121_best.pth', map_location=DEVICE, weights_only=True))
    models_dict["DenseNet121"] = model_dense
    
    return models_dict


# Ensemble prediction
def get_ensemble_prediction(models_dict, image):
    image = image.convert('RGB')
    for model in models_dict.values():
        model.eval()

    probs = None
    with torch.no_grad():
        img_tensor = transform_rgb(image).unsqueeze(0).to(DEVICE)
        outputs_list = [model(img_tensor) for model in models_dict.values()]
        avg_outputs = sum(outputs_list) / len(outputs_list)
        probs = torch.softmax(avg_outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)

    prob_dict = {labels_dict[i]: float(probs[i]) for i in range(5)}
    print(prob_dict)
    return prob_dict

# Function to classify image
def classify_retina(image, model_name):
    if model_name == "ViT":
        model, processor = get_vit()
        inputs = processor(images=image, return_tensors="pt")
        img_tensor = inputs["pixel_values"].to(DEVICE)
        
    elif model_name == "DenseNet":
        model = get_densenet()
        image = image.convert("L")  # Convert to grayscale
        img_tensor = transform_gray(image).unsqueeze(0).to(DEVICE)
        
    else:  # Ensemble Model (EfficientNet + DenseNet121)
        models_dict = get_ensemble()
        prob_dict = get_ensemble_prediction(models_dict, image)
        
        probs = np.array(list(prob_dict.values()))
        pred_class = np.argmax(probs)
        print(pred_class)
        
        gradcam_images = []  # Store Grad-CAM images

        for name, model in models_dict.items():
    
            # Generate Grad-CAM
            gradcam_image = generate_gradcam(image, model, name, pred_class)
            gradcam_images.append(gradcam_image)

        return prob_dict, gradcam_images[0], gradcam_images[1]

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output.logits if model_name == "ViT" else output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    
    gradcam_img = generate_gradcam(image, model, model_name, pred_class)
    prob_dict = {labels_dict[i]: float(probs[i]) for i in range(5)}
    
    return prob_dict, gradcam_img, gradcam_img

# Gradio Interface
# iface = gr.Interface(
#     fn=classify_retina,
#     inputs=[
#         gr.Image(type="pil"),
#         gr.Radio(["ViT", "DenseNet", "Ensemble"], label="Select Model", value="ViT")
#     ],
#     outputs=[
#         gr.Label(num_top_classes=5),
#         gr.Image(type="pil"),
#         gr.Image(type="pil", visible=False)  # Second Grad-CAM image, hidden by default
#     ],
#     title="Retina Disease Classification with Explainability",
#     description="Upload a retina image and select a model (ViT, DenseNet, or Ensemble) to classify into one of 5 classes. Model explainability is provided using Grad-CAM visualization.",
# )

def update_ui(image, model_name):
    if model_name == "Ensemble":
        return gr.update(visible=True), gr.update(visible=True)  # Show both Grad-CAM images
    return gr.update(visible=True), gr.update(visible=False)  # Hide the second one

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Retina Image")
            model_selector = gr.Radio(["ViT", "DenseNet", "Ensemble"], label="Select Model", value="ViT")
            classify_btn = gr.Button("Classify")
            inference_speed = gr.Textbox(label="Inference Speed (ms)", interactive=False)
            prob_output = gr.Label(num_top_classes=5, label="Prediction Results")
        
        with gr.Column():
            gr.Markdown("## Grad-CAM Visualizations")
            gradcam1 = gr.Image(type="pil", label="Grad-CAM Output 1")
            gradcam2 = gr.Image(type="pil", label="Grad-CAM Output 2", visible=False)
    
    def classify_with_speed(image, model_name):
        start_time = time.time()
        results = classify_retina(image, model_name)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return (*results, f"{inference_time:.2f} ms")
    
    classify_btn.click(classify_with_speed, inputs=[image_input, model_selector], outputs=[prob_output, gradcam1, gradcam2, inference_speed])
    model_selector.change(update_ui, inputs=[image_input, model_selector], outputs=[gradcam1, gradcam2])

iface.launch(share=True)
