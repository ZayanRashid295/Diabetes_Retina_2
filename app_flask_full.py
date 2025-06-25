from flask import Flask, request, render_template_string, send_file
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import io
import os

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

labels_dict = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}

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
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, num_classes)
    )

def get_vit():
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./models/best_model_v1.pth', map_location=DEVICE))
    model.eval()
    return model, processor

def get_densenet():
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('./models/best_densenet201.pth', map_location=DEVICE))
    model.eval()
    return model

def get_ensemble():
    models_dict = {}
    model_eff = models.efficientnet_b4(pretrained=True)
    in_features_eff = model_eff.classifier[1].in_features
    num_classes = 5
    model_eff.classifier[1] = classifier_head(in_features_eff, num_classes, 0.5)
    model_eff = model_eff.to(DEVICE)
    model_eff.load_state_dict(torch.load('./models/EfficientNetB4_best.pth', map_location=DEVICE))
    model_eff.eval()
    models_dict["EfficientNetB4"] = model_eff

    model_dense = models.densenet121(pretrained=True)
    in_features_dense = model_dense.classifier.in_features
    model_dense.classifier = classifier_head(in_features_dense, num_classes, 0.5)
    model_dense = model_dense.to(DEVICE)
    model_dense.load_state_dict(torch.load('./models/DenseNet121_best.pth', map_location=DEVICE))
    model_dense.eval()
    models_dict["DenseNet121"] = model_dense
    return models_dict

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
    targets = [ClassifierOutputTarget(pred_class)]
    target_layers = [model.features[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        norm_img = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
        grayscale_cam_resized = cv2.resize(grayscale_cam, (norm_img.shape[1], norm_img.shape[0]))
        visualization = show_cam_on_image(norm_img, grayscale_cam_resized, use_rgb=True)
        return Image.fromarray(visualization)

def classify_retina(image, model_name):
    if model_name == "ViT":
        model, processor = get_vit()
        inputs = processor(images=image, return_tensors="pt")
        img_tensor = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output.logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
        gradcam_img = generate_gradcam(image, model, model_name, pred_class)
        prob_dict = {labels_dict[i]: float(probs[i]) for i in range(5)}
        return prob_dict, gradcam_img, gradcam_img
    elif model_name == "DenseNet":
        model = get_densenet()
        image_gray = image.convert("L")
        img_tensor = transform_gray(image_gray).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
        gradcam_img = generate_gradcam(image, model, model_name, pred_class)
        prob_dict = {labels_dict[i]: float(probs[i]) for i in range(5)}
        return prob_dict, gradcam_img, gradcam_img
    else:
        models_dict = get_ensemble()
        image_rgb = image.convert('RGB')
        with torch.no_grad():
            img_tensor = transform_rgb(image_rgb).unsqueeze(0).to(DEVICE)
            outputs_list = [m(img_tensor) for m in models_dict.values()]
            avg_outputs = sum(outputs_list) / len(outputs_list)
            probs = torch.softmax(avg_outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
        prob_dict = {labels_dict[i]: float(probs[i]) for i in range(5)}
        gradcam_images = []
        for name, model in models_dict.items():
            gradcam_image = generate_gradcam(image, model, name, pred_class)
            gradcam_images.append(gradcam_image)
        return prob_dict, gradcam_images[0], gradcam_images[1]

HTML_FORM = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retina Disease Classifier</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f8f9fa; }
    .container { max-width: 700px; margin-top: 40px; }
    .footer { margin-top: 40px; color: #888; font-size: 0.9em; text-align: center; }
    .navbar-brand { font-weight: bold; }
    .result-box { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #0001; padding: 24px; margin-top: 24px; }
    .gradcam-img { max-width: 100%; border-radius: 8px; margin: 10px 0; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-primary">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">Retina Disease Classifier</a>
  </div>
</nav>
<div class="container">
  <h2 class="mt-4">Upload Retina Image</h2>
  <form method=post enctype=multipart/form-data class="mb-4">
    <div class="mb-3">
      <input type=file name=image class="form-control" required>
    </div>
    <div class="mb-3">
      <label class="form-label">Select Model:</label>
      <select name=model_name class="form-select">
        <option value="ViT">ViT</option>
        <option value="DenseNet">DenseNet</option>
        <option value="Ensemble">Ensemble</option>
      </select>
    </div>
    <button type=submit class="btn btn-primary">Classify</button>
  </form>
  {% if result %}
  <div class="result-box">
    <h4>Prediction Results</h4>
    <pre>{{ result }}</pre>
    <h5 class="text-success">Best Prediction: {{ best_class }}</h5>
    <h5>Grad-CAM Visualizations</h5>
    <img src="/gradcam1.png" class="gradcam-img">
    {% if gradcam2 %}
      <img src="/gradcam2.png" class="gradcam-img">
    {% endif %}
  </div>
  {% endif %}
</div>
<div class="footer">
  &copy; {{ 2024 }} Retina Disease Classifier | Powered by Flask, PyTorch, and Bootstrap
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    best_class = None
    gradcam1 = None
    gradcam2 = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result = 'No image uploaded.'
        else:
            file = request.files['image']
            if file.filename == '':
                result = 'No image selected.'
            else:
                image = Image.open(file.stream).convert('RGB')
                model_name = request.form.get('model_name', 'ViT')
                prob_dict, gradcam_img1, gradcam_img2 = classify_retina(image, model_name)
                result = prob_dict
                best_class = max(prob_dict, key=prob_dict.get)
                gradcam1 = gradcam_img1
                gradcam2 = gradcam_img2 if model_name == 'Ensemble' else None
                gradcam_img1.save('gradcam1.png')
                if gradcam2:
                    gradcam2.save('gradcam2.png')
    return render_template_string(HTML_FORM, result=result, best_class=best_class, gradcam2=gradcam2 is not None)

@app.route('/gradcam1.png')
def gradcam1_img():
    return send_file('gradcam1.png', mimetype='image/png')

@app.route('/gradcam2.png')
def gradcam2_img():
    return send_file('gradcam2.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
