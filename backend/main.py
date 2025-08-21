import io
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
import numpy as np
import cv2
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Forest Fire Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model setup
def load_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model_path = os.path.join(os.path.dirname(__file__), "models", "fire_detection_resnet18.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"⚠️ Error loading model, using base weights: {e}")
    else:
        logger.warning(f"⚠️ Model not found at {model_path}, using ImageNet weights")

    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = ["Fire", "No Fire"]

# ---------------------------
# 🔥 Grad-CAM Implementation
# ---------------------------
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # ✅ Hook into the last convolutional block
    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    loss = output[:, target_class].sum()

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get Grad-CAM weights
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)  # GAP over H, W
    activation = activations[0]

    cam = (activation * grads).sum(dim=1).squeeze()
    cam = torch.relu(cam)

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    forward_handle.remove()
    backward_handle.remove()

    logger.info(f"🔥 Grad-CAM generated: shape={cam.shape}, min={cam.min()}, max={cam.max()}")
    return cam


def overlay_heatmap(img: Image.Image, cam: np.ndarray):
    img = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file or not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
            label = labels[predicted_class]

        # 🔥 Generate Grad-CAM heatmap
        cam = generate_gradcam(model, img_tensor, predicted_class)
        overlay = overlay_heatmap(img, cam)

        # Encode overlay as base64
        _, buffer = cv2.imencode(".png", overlay)
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
        logger.info(f"✅ Heatmap encoded, length={len(heatmap_base64)}")

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                "Fire": round(probabilities[0][0].item(), 4),
                "No Fire": round(probabilities[0][1].item(), 4)
            },
            "heatmap": heatmap_base64
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
