import io
import os
import base64
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Forest Fire Detection API", version="1.2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "fire_detection_resnet18.pth")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"⚠️ Error loading saved model: {e}, using ImageNet weights")
    else:
        logger.warning(f"⚠️ Model not found at {model_path}, using ImageNet weights")

    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
labels = ["Fire", "No Fire"]

# Grad-CAM storage
gradients = None
activations = None

def save_gradient(grad):
    global gradients
    gradients = grad
    logger.info("✅ Gradient captured")

def forward_hook(module, input, output):
    global activations
    activations = output
    logger.info("✅ Activations captured")

# Attach hooks to last conv block of ResNet18
target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(lambda m, g_in, g_out: save_gradient(g_out[0]))

def generate_heatmap(img_tensor, class_idx):
    global gradients, activations
    try:
        gradients = None
        activations = None

        output = model(img_tensor)
        model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        if gradients is None or activations is None:
            logger.error("❌ Gradients or activations were not captured")
            return None

        grads = gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((activations * grads).sum(dim=1)).squeeze().detach().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize((224, 224), Image.BILINEAR)

        # Overlay
        heatmap = np.array(cam)
        plt.figure(figsize=(4, 4))
        plt.imshow(transforms.ToPILImage()(img_tensor.squeeze(0)))
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)

        logger.info("✅ Heatmap generated")
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.error(f"❌ Error generating heatmap: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = outputs.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
        label = labels[predicted_class]

        # Generate heatmap
        heatmap_base64 = generate_heatmap(img_tensor, predicted_class)

        response = {
            "prediction": label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                "Fire": round(probabilities[0][0].item(), 4),
                "No Fire": round(probabilities[0][1].item(), 4),
            }
        }

        if heatmap_base64:
            response["heatmap"] = heatmap_base64
        else:
            response["heatmap"] = None
            logger.warning("⚠️ Heatmap is None")

        return response

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
