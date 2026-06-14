from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
import uvicorn
import os

app = FastAPI(title="Document Forgery Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)
model = model.to(device)

MODEL_PATH = r"E:\Taftesh_AI\model.pth"
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}")

model.eval()

# ===== PREPROCESSING =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== ELA =====
def ela_analysis(image_bytes, quality=90):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        ela = np.abs(np.array(img).astype(int) - np.array(compressed).astype(int))
        max_val = ela.max()
        if max_val == 0:
            return 0.0
        return float(ela.mean() / max_val)
    except Exception as e:
        print(f"ELA error: {e}")
        return 0.5

# ===== ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "✅ Server running", "model": "EfficientNet Forgery Detector"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ELA
        ela_score = ela_analysis(image_bytes)

        # Model
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            forgery_prob = probs[0][0].item()
            auth_prob = probs[0][1].item()

        # Combined
        combined_score = (forgery_prob * 0.7) + (ela_score * 0.3)
        is_forged = combined_score > 0.35
        confidence = round(max(forgery_prob, auth_prob) * 100, 2)

        print(f"📊 أصلي: {auth_prob*100:.1f}% | مزور: {forgery_prob*100:.1f}% | ELA: {ela_score:.3f} | combined: {combined_score:.3f}")

        return {
            "is_forged": bool(is_forged),
            "confidence": confidence,
            "verdict": "مزورة ❌" if is_forged else "أصلية ✅",
            "details": {
                "model_score": round(forgery_prob * 100, 2),
                "ela_score": round(ela_score * 100, 2)
            }
        }

    except Exception as e:
        print(f"❌ خطأ: {e}")
        return {"error": str(e), "is_forged": False, "confidence": 0.0, "verdict": "خطأ"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)