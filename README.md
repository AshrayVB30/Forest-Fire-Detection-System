Forest Fire Detection System
---

# 🔥 Forest Fire Detection System

An AI-powered **Forest Fire Detection System** that uses **Deep Learning** (ResNet18) to classify images as **Fire** or **No Fire** and visualize the decision process with **Grad-CAM heatmaps**.

The system is built with:

* **Backend** → FastAPI (model inference + heatmap generation)
* **Frontend** → Streamlit (image upload, prediction, visualization)
* **Model** → ResNet18 (trained / ImageNet fallback)

---

## 🚀 Features

* ✅ Upload forest images and detect if fire is present.
* ✅ Real-time predictions using a **PyTorch ResNet18** model.
* ✅ Class probabilities (**Fire** vs **No Fire**).
* ✅ Modular backend–frontend architecture.

---

## 📂 Project Structure

```
forest-fire-detection/
│── backend/
│   ├── app.py                # FastAPI backend
│   ├── main.py
│   ├── train.py
│   ├── requirment.txt
│   └── __init__.py
│├───data
│   ├───CSV_data
│   └───Image_data
│       └───ForestFireDataset
│           └───train
│               ├───fire
│               └───nofire
│── frontend/
│   ├── app.py                # Streamlit frontend
│   ├── requirment.txt
│   └── __init__.py
│
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone git@github.com:AshrayVB30/Forest-Fire-Detection-System.git
cd forest-fire-detection
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start the backend (FastAPI)

```bash
cd forest-fire-detection
uvicorn app:app --reload
```

The backend will run at → `http://127.0.0.1:8000`

### 5️⃣ Start the frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

The frontend will run at → `http://localhost:8501`

---

## 🧠 Model Training

* The project uses **ResNet18**.
* You can either use the provided pre-trained model (`fire_detection_resnet18.pth`) or ImageNet fallback.
* Training script (to be added soon) can be used for fine-tuning with a forest fire dataset.

---
## Workflow Diagram

![img.png](img.png)

---

## 📦 Requirements

See `requirements.txt`:

* fastapi
* uvicorn
* streamlit
* torch
* torchvision
* pillow
* matplotlib
* numpy
* opencv-python
* streamlit
* requests

---

## Sample Output

![Output-1](images/local-image.png)

![Output-2](images/img.png)

---