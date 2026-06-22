import os
import io
import torch
import torch.nn as nn
import threading
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from classes import FOOD101_CLASSES

app = FastAPI(title="Food Recognition API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "model_cnn_project.pth"
NUM_CLASSES = 101
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ResNet18 model architecture for custom/local inference
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Try loading the saved PyTorch model weights
model_status = "demo_untrained"
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Successfully loaded model weights from: {MODEL_PATH}")
        model_status = "loaded"
    except Exception as e:
        print(f"Error loading model weights: {e}. Running in untrained/demo mode.")
else:
    print(f"Warning: {MODEL_PATH} not found. Running in untrained/demo mode. Run training via API to generate weights.")

model = model.to(DEVICE)
model.eval()

# Hugging Face Model pipeline - Lazy loaded to avoid slow API startup
hf_classifier = None
hf_loading_lock = threading.Lock()

def get_hf_classifier():
    global hf_classifier
    if hf_classifier is None:
        with hf_loading_lock:
            if hf_classifier is None:
                print("Loading Hugging Face model (nateraw/vit-base-food101)...")
                from transformers import pipeline
                # Use CPU/GPU based on PyTorch availability
                device_id = 0 if torch.cuda.is_available() else -1
                hf_classifier = pipeline(
                    "image-classification", 
                    model="nateraw/vit-base-food101", 
                    device=device_id
                )
                print("Hugging Face model loaded successfully!")
    return hf_classifier

# Image pre-processing transformations for local model
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training Data Augmentation
transformasi_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformasi_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom PyTorch Dataset for Hugging Face streaming data
class FoodDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom PyTorch Dataset for synthetic fallback data
class SyntheticFoodDataset(Dataset):
    def __init__(self, num_samples=64, num_classes=101):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random image tensor with shape (3, 224, 224) normalized
        image = torch.randn(3, 224, 224) * 0.5 + 0.5
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

# Thread-safe training states
training_state = {
    "status": "idle",  # "idle", "training", "completed", "failed", "cancelled"
    "epochs": 3,
    "current_epoch": 0,
    "current_loss": 0.0,
    "accuracy": 0.0,
    "message": "Model is ready to be trained.",
    "error": None
}
training_lock = threading.Lock()
cancel_training_event = threading.Event()

class TrainConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    lr: float = 0.005
    subset_train_size: int = 100
    subset_test_size: int = 20

def run_training_background(epochs: int, batch_size: int, lr: float, subset_train_size: int, subset_test_size: int):
    global model, model_status, training_state
    
    cancel_training_event.clear()
    
    with training_lock:
        training_state.update({
            "status": "training",
            "epochs": epochs,
            "current_epoch": 0,
            "current_loss": 0.0,
            "accuracy": 0.0,
            "message": "Initializing training dataset...",
            "error": None
        })
        
    try:
        use_synthetic = False
        try:
            from datasets import load_dataset
            # Use streaming=True to load dataset on-the-fly without downloading 5GB!
            dataset_mentah = load_dataset("food101", streaming=True)
            
            # Check for cancellation
            if cancel_training_event.is_set():
                raise InterruptedError("Training cancelled by user.")
                
            train_stream = dataset_mentah["train"]
            test_stream = dataset_mentah["validation"]
            
            # Materialize a small slice to lists
            train_list = list(train_stream.take(subset_train_size))
            test_list = list(test_stream.take(subset_test_size))
            
            train_dataset = FoodDataset(train_list, transform=transformasi_train)
            test_dataset = FoodDataset(test_list, transform=transformasi_test)
            
            with training_lock:
                training_state["message"] = "Successfully loaded Food101 dataset from Hugging Face!"
        except Exception as e:
            print(f"\n[WARNING] Failed to load Hugging Face dataset: {e}")
            print("Using synthetic fallback data...")
            use_synthetic = True
            train_dataset = SyntheticFoodDataset(num_samples=subset_train_size)
            test_dataset = SyntheticFoodDataset(num_samples=subset_test_size)
            
            with training_lock:
                training_state["message"] = "Using synthetic dataset fallback (offline mode)."

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Check for cancellation
        if cancel_training_event.is_set():
            raise InterruptedError("Training cancelled by user.")
            
        with training_lock:
            training_state["message"] = "Initializing model weights (Transfer Learning)..."
            
        # Download pretrained weights
        try:
            train_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}. Using random weights.")
            train_model = models.resnet18()
            
        # Freeze lower layers (hanya melatih classifier akhir)
        for param in train_model.parameters():
            param.requires_grad = False
            
        # Replace the final linear classification layer
        num_ftrs = train_model.fc.in_features
        train_model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        train_model = train_model.to(DEVICE)
        
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(train_model.fc.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            if cancel_training_event.is_set():
                raise InterruptedError("Training cancelled by user.")
                
            train_model.train()
            running_loss = 0.0
            
            with training_lock:
                training_state["current_epoch"] = epoch + 1
                training_state["message"] = f"Training Epoch {epoch + 1}/{epochs} in progress..."
                
            for images, labels in train_loader:
                if cancel_training_event.is_set():
                    raise InterruptedError("Training cancelled by user.")
                    
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = train_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(train_loader)
            with training_lock:
                training_state["current_loss"] = epoch_loss
                
        # Model Evaluation
        if cancel_training_event.is_set():
            raise InterruptedError("Training cancelled by user.")
            
        with training_lock:
            training_state["message"] = "Evaluating trained model accuracy..."
            
        train_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if cancel_training_event.is_set():
                    raise InterruptedError("Training cancelled by user.")
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = train_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total if total > 0 else 0.0
        
        # Check for cancellation before saving
        if cancel_training_event.is_set():
            raise InterruptedError("Training cancelled by user.")
            
        # Save model weights
        torch.save(train_model.state_dict(), MODEL_PATH)
        
        # Reload model weights into the active inference model
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        
        with training_lock:
            model_status = "loaded"
            training_state.update({
                "status": "completed",
                "accuracy": accuracy,
                "message": f"Training completed! Accuracy: {accuracy * 100:.2f}%. Model weights reloaded successfully."
            })
            
    except InterruptedError as ie:
        with training_lock:
            training_state.update({
                "status": "cancelled",
                "message": "Training was cancelled by the user."
            })
    except Exception as e:
        with training_lock:
            training_state.update({
                "status": "failed",
                "error": str(e),
                "message": f"Training failed: {str(e)}"
            })

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(DEVICE),
        "message": "Food Recognition API is running."
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("custom", description="Choose model: 'custom' (local ResNet18) or 'huggingface' (ViT model)")
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if model_type == "huggingface":
            # Call Hugging Face Pipeline
            classifier = get_hf_classifier()
            # The pipeline accepts PIL images directly!
            results = classifier(image)
            top_result = results[0]
            
            return {
                "prediction": top_result["label"],
                "confidence": top_result["score"],
                "model_status": "huggingface"
            }
            
        else:
            # Pre-process image for custom local ResNet18 model
            tensor_image = inference_transforms(image).unsqueeze(0).to(DEVICE)
            
            # Perform model inference
            with torch.no_grad():
                outputs = model(tensor_image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, dim=0)
                
            prediction_label = FOOD101_CLASSES[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return {
                "prediction": prediction_label,
                "confidence": confidence_score,
                "model_status": model_status
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/train")
def train_model(config: TrainConfig, background_tasks: BackgroundTasks):
    global training_state
    
    # Check if already training
    with training_lock:
        if training_state["status"] == "training":
            raise HTTPException(status_code=400, detail="Training is already in progress.")
            
    background_tasks.add_task(
        run_training_background,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        subset_train_size=config.subset_train_size,
        subset_test_size=config.subset_test_size
    )
    return {"message": "Training started in background.", "status": "training"}

@app.get("/train/status")
def get_training_status():
    global training_state
    with training_lock:
        return training_state

@app.post("/train/cancel")
def cancel_training():
    global training_state
    with training_lock:
        if training_state["status"] != "training":
            return {"message": "No active training to cancel."}
        
        cancel_training_event.set()
        training_state["status"] = "cancelled"
        training_state["message"] = "Cancellation requested..."
    return {"message": "Cancellation request submitted."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
