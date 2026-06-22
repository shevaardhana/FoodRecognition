import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Ensure datasets and tqdm are installed
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Missing required packages. Please install datasets and tqdm:")
    print("pip install datasets tqdm")
    sys.exit(1)

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

# Data Augmentation & Normalization
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

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================
print("Mencoba memuat dataset Food101 secara streaming dari Hugging Face...")
use_synthetic = False

try:
    # Use streaming=True to load dataset on-the-fly without downloading 5GB!
    dataset_mentah = load_dataset("food101", streaming=True)
    
    print("Mengambil subset data dari streaming (100 training, 20 test)...")
    train_stream = dataset_mentah["train"]
    test_stream = dataset_mentah["validation"]
    
    # Materialize a small slice to lists
    train_list = list(train_stream.take(100))
    test_list = list(test_stream.take(20))
    
    # Create dataset objects
    train_dataset = FoodDataset(train_list, transform=transformasi_train)
    test_dataset = FoodDataset(test_list, transform=transformasi_test)
    
    print("Berhasil memuat dataset Food101 secara streaming!")
except Exception as e:
    print(f"\n[PERINGATAN] Gagal memuat dataset dari Hugging Face: {e}")
    print("Menggunakan data sintetis sebagai fallback agar proses training dapat berjalan offline...")
    use_synthetic = True
    train_dataset = SyntheticFoodDataset(num_samples=64)
    test_dataset = SyntheticFoodDataset(num_samples=16)

print(f"Jumlah data latihan (train_dataset): {len(train_dataset)}")
print(f"Jumlah data pengujian (test_dataset): {len(test_dataset)}")

# ==========================================
# 2. DATALOADER AND MODEL CONFIGURATION
# ==========================================
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = 101

# Fine-tuning ResNet18
print("\nKonfigurasi model ResNet18 (Transfer Learning)...")
# Download pretrained weights
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception as e:
    print(f"Gagal mendownload bobot pretrained ResNet18: {e}. Menggunakan bobot acak.")
    model = models.resnet18()

# Freeze lower layers (hanya melatih classifier akhir)
for param in model.parameters():
    param.requires_grad = False

# Ganti layer klasifikasi terakhir
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Menggunakan perangkat: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.005)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
model.train()
epochs = 3
print(f"\nMemulai Pelatihan Model ({epochs} Epoch)...")
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Loss Epoch {epoch+1}: {running_loss/len(train_loader):.4f}")

# ==========================================
# 4. MODEL EVALUATION
# ==========================================
model.eval()
correct = 0
total = 0
print("\nMemulai Evaluasi Akurasi...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"\nHasil Akhir -> Skor Accuracy Anda: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Save model weights
torch.save(model.state_dict(), 'model_cnn_project.pth')
print("\nModel kustom berhasil disimpan sebagai 'model_cnn_project.pth'!")
