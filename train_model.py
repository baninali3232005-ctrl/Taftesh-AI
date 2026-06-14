import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from PIL import Image

authentic_path = r"E:\Taftesh_AI\archive\CASIA2\Au"
tampered_path  = r"E:\Taftesh_AI\archive\CASIA2\Tp"
model_save     = r"E:\Taftesh_AI\model.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class FolderDataset(Dataset):
    def __init__(self, folder, label):
        self.label = label
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return transform(img), self.label

print("تحميل البيانات...")
auth_data = FolderDataset(authentic_path, 0)
tamp_data = FolderDataset(tampered_path, 1)
dataset   = ConcatDataset([auth_data, tamp_data])

# تقسيم train/val
val_size   = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False)

print(f"تدريب: {train_size} | تحقق: {val_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"الجهاز: {device}")

# ✅ pretrained=True هذا هو الفرق المهم!
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(1280, 2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_acc = 0
for epoch in range(15):
    # Training
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
    train_acc = correct / train_size * 100

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            val_correct += (output.argmax(1) == labels).sum().item()
    val_acc = val_correct / val_size * 100

    print(f"Epoch {epoch+1}/15 - Loss: {total_loss:.3f} - Train: {train_acc:.1f}% - Val: {val_acc:.1f}%")
    scheduler.step()

    # احفظ أفضل موديل
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save)
        print(f"✅ حفظ أفضل موديل: {val_acc:.1f}%")

print(f"🎉 انتهى التدريب! أفضل دقة: {best_acc:.1f}%")