import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from tqdm import tqdm
from PIL import Image
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
data_dir = "/kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

# Magnification (only using 100X)
magnification = "100X"

# Image size for models
image_sizes = {
    "swin_b": 224,
}

class BreaKHisDataset(Dataset):
    def __init__(self, root_dir, magnification, transform=None): 
        self.root_dir = root_dir
        self.magnification = magnification
        self.transform = transform
        self.classes = ['benign', 'malignant']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)

            for root, _, files in os.walk(class_dir):
                if os.path.basename(root) == self.magnification:
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            path = os.path.join(root, file)
                            samples.append((path, self.class_to_idx[class_name], self.magnification))
        return samples

    def __len__(self):  # Fixed __len__
        return len(self.samples)

    def __getitem__(self, idx):  # Fixed __getitem__
        path, label, magnification = self.samples[idx]
        image = Image.open(path).convert('RGB')

        # Convert to OpenCV format and apply CLAHE
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image) #morphological operations--try

        return image, label, magnification  # Return magnification as part of the data


# Function to get transformations
def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

# Model definition
def create_swin_model(name):
    model = timm.create_model(name, pretrained=True, num_classes=2)
    model.head = nn.Sequential(
        nn.Dropout(0.7), #prove 0.7 droput
        model.head
    )
    return model

models = {
    "swin_b": create_swin_model("swin_base_patch4_window7_224"),
}

# Initialize dataset for 40X magnification and model
datasets = {}
for model_name in models:
    datasets[model_name] = BreaKHisDataset(data_dir, magnification, transform=get_transform(image_sizes[model_name]))

# Combine all values (path, label, magnification) for stratification
all_paths = []
all_labels = []
all_magnifications = []
for model_name in models:
    for path, label, magnification in datasets[model_name].samples:
        all_paths.append(path)
        all_labels.append(label)
        all_magnifications.append(magnification)

# K-Fold settings
k_folds = 5
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Metrics tracking
all_preds = []
all_labels_list = []
all_probs = []
train_losses = []  # List to track training losses
val_losses = []    # List to track validation losses

# Training loop with 30 epochs per fold
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.to(device)
    
    # Optimizer settings with learning rate 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=5e-3)  # Further reduced learning rate to 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    # K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(all_labels)), all_labels)):
        print(f"\nFold {fold + 1} / {k_folds}")

        # Load dataset for 40X magnification only
        train_dataset = Subset(datasets[model_name], [i for i in train_idx if i < len(datasets[model_name])])
        val_dataset = Subset(datasets[model_name], [i for i in val_idx if i < len(datasets[model_name])])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

        # Training loop
        for epoch in range(30):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for images, labels, magnifications in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            scheduler.step()
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)  # Append the training loss for plotting

            # Validation loop
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_val_preds, all_val_labels, all_val_probs = [], [], []

            with torch.no_grad():
                for images, labels, magnifications in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of malignant class
                    preds = torch.argmax(outputs, dim=1)

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_probs.extend(probs.cpu().numpy())

            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)  # Append the validation loss for plotting

            precision = precision_score(all_val_labels, all_val_preds)
            recall = recall_score(all_val_labels, all_val_preds)
            f1 = f1_score(all_val_labels, all_val_preds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Precision: {precision:.4f} | "
                  f"Recall: {recall:.4f} | "
                  f"F1 Score: {f1:.4f}")

            # Store for confusion matrix and ROC curve
            all_preds.extend(all_val_preds)
            all_labels_list.extend(all_val_labels)
            all_probs.extend(all_val_probs)

# After all folds are done, plot loss graphs
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print confusion matrix and ROC curve
cm = confusion_matrix(all_labels_list, all_preds)
print(f"Confusion Matrix:\n{cm}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels_list, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {model_name} - 100X Magnification")
plt.legend()
plt.show()

print("Training complete!")
