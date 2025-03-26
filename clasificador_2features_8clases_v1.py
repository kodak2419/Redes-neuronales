import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# --- Cargar datos ---
# features_raw = pd.read_csv('features_400_2features_8classes.csv', header=None, sep=r'\s+').values
# labels_raw = pd.read_csv('labels_400_2features_8classes.csv', header=None, sep=r'\s+').values
# labels_raw = np.argmax(labels_raw, axis=1)


features_raw = pd.read_csv('features_400_2features_8classes_overlap.csv', header=None, sep=r'\s+').values
labels_raw = pd.read_csv('labels_400_2features_8classes_overlap.csv', header=None, sep=r'\s+').values
labels_raw = np.argmax(labels_raw, axis=1)

# --- Visualización inicial de los datos ---
plt.figure(figsize=(10, 6))
plt.xlim(features_raw[:, 0].min(), features_raw[:, 0].max())
plt.ylim(features_raw[:, 1].min(), features_raw[:, 1].max())
plt.scatter(features_raw[:, 0], features_raw[:, 1], c=labels_raw, cmap='tab10', edgecolor='k', s=20)
plt.title('Distribución original de clases')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.colorbar(ticks=range(8), label="Clase")
#plt.colorbar(label="Clase")
plt.show()

# --- Normalización ---
scaler = StandardScaler()
features = scaler.fit_transform(features_raw)

# --- Separación 80/20 para entrenamiento y prueba ---
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(features, labels_raw, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.long)

batch_size = 128
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # --- Red neuronal ---
# class NeuralNet(nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=150, output_dim=8):
#         super(NeuralNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
# #         )

# --- Red neuronal ---
class NeuralNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=8):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, output_dim)
        )


    def forward(self, x):
        return self.net(x)

# --- Inicialización ---
device = torch.device("cpu")
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

train_losses = []
print("\n--- Entrenamiento de la red neuronal ---")

# --- Entrenamiento ---
epochs = 200
for epoch in range(epochs):
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, axis=1).cpu().numpy()
        labels_batch = batch_y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_batch)

    scheduler.step()
    train_losses.append(running_loss / len(train_loader))
    
    epoch_accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Época {epoch+1}/{epochs} - Pérdida: {running_loss/len(train_loader):.4f} - Accuracy: {epoch_accuracy:.2f}%")    

# --- Evaluación ---
model.eval()
with torch.no_grad():
    outputs_test = model(X_test)
    preds_test = torch.argmax(outputs_test, axis=1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
# --- Evaluación en conjunto de entrenamiento ---
model.eval()
with torch.no_grad():
    outputs_train = model(X_train)
    preds_train = torch.argmax(outputs_train, axis=1).cpu().numpy()
    y_train_true = y_train.cpu().numpy()  
    
print("\nAccuracy en conjunto de entrenamiento: {:.2f}%".format(accuracy_score(y_train_true, preds_train) * 100))
print("\nAccuracy en conjunto de prueba: {:.2f}%".format (accuracy_score(y_true, preds_test) * 100))
#print("\nAccuracy en conjunto de prueba:", accuracy_score(y_true, preds_test))
print("\nReporte de clasificación en prueba:\n")
print(classification_report(y_true, preds_test, digits=4))

# --- Gráfica de pérdida ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Pérdida (Loss)")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Matriz de confusión ---
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, preds_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, cmap="Blues",
            xticklabels=range(8), yticklabels=range(8))
plt.xlabel("Predicción")
plt.ylabel("Clase Real")
plt.title("Matriz de Confusión Normalizada (Prueba)")
plt.show()

# --- Frontera de decisión sobre TODOS los datos ---
x_min, x_max = features_raw[:,0].min(), features_raw[:,0].max()
y_min, y_max = features_raw[:,1].min(), features_raw[:,1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    preds_grid = model(grid_tensor)
    preds_grid = torch.argmax(preds_grid, axis=1).cpu().numpy()
    
Z = preds_grid.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=8, alpha=0.5, cmap='tab10')
plt.contour(xx, yy, Z, levels=np.arange(-0.5, 8, 1), colors='k', linewidths=1)
plt.scatter(features_raw[:,0], features_raw[:,1], c=labels_raw, cmap='tab10', edgecolor='k', s=20)
plt.title("Frontera de decisión - Todos los datos")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(ticks=range(8), label="Clase")
plt.grid(True)
plt.show()

