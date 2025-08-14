import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder 
import json

def normalize_hand_keypoints(df):
    normalized_rows = []

    for _, row in df.iterrows():
        x_coords = np.array(row.iloc[:21])
        y_coords = np.array(row.iloc[21:])

        keypoints = np.stack([x_coords, y_coords], axis=1)  # Shape (21, 2)

        # Step 1: Centralize (wrist at origin)
        origin = keypoints[0]  # (x0, y0)
        centralized = keypoints - origin

        # Step 2: Normalize scale (max distance from wrist)
        scale = np.linalg.norm(centralized, axis=1).max()
        if scale == 0:  # avoid division by zero
            scale = 1
        normalized = centralized / scale

        # Step 3: Flatten back to [x0..x20, y0..y20] format
        x_norm = normalized[:, 0]
        y_norm = normalized[:, 1]
        normalized_row = np.concatenate([x_norm, y_norm])
        normalized_rows.append(normalized_row)

    return pd.DataFrame(normalized_rows, columns=df.columns)

NUM_CLASSES = 5
df = pd.read_csv("StaticGestures/static_gestures_data.csv")

X = df.iloc[:,1:]
X = normalize_hand_keypoints(X)
print(X.head())
y = df.iloc[:, 0]

LE = LabelEncoder()
LE.fit(y)
y = LE.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 69, test_size = 0.2)

print(X_train.shape)
X_train = torch.tensor(X_train.to_numpy(), dtype = torch.float32)
X_test = torch.tensor(X_test.to_numpy(), dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32)


class StaticGesturesDetector(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StaticGesturesDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, X):
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
SGD = StaticGesturesDetector(X_train.size(1), NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(SGD.parameters(), lr = 0.001)
epochs = 30

def train_model(epochs):

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for i in range(epochs):
        SGD.train()
        train_loss, total, correct = 0,0,0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = SGD(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_losses.append(train_loss/len(train_loader))
        train_accuracies.append(correct/total)

        SGD.eval()
        test_loss, total, correct = 0,0,0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = SGD(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                _,predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        test_losses.append(test_loss/len(test_loader))
        test_accuracies.append(correct/total)

        print(f"Epoch({i+1}/{epochs}): Train Loss: {train_losses[-1]:.4f} | Train Accuracy: {train_accuracies[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Accuracy: {test_accuracies[-1]:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model():
    SGD.eval()

    predicted = []
    actual = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = SGD(X_batch)
            _, preds = torch.max(outputs, 1)
            predicted.extend(preds.detach().cpu().numpy())
            actual.extend(y_batch.detach().cpu().numpy())

    print("Classification Report:\n")
    print(classification_report(actual, predicted, target_names=LE.classes_))

    f1 = f1_score(actual, predicted, average='macro')
    print(f"Macro-average F1 Score: {f1:.4f}")

Gesture_index_relation = {}
for index, gesture in enumerate(LE.classes_):
    Gesture_index_relation[index] = gesture
with open("StaticGestures/gesture_index_relation.json", "w") as f:
    json.dump(Gesture_index_relation, f, indent=4)

train_model(epochs)
evaluate_model()
# torch.save(SGD.state_dict(), "StaticGestures/static_gesture_detector_model.pth")