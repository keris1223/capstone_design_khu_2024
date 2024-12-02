import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and preprocess the data
data = pd.read_csv('output_data3_preprocessed.csv')  # 데이터를 로드합니다.

# Convert 'event_time' to datetime format and sort by it
data['event_time'] = pd.to_datetime(data['event_time'])
data = data.sort_values('event_time').reset_index(drop=True)

# Select relevant features and normalize them
features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_memory',
            'random_sample_usage_cpus', 'assigned_memory', 'page_cache_memory']
target = 'Failed'

# Normalize features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 2. Create sequences for LSTM input
sequence_length = 10  # Example sequence length

def create_sequences(df, features, target, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i + sequence_length].values)
        y.append(df[target].iloc[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data, features, target, sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to device
X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# 3. Define Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # Using the last hidden state
        return self.sigmoid(out)

# Model parameters
input_size = len(features)
hidden_size = 50
output_size = 1  # Binary classification

model = LSTMModel(input_size, hidden_size, output_size).to(device)  # Move model to device

# 5. Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Train the model and record training history
epochs = 20
train_losses = []
test_losses = []

from tqdm import tqdm

# 학습 과정
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"Epoch [{epoch+1}/{epochs}]")

    # tqdm을 사용하여 batch 진행 상황을 표시
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # 매 epoch 후 테스트 셋에서 평가
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing", leave=False):
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


# 7. Plot training and test loss over epochs
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 8. Evaluate the model
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch).squeeze()
        predictions = (predictions > 0.5).float()  # Binary threshold
        y_true.extend(y_batch.tolist())
        y_pred.extend(predictions.tolist())

# 9. Classification report
print(classification_report(y_true, y_pred))
