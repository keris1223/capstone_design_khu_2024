import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pandas as pd
import numpy as np

# 데이터셋 클래스 정의
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.machine_sequences = {}  # machine_id별 시퀀스 저장
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id')  # machine_id별 데이터 그룹화
        for machine_id, group in grouped:
            values = group[self.features].values  # 여러 특성의 값 추출
            sequences, targets = [], []
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                targets.append(values[i + self.sequence_length])
            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'targets': targets
            }

    def get_machine_dataset(self, machine_id):
        """
        특정 machine_id의 데이터를 반환합니다.
        """
        if machine_id not in self.machine_sequences:
            raise ValueError(f"Machine ID {machine_id} not found in dataset.")
        sequences = torch.tensor(
            self.machine_sequences[machine_id]['sequences'], dtype=torch.float32
        )
        targets = torch.tensor(
            self.machine_sequences[machine_id]['targets'], dtype=torch.float32
        )
        return TensorDataset(sequences, targets)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # LSTM 실행, 최종 히든 상태만 추출
        hidden = hidden[-1]  # 마지막 LSTM 레이어의 히든 상태
        output = self.fc(hidden)  # 히든 상태를 FC 레이어에 전달
        return output

# 데이터 준비 함수
def prepare_dataloaders(data, sequence_length, batch_size, features):
    """
    machine_id별 DataLoader를 생성합니다.
    """
    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=batch_size, shuffle=True)
        print(f"Prepared DataLoader for Machine ID {machine_id}, Total Sequences: {len(machine_dataset)}")
    return dataloaders


# 학습 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()  # 모델 학습 모드
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # GPU/CPU로 전송
            optimizer.zero_grad()
            outputs = model(inputs)  # 예측값 계산
            loss = criterion(outputs, targets)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# 예측 함수
def predict(model, dataloader, device):
    model.eval()  # 평가 모드
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # 모델 예측
            predictions.extend(outputs.cpu().numpy())  # 예측값 저장
            targets.extend(target.numpy())  # 실제값 저장
    return np.array(predictions), np.array(targets)

def save_model(model, model_path = "performance_predict_model.pth"):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 주요 설정
sequence_length = 100
batch_size = 32
num_epochs = 20
learning_rate = 0.0005
hidden_size = 50
num_layers = 2

# 성능 지표
features = [
    'average_usage_cpus',
    'average_usage_memory',
    'maximum_usage_cpus',
    'maximum_usage_memory',
    'random_sample_usage_cpus',
    'assigned_memory',
    'page_cache_memory',
    'ma_average_usage_cpus',
    'ma_average_usage_memory',
    'diff_maximum_usage_cpus',
]
input_size = len(features)
output_size = len(features)

file_path = 'output_data3_pre_preprocessed_nis5_machine_id.csv'
data = pd.read_csv(file_path)
data = data[data['machine_id'] != -1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataloaders = prepare_dataloaders(data, sequence_length, batch_size, features)

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for machine_id, dataloader in dataloaders.items():
    print(f"Training on Machine ID {machine_id}")
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)


# 모델 저장
save_model(model)
