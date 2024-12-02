import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader,TensorDataset
import pandas as pd


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
        sequences = torch.tensor(
            self.machine_sequences[machine_id]['sequences'], dtype=torch.float32
        )
        targets = torch.tensor(
            self.machine_sequences[machine_id]['targets'], dtype=torch.float32
        )
        return TensorDataset(sequences, targets)
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
def prepare_predict_dataloader(data, sequence_length, features):

    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=1, shuffle=False)
        print(f"Prepared Predict DataLoader for Machine ID {machine_id}, Total Sequences: {len(machine_dataset)}")
    return dataloaders


def predict_and_save(model, dataloaders, device, output_file):
    results = []
    for machine_id, dataloader in dataloaders.items():
        print(f"Machine ID {machine_id}")
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)  # 모델 예측
                predictions.append(outputs.cpu().numpy())

        # 예측값을 정리하여 리스트에 추가
        for pred in predictions:
            results.append({
                "machine_id": machine_id,
                **{f"pred_{feature}": value for feature, value in zip(features, pred[0])}
            })

    # 결과를 DataFrame으로 변환 후 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# 주요 설정
sequence_length = 100
batch_size = 32
num_epochs = 20
learning_rate = 0.0001
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
    'page_cache_memory'
]
input_size = len(features)
output_size = len(features)

# 새로운 데이터셋 로드
predict_file_path = 'output_data3_preprocessed_151918055.csv'
predict_data = pd.read_csv(predict_file_path)
predict_data = predict_data[predict_data['machine_id'] != -1]

# DataLoader 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

predict_dataloaders = prepare_predict_dataloader(predict_data, sequence_length, features)

# 학습된 모델 로드
model_path = "performance_predict_model.pth"
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))
print(f"Model loaded from {model_path}")

# 예측 수행 및 저장
output_file = "prediction_results.csv"
predict_and_save(model, predict_dataloaders, device, output_file)
