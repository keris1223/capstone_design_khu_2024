import pandas as pd
import random

# CSV 파일 읽기
file_path = 'output_data3_preprocessed.csv'  # 원본 파일 경로
data = pd.read_csv(file_path)

# machine_id에서 고유한 값 추출
unique_machine_ids = data['machine_id'].unique()

# 랜덤하게 100개의 machine_id를 선택
random_machine_ids = random.sample(list(unique_machine_ids), k=100)

# 선택된 machine_id로 필터링
filtered_data = data[data['machine_id'].isin(random_machine_ids)]

# 결과를 새로운 CSV 파일로 저장
output_path = 'output_data3_preprocessed_random_100_machine_id.csv'  # 저장 파일 경로
filtered_data.to_csv(output_path, index=False)

print(f"{output_path}에 저장했습니다.")
