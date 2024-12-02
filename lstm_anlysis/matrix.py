import numpy as np

hidden_dim = 5
batch_size = 4

# 가중치 곱의 결과: (hidden_dim, batch_size) 크기
weighted_sum = np.random.randn(hidden_dim, batch_size)
print(weighted_sum)
# bias 벡터: (hidden_dim, 1) 크기
bias = np.zeros((hidden_dim, 1))

# 연산: 브로드캐스팅이 적용되어 (hidden_dim, batch_size) 크기로 확장
result = weighted_sum + bias
print(result)
