import torch
import torch.nn as nn
import torch.optim as optim
import math


# Positional Encoding: 입력에 위치 정보를 추가하여 트랜스포머가 순서 정보를 학습할 수 있게 함
# 입력 차원: (batch_size, seq_len, d_model)
# 출력 차원: (batch_size, seq_len, d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # d_model: 입력 임베딩의 차원 수, max_len: 최대 시퀀스 길이
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)  # 위치 인코딩을 저장할 텐서(max_len x d_model) 초기화
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 위치 인덱스 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 주기적 인코딩을 위한 스케일링 값
        # 짝수 인덱스의 값을 생성하는 텐서
        # -math.log(10000)/d_model은 점진적으로 감소하는 주기성을 도입하여 차원이 증가할수록 작아져
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 sin 함수 적용
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 cos 함수 적용
        self.encoding = self.encoding.unsqueeze(0)  # 배치 차원 추가
        print(self.encoding)

    def forward(self, x):
        seq_len = x.size(1)
        x_forward = x + self.encoding[:, :seq_len, :].to(x.device)
        print("입력에 위치 인코딩 추가")
        print(x_forward.size())
        return (x_forward)  # 입력에 위치 인코딩 추가


# Scaled Dot-Product Attention: Self-Attention 계산, Query, Key, Value 행렬에 대한 어텐션 가중치 계산하고 이를 통해
# 이를 통해 value를 가중합함.
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)  # 스케일링을 위한 차원 수
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # Q와 K의 내적에 스케일링 적용
        # math.sqrt(d_k)는 안정적인 학습을 위해 (차원이 커지면 score가 커지는 문제)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 마스크 적용하여 특정 위치를 -1e9로 설정
            # 이를통해 Softmax 계산시 해당 위치의 확률이 0이 되도록
        attention = torch.softmax(scores, dim=-1)  # 소프트맥스 적용하여 Attention 가중치 계산, 이를 마지막 차원에 적용
        output = torch.matmul(attention, V)  # 가중치를 V에 적용
        print("ScaledDotProductAttention_attention size:")
        print(attention.size())
        print("ScaledDotProductAttention_output size:")
        print(output.size())
        return output, attention


# Multi-Head Attention: 여러 개의 Attention Head로 정보를 병렬적으로 학습
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # 각 헤드의 차원 수

        # Q, K, V에 대한 선형 변환 레이어 정의
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)  # 최종 결합 후 출력 레이어

        self.attention = ScaledDotProductAttention()  # Scaled Dot-Product Attention 사용

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        # Q, K, V를 Multi-Head로 분할
        # Q, K, V는 (batch_size, number_heads, seq_len, d_k)
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        print("Q, K, V size :", Q.size(), K.size(), V.size())

        # Scaled Dot-Product Attention 수행
        context, attention = self.attention(Q, K, V, mask=mask)  # output, attention
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # Multi-Head 결합
        output = self.fc(context)  # 최종 선형 변환
        print("Q, K, V의 헤드 결합 :", output.size())
        return output, attention


# Feed Forward Network: 각 Attention 결과에 대해 비선형 변환 적용
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 첫 번째 선형 변환
        self.fc2 = nn.Linear(d_ff, d_model)  # 두 번째 선형 변환

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))  # ReLU 활성화 함수 적용 후 출력


# Transformer Encoder Layer: 한 개의 인코더 레이어, Multi-Head Attention과 FFN으로 구성
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask=mask)  # Self-Attention 수행
        x = x + self.dropout1(attn_output)  # 잔차 연결 및 드롭아웃
        x = self.norm1(x)  # LayerNorm 적용
        ffn_output = self.ffn(x)  # Feed-Forward Network 적용
        x = x + self.dropout2(ffn_output)  # 잔차 연결 및 드롭아웃
        x = self.norm2(x)  # LayerNorm 적용
        return x


# Transformer Encoder: 여러 개의 인코더 레이어를 쌓아 구성
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)  # 위치 인코딩 추가
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)  # 다중 레이어 생성
        ])
        self.norm = nn.LayerNorm(d_model)  # 최종 정규화

    def forward(self, x, mask=None):
        x = self.positional_encoding(x)  # 위치 인코딩 적용
        for layer in self.layers:
            x = layer(x, mask)  # 각 레이어에 대해 순차적으로 처리
            print("encoder layer: ",x.size())
        return self.norm(x)


# Transformer Decoder Layer: 인코더 출력과의 크로스 어텐션 추가
# 입력 차원: (batch_size, seq_len, d_model)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    # enc_output의 출력 차원: (batch_size, src_len, d_model)
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # x는 Q,K,V로 각각 변환 (batch_size, seq_len, d_model)

        self_attn_output, _ = self.self_attention(x, x, x, mask=tgt_mask)  # Self-Attention
        x = x + self.dropout1(self_attn_output)  # 잔차 연결
        x = self.norm1(x)

        # x차원(Q) (batch_size, seq_len, d_model)
        # K,V차원 (batch_size, src_len, d_model)

        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output,
                                                    mask=src_mask)  # Cross-Attention(multi-head attention)
        x = x + self.dropout2(cross_attn_output)  # 잔차 연결
        x = self.norm2(x)
        print("cross_attention", x.size())
        ffn_output = self.ffn(x)  # Feed-Forward Network 적용
        x = x + self.dropout3(ffn_output)  # 잔차 연결
        x = self.norm3(x)
        print("Decoder layer: ", x.size())
        return x


# Transformer Decoder: 여러 개의 디코더 레이어로 구성
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)  # 다중레이어
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            print("decoder output",x.size())
        return self.norm(x)


# Transformer Model: 인코더와 디코더를 결합한 최종 트랜스포머 모델
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)  # 최종 출력 레이어 (어휘 크기에 맞춤)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        print(f"트랜스포머 인코더 {epoch + 1}/{epochs}")
        enc_output = self.encoder(src, src_mask)  # 인코더 처리
        print(f"트랜스포머 디코더 {epoch + 1}/{epochs}")
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)  # 디코더 처리
        output = self.fc_out(dec_output)
        output = torch.softmax(output, dim=-1)  # 최종 출력 생성
        print("트랜스포머 결과",output.size())
        return output


# 모델 초기화 및 학습 설정
d_model = 512  # 입력 모델의 차원 수
num_heads = 8  # 멀티헤드 어텐션의 헤드 수
d_ff = 2048  # 피드 포워드 네트워크의 차원 수
num_layers = 6  # layer 개수
vocab_size = 10000  # 예제 어휘 크기
dropout = 0.1  # 드롭아웃 비율
batch_size = 2
seq_len = 10  # 토큰의 개수

model = Transformer(d_model, num_heads, d_ff, num_layers, vocab_size, dropout)  # 모델 초기화
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 옵티마이저 설정
criterion = nn.CrossEntropyLoss()  # 손실 함수 설정 크로스 엔트로피

# 샘플 학습 루프
epochs = 5
for epoch in range(epochs):
    src = torch.randint(0, vocab_size,
                        (batch_size, seq_len, d_model)).float()  # 예시 입력 데이터 생성 (2,10,512) (batch_size,seq_len,d_model
    tgt = torch.randint(0, vocab_size, (
    batch_size, seq_len, d_model)).float()  # 예시 타겟 데이터 생성 (2,10,512) vocab_size 범위에서 임의로, 디코더의 입력으로 사용되는 타겟 데이터

    optimizer.zero_grad()  # 옵티마이저 초기화 매 학습 루프마다 기울기 초기화
    output = model(src, tgt)  # 모델의 출력 얻기 출력 결과는 (batch_size, seq_len, vocab_size)

    target = torch.randint(0, vocab_size, output.shape[:-1]).long()  # 임의 타겟 텐서 생성, 정답 비교
    loss = criterion(output.view(-1, vocab_size), target.view(-1))  # 손실 계산

    loss.backward()  # 역전파
    optimizer.step()  # 옵티마이저 스텝 적용
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")  # 손실 출력
