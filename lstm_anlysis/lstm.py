import numpy as np

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, batch_size):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # forget gate의 weight와 bias
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim)
        print("forget gate의 가중치 매트릭스:\n",self.W_f)
        self.b_f = np.zeros((hidden_dim, 1))
        print("forget gate의 bias\n",self.b_f)

        # input gate의 weight와 bias
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim)
        print("input gate의 가중치 매트릭스:\n",self.W_i)
        self.b_i = np.zeros((hidden_dim, 1))
        print("input gate의 bias\n",self.b_i)

        # cell gate의 weight와 bias
        self.W_C = np.random.randn(hidden_dim, hidden_dim + input_dim)
        print("cell gate의 가중치 매트릭스:\n",self.W_C)
        self.b_C = np.zeros((hidden_dim, 1))
        print("cell gate의 bias\n",self.b_C)

        # output gate의 weight와 bias
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim)
        print("cell gate의 가중치 매트릭스:\n",self.W_o)
        self.b_o = np.zeros((hidden_dim, 1))
        print("output gate의 bias\n",self.b_o)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, C_prev):
        combined = np.vstack((h_prev, x))
        print("combined:\n",combined) # 8 x 1 매트릭스

        # Forget gate
        f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f) # (5 x 8 매트릭스) 행렬 곱 (8 x 1 매트릭스) + 5 x 1 매트릭스
        print("f_t:\n",f_t)

        # Input gate
        i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
        print("i_t:\n",i_t)

        # Cell gate
        g_t = self.tanh(np.dot(self.W_C, combined) + self.b_C)
        print("g_t:\n",g_t)

        # Update cell state
        C_t = f_t * C_prev + i_t * g_t  # (5 x 1 메트릭스) 아다마르 곱 (5 x 1 메트릭스) + (5 x 1 메트릭스) 아다마르 곱 (5 x 1 메트릭스)
        print("C_t:\n",C_t)

        # Output gate
        o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
        print("o_t:\n",o_t)

        h_t = o_t * self.tanh(C_t)
        print("h_t:\n",h_t)

        self.cache = (x, h_prev, C_prev, combined, f_t, i_t, g_t, o_t, C_t, h_t)

        return h_t, C_t


# 임의의 입력 값
input_dim = 3  # 입력 차원
hidden_dim = 5  # 은닉 상태 차원
batch_size = 1 # 한 배치에 넣을 샘플의 수
x = np.random.randn(input_dim, batch_size)  # 임의의 입력
print("입력 x:\n", x)

h_prev = np.zeros((hidden_dim, batch_size))  # 초기 은닉 상태
print("초기 은닉 상태 h_prev:\n", h_prev)

C_prev = np.zeros((hidden_dim, batch_size))  # 초기 셀 상태
print("초기 셀 상태 C_prev:\n", C_prev)


# LSTM 셀 생성 및 순방향 계산
lstm_cell = LSTMCell(input_dim, hidden_dim, batch_size)
h_next, C_next = lstm_cell.forward(x, h_prev, C_prev)

print("다음 히든 상태:\n", h_next)
print("다음 셀 상태:\n", C_next)