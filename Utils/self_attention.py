import torch
import torch.nn.functional as F

# 定义输入序列 (batch_size=1, sequence_length=3, embedding_dim=4)
inputs = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # 第一个单词的嵌入
    [0.0, 2.0, 0.0, 2.0],  # 第二个单词的嵌入
    [1.0, 1.0, 1.0, 1.0]   # 第三个单词的嵌入
]).unsqueeze(0)  # 添加批次维度 (1, 3, 4)

# 定义参数：用于将 inputs 映射到 Q, K, V
embedding_dim = 4  # 嵌入维度
d_k = 4  # Q 和 K 的维度（通常等于 embedding_dim）

# 初始化权重矩阵
W_q = torch.nn.Linear(embedding_dim, d_k, bias=False)
W_k = torch.nn.Linear(embedding_dim, d_k, bias=False)
W_v = torch.nn.Linear(embedding_dim, d_k, bias=False)

# 计算 Query, Key, Value
Q = W_q(inputs)  # (1, 3, 4)
K = W_k(inputs)  # (1, 3, 4)
V = W_v(inputs)  # (1, 3, 4)

# 计算注意力得分： Q * K^T / sqrt(d_k)
# 矩阵乘法，先对 K 转置
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # (1, 3, 3)

# 计算注意力权重 (Softmax)
attention_weights = F.softmax(scores, dim=-1)  # (1, 3, 3)

# 生成输出
outputs = torch.matmul(attention_weights, V)  # (1, 3, 4)

# 打印结果
print("Inputs:\n", inputs)
print("\nW_Q (Q):\n", W_q)
print("\nW_K (K):\n", W_k)
print("\nW_V (V):\n", W_v)
print("\nQuery (Q):\n", Q)
print("\nKey (K):\n", K)
print("\nValue (V):\n", V)
print("\nAttention Scores (before softmax):\n", scores)
print("\nAttention Weights (after softmax):\n", attention_weights)
print("\nOutputs:\n", outputs)
