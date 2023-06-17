"""
Modified based on Informer.
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and Shuai Zhang and Jianxin Li and
               Hui Xiong and Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35}, number    = {12}, pages     = {11106--11115}, publisher = {{AAAI} Press}, year      = {2021},
}
"""

import torch
import torch.nn as nn
import math

# 定义位置编码模块，用于将序列中每个位置的信息编码成一个向量
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 生成位置信息向量，每个元素代表该位置的索引
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 生成除数项，用于计算正弦和余弦函数的参数
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 根据公式计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码转换为一个可学习的参数，添加到模型的缓冲区中
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入张量的长度，截取相应长度的位置编码
        return self.pe[:, :x.size(1)]

# 定义标记嵌入模块，用于将输入序列中的每个标记映射到一个d_model维向量
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 使用Conv1d模块对输入序列进行卷积操作，将c_in个标记转换为d_model维向量
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')

        # 对模块中的所有卷积层进行权重初始化，采用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 将输入张量按照标记维和时间维进行转置，进行卷积操作后再进行转置
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# 定义固定嵌入模块，用于将每个标记的位置信息和标记本身的信息相结合
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 定义一个大小为c_in × d_model的零张量
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # 生成位置信息向量，每个元素代表该位置的索引
        position = torch.arange(0, c_in).float().unsqueeze(1)
        # 生成除数项，用于计算正弦和余弦函数的参数
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 根据公式计算位置编码
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码转换为一个可学习的参数，添加到模型的嵌入层中
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # 将输入张量映射到d_model维空间，同时将嵌入层的参数设为不可训练
        return self.emb(x).detach()


# 定义时间特征嵌入模块，用于将时间特征嵌入到模型中
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)


"""嵌入模块。DataEmbedding用于ETT数据集的长期预测。"""
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)  # 标记嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 位置嵌入
        self.temporal_embedding = TimeFeatureEmbedding(d_model)  # 时间嵌入

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 将标记嵌入、位置嵌入和时间嵌入相加作为最终嵌入结果
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)


"""CustomEmbedding用于电力和应用流量数据集的长期预测。"""
class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)  # 标记嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 位置嵌入
        self.temporal_embedding = nn.Linear(temporal_size, d_model)  # 时间嵌入
        self.seqid_embedding = nn.Embedding(seq_num, d_model)  # 序列 ID 嵌入

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 将标记嵌入、位置嵌入、时间嵌入和序列 ID 嵌入相加作为最终嵌入结果
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1]) \
            + self.seqid_embedding(x_mark[:, :, -1].long())

        return self.dropout(x)


class SimpleClassEmbedding(nn.Module):
    def __init__(self, seq_len, d_input, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_input = d_input
        # 构建数据特征嵌入
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=3, padding=padding,
                                padding_mode='circular')

        # 保留位置编码
        self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        x = self.position_embedding(x) + self.data_emb(x.transpose(1, 2)).transpose(1, 2)
        return x



"""定义单步预测嵌入模块，用于所有数据集的单步预测"""
class SingleStepEmbedding(nn.Module):
    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq

        # 构建覆盖率特征嵌入
        self.cov_emb = nn.Linear(cov_size + 1, d_model)

        # 构建数据特征嵌入
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding,
                                  padding_mode='circular')

        # 构建位置编码
        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
                                         device=device)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        输入：batch*seq_len。
        输出：batch*seq_len*d_model。
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        # 将输入拆分为覆盖率特征和数据特征
        covs = x[:, :, 1:(1 + self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)

        # 进行嵌入
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(x[:, :, 0].unsqueeze(2).permute(0, 2, 1)).transpose(1, 2)
        embedding = cov_embedding + data_embedding

        # 添加位置编码
        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(position, self.position_vec.to(x.device))
        embedding += position_emb

        return embedding

