"""
Test the time and CUDA memory consumption of different attention mechanisms.
"""

from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from hierarchical_mm_tvm import graph_mm as graph_mm_tvm
import argparse
import time
import numpy as np
from math import sqrt

torch.cuda.set_device(0)
print('Using device: {}'.format(torch.cuda.get_device_name()))
import pynvml
pynvml.nvmlInit()


def get_q_k(input_size, window_size, stride, device):
    """Get the query-key index for PAM-TVM"""
    '''
    这段代码定义了一个名为get_q_k的函数，函数有四个参数：

    input_size：表示输入的序列长度，类型为整数。
    window_size：表示窗口的大小，类型为整数。
    stride：表示步长，类型为整数。
    device：表示使用的设备，类型为字符串。
    函数返回一个二维的张量mask，其中每一行表示一个位置对应的注意力分数。

    函数内部首先根据输入的序列长度、窗口大小和步长计算出一些辅助变量，包括：

    second_length：第二阶段序列的长度。
    second_last：第二阶段序列的最后一个位置的下标。
    third_start：第三阶段序列的起始下标。
    third_length：第三阶段序列的长度。
    third_last：第三阶段序列的最后一个位置的下标。
    max_attn：记录窗口大小和三个阶段序列的最大值。
    fourth_start：第四阶段序列的起始下标。
    fourth_length：第四阶段序列的长度。
    full_length：所有阶段序列的总长度。
    fourth_last：第四阶段序列的最后一个位置的下标。
    然后创建一个二维的张量mask，大小为(full_length, max_attn)，并初始化为-1。
    接着对每个位置计算对应的注意力分数。
    具体来说，对于第一个阶段中的每个位置$i$，将其前window_size个位置的下标保存到mask的第i行的前window_size列，
    如果下标越界则将对应的位置设为-1，将位置$i$映射到第三阶段序列中的位置并保存在mask的第i行的最后一列。
    对于第二阶段和第三阶段的每个位置也进行类似的处理。最后，将计算出的mask返回。
    '''
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """从查询-键索引获取键-查询索引，用于PAM-TVM"""
    # 克隆查询-键掩码，得到一个键-查询掩码
    k_q_mask = q_k_mask.clone()
    # 遍历查询-键掩码中的每个元素
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            # 如果查询-键掩码的当前元素是正数
            if q_k_mask[i, j] >= 0:
                # 在键-查询掩码中找到该查询在哪个位置出现过
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]

    return k_q_mask


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    '''
    函数名：get_mask

        输入参数：
        
        input_size：输入序列的长度
        window_size：每个注意力头的窗口大小
        inner_size：每个尺度内的内部注意力窗口大小
        device：使用的设备
        输出：PAM-Naive的注意力掩码
        
        功能：生成PAM-Naive的注意力掩码，其中包含了多个尺度的注意力信息。具体实现为，对于每个尺度，计算其内部注意力掩码，然后把这些掩码拼接在一起得到最终的注意力掩码。
        
        实现细节：
        
        首先，计算每个尺度的大小，这里假设每个尺度内的元素数量为窗口大小的整数倍，因此可以用floor函数计算每个尺度的大小。
        然后，计算所有尺度中元素的总数量。
        接着，对于每个尺度，根据其大小计算内部注意力掩码，具体实现为，在每个位置上，将左右内部窗口内的位置全部标记为1。
        最后，将所有尺度的注意力掩码拼接在一起，得到最终的注意力掩码。
'''
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    second_size = math.floor(input_size / window_size)
    all_size.append(second_size)
    third_size = math.floor(second_size / window_size)
    all_size.append(third_size)
    fourth_size = math.floor(third_size / window_size)
    all_size.append(fourth_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # Get the intra-scale mask of each scale
    inner_window = inner_size // 2
    # The first scale
    for i in range(input_size):
        left_side = max(i - inner_window, 0)
        right_side = min(i + inner_window + 1, input_size)
        mask[i, left_side:right_side] = 1
    # The second scale
    start = input_size
    for i in range(start, start + second_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + second_size)
        mask[i, left_side:right_side] = 1
    # The third scale
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + third_size)
        mask[i, left_side:right_side] = 1
    # The fourth scale
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + fourth_size)
        mask[i, left_side:right_side] = 1

    # Get the inter-scale mask
    start = input_size
    for i in range(start, start + second_size):
        left_side = (i - input_size) * window_size
        if i == (start + second_size - 1):
            right_side = start
        else:
            right_side = (i - input_size + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    # The third scale
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = input_size + (i - start) * window_size
        if i == (start + third_size - 1):
            right_side = start
        else:
            right_side = input_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    # The fourth scale
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = input_size + second_size + (i - start) * window_size
        if i == (start + fourth_size - 1):
            right_side = start
        else:
            right_side = input_size + second_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


"""PAM"""
class GraphSelfAttention(nn.Module):
    def __init__(self, opt):
        super(GraphSelfAttention, self).__init__()
        self.normalize_before = opt.normalize_before  # 是否在每个sub-layer之前进行Layer Normalization
        self.n_head = opt.n_head  # 头数
        self.d_k = opt.d_k  # 每个头的维度

        # 定义三个线性层，用于计算Q、K、V
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)

        # 初始化线性层的参数
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        # 定义全连接层，用于将多头的输出连接起来
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        # 定义Layer Normalization和两个dropout层
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)

        # 获取PAM-TVM中需要用到的掩码
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size
        self.q_k_mask = get_q_k(self.seq_len, self.window_size, self.stride_size, opt.device)
        self.k_q_mask = get_k_q(self.q_k_mask)

    def forward(self, hidden_states):
        """Graph Attention Network 模型前向传播

        Args:
            hidden_states (tensor): 输入的 tensor，形状为 [batch_size, seq_len, d_model]。

        Returns:
            tensor: 输出的 tensor，形状为 [batch_size, seq_len, d_model]。
        """
        residual = hidden_states

        # 得到输入序列的信息
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        # 生成查询向量、键向量、值向量
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        # 对查询向量、键向量的最后一维进行拆分，拆成 num_heads 个 d_k 维度，便于并行计算
        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()

        # 计算图卷积
        # attn_weights.size(): (batch_size, L, num_heads, 11)
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, -1000000000)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        # 对值向量的最后一维进行拆分，拆成 num_heads 个 d_k 维度，便于并行计算
        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()

        # 计算图卷积
        # is_t1_diagonaled=True
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()

        # 将每个头计算得到的结果进行拼接，得到输出的 tensor
        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


class NormalSelfAttention(nn.Module):
    def __init__(self, opt):
        super(NormalSelfAttention, self).__init__()

        # 是否在layer norm之前进行self-attention
        self.normalize_before = opt.normalize_before
        # 多头数
        self.n_head = opt.n_head
        # 每个头的维度
        self.d_k = opt.d_k

        # 线性层，用于计算query、key、value
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        # 权重初始化
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        # 线性层，用于合并多头结果
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        # 权重初始化
        nn.init.xavier_uniform_(self.fc.weight)

        # Layer Norm层
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)

        # Dropout层，用于self-attention和合并多头结果
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)

        # 序列长度、窗口大小和步长大小
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size

        # 是否使用掩码（Mask）
        if opt.mask:
            self.mask, _ = get_mask(self.seq_len, self.stride_size, self.window_size, opt.device)
        else:
            self.mask = None

    def forward(self, hidden_states):
        # 保存输入的残差
        residual = hidden_states

        # 对输入进行一些处理
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        # 获取查询、键、值
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        # 重塑形状以进行矩阵乘法
        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        # 进行矩阵乘法，得到注意力权重
        attn = torch.matmul(q, k.transpose(2, 3))

        # 如果使用了mask，则将mask应用于注意力权重
        if self.mask is not None:
            attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(1), -1e9)

        # 对注意力权重进行softmax，得到注意力值
        attn = self.dropout_attn(F.softmax(attn, dim=-1))

        # 根据注意力值和值计算上下文向量
        attn = torch.matmul(attn, v).transpose(1, 2).contiguous()
        attn = attn.view(bsz, seq_len, self.n_head * self.d_k)
        context = self.dropout_fc(self.fc(attn))
        context += residual

        # 是否使用Layer Normalization
        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


"""Prob-sparse attention"""
class ProbSparseAttention(nn.Module):
    def __init__(self, opt):
        super(ProbSparseAttention, self).__init__()
        self.normalize_before = opt.normalize_before  # 是否在Multi-head self attention前进行 Layer Normalization
        self.n_head = opt.n_head  # 头数
        self.d_k = opt.d_k  # Q/K/V的维度

        # 初始化Linear变换
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        # 初始化全连接层
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        # 初始化Layer Normalization、Dropout等参数
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)

        self.seq_len = opt.seq_len  # 序列长度
        self.factor = opt.factor  # 超参，控制稀疏程度

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        '''
        定义了 ProbSparseAttention 类，其中的 _prob_QK 方法用于计算稀疏 Q-K 矩阵，它的输入参数包括 Q、K、sample_k 和 n_top。
        参数 Q、K 分别表示 query 和 key，均为四维张量（batch_size, n_head, seq_len, d_k），
            其中 n_head 表示头的个数，d_k 表示每个头的维度。sample_k 是采样数，
        n_top 表示 Q-K 矩阵中需要保留的元素个数。
        方法中先计算样本 Q_K（大小为 batch_size * n_head * n_top * L_K），其中 L_K 表示 K 矩阵的长度，即 seq_len。
        通过对每个 query 的 Q_K 矩阵计算它的稀疏程度 M，再取最稀疏的 n_top 个 query 所对应的 Q，用于计算稀疏 Q-K 矩阵。
        该方法返回的 Q_K 矩阵大小为 batch_size * n_head * n_top * L_K，M_top 是稀疏程度最高的 n_top 个 query 的索引。
        '''
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top


    """
    _get_initial_context方法用于初始化上下文，将输入的V求均值，并将结果进行复制扩展，
    以便与query矩阵相乘。其中，V的形状为[B, H, L_V, D]，L_Q表示query矩阵的长度，contex的形状为[B, H, L_Q, D]。
    """
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()

        return contex

    """
       _update_context方法用于在每个采样的query位置更新上下文，其中attn矩阵由得分计算而来，而得分来自query和key矩阵的点积。
       context_in为当前的上下文矩阵，index为对应query的索引。context_in将根据得分和value矩阵的加权平均进行更新。
       最后，将更新后的context_in返回。
   """
    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, hidden_states):
        residual = hidden_states  # 建立残差连接

        hidden_states = hidden_states # 获取隐藏状态
        bsz, seq_len, _ = hidden_states.size()

        q = hidden_states # Q,K,V 的计算
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q) # Query
        k = self.w_ks(hidden_states) # Key
        v = self.w_vs(hidden_states) # Value
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2) # 多头注意力计算
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous() # 强制转换为浮点数
        k = k.float().contiguous()
        v = v.float().contiguous()

        u = U_part = self.factor * np.ceil(np.log(seq_len)).astype('int').item() # c*ln(L_k)

        U_part = U_part if U_part<seq_len else seq_len
        u = u if u < seq_len else seq_len

        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)  # 进行稀疏化注意力机制的计算，返回稀疏化注意力机制的得分和索引

        # get the context
        context = self._get_initial_context(v, seq_len)  # 获取初始化的上下文
        # update the context with selected top_k queries 使用上述稀疏化的注意力机制更新上下文，并转置
        context = self._update_context(context, v, scores_top, index, seq_len).transpose(1, 2).contiguous()

        context = context.view(bsz, seq_len, self.n_head * self.d_k) # 将得到的上下文重塑形状

        context = self.dropout_fc(self.fc(context)) # Dropout + 全连接层
        context += residual # 添加残差连接

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


"""
    用于解析命令行参数并返回一个args对象，args对象中包含了这些参数的值。
    
    argparse是一个Python内置库，它可以帮助我们解析命令行参数，并把它们转换成Python中的对象。
    parser = argparse.ArgumentParser(description='Needed for graph self attention.')：创建一个命令行解析器对象，
        description是一个可选的参数，用于提供程序的简短描述。
    parser.add_argument()：定义需要解析的命令行参数。
    '-d_model', '-d_k', '-normalize_before', '-n_head', '-dropout'分别是需要解析的参数的名称。
    type=int表示参数值的类型是整数。
    default=xxx表示当参数未提供时，将使用默认值xxx。
    '-window_size', '-stride_size', '-factor', '-mask', '-seq_len'与上述参数类似，不再赘述。
    args = parser.parse_args()：将解析器解析命令行参数，并返回一个命名空间（Namespace）对象args，其中包含了所有解析出来的参数和它们的值。
    最后，将args对象返回。
"""
def parsing():
    parser = argparse.ArgumentParser(description='Needed for graph self attention.')
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-normalize_before', type=bool, default=False)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)

    # arguments for Multiformer
    parser.add_argument('-window_size', type=int, default=3)
    parser.add_argument('-stride_size', type=int, default=25)

    # arguments for ProbSparse
    parser.add_argument('-factor', type=int, default=5)

    # arguments for full-attention
    parser.add_argument('-mask', type=int, default=0)

    parser.add_argument('-seq_len', type=int, default=1000)
    args = parser.parse_args()

    return args


def test_NSA(args, input_len):
    """Test the time and CUDA memory consumption of normal self attention."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    NSA_Layer = NormalSelfAttention(args).to(args.device)
    optimizer = optim.Adam(NSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32).to(args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model).to(args.device)

    # Preload the layer
    result = NSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    start_time = time.time()
    for i in range(1000):
        result = NSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('NSA used average time: {} s'.format(round((time.time() - start_time) / 1000, 4)))
    used_memory = used_memory / 1000
    print('NSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


def test_GSA(args, input_len):
    """Test the time and CUDA memory consumption of PAM."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    GSA_Layer = GraphSelfAttention(args).to(args.device)
    optimizer = optim.Adam(GSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32, device=args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model, device=args.device)

    # Preload the layer
    result = GSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    repeat_times = 1000
    start_time = time.time()
    for i in range(repeat_times):
        result = GSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('GSA used time:{} s'.format(round((time.time() - start_time) / repeat_times, 4)))
    used_memory = used_memory / repeat_times
    print('GSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


def test_PSA(args, input_len):
    """Test the time and CUDA memory consumption of Prob-sparse self attention."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    LSA_Layer = ProbSparseAttention(args).to(args.device)
    optimizer = optim.Adam(LSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32, device=args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model, device=args.device)

    # Preload the layer
    result = LSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    repeat_times = 1000
    start_time = time.time()
    for i in range(repeat_times):
        result = LSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('LSA used time:{} s'.format(round((time.time() - start_time) / repeat_times, 4)))
    used_memory = used_memory / repeat_times
    print('LSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


if __name__ == '__main__':
    args = parsing()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    input_size = args.seq_len
    stride = args.stride_size
    second_length = input_size // stride
    third_length = second_length // stride
    fourth_length = third_length // stride
    input_len = input_size + second_length + third_length + fourth_length

    if args.mask:
        print('sequence length: {}'.format(input_len))
        test_NSA(args, input_len)
    else:
        print('sequence length: {}'.format(input_size))
        test_NSA(args, input_size)

    print('sequence length: {}'.format(input_len))
    test_GSA(args, input_len)
    print('sequence length: {}'.format(input_size))
    test_PSA(args, input_size)

