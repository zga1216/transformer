import torch
import torch.nn as nn
import torch.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention 缩放点积注意力机制 '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 缩放因子，通常是 d_k 的平方根
        self.dropout = nn.Dropout(attn_dropout)  # dropout层，防止过拟合

    def forward(self, q, k, v, mask=None):
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # q.shape: (batch_size, n_head, seq_len, d_k)
        # k.transpose(2, 3).shape: (batch_size, n_head, d_k, seq_len)
        # attn.shape: (batch_size, n_head, seq_len, seq_len)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:  # 加入掩码，处理padding等情况
            # mask.shape: (batch_size, 1, seq_len) 或 (batch_size, 1, 1, seq_len)
            # 将mask中为0的位置（需要被忽略的位置）的注意力分数设为极小的负数
            attn = attn.masked_fill(mask == 0, -1e9)

        # 对注意力分数进行softmax归一化，得到注意力权重
        # 然后应用dropout进行正则化
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # 用注意力权重加权求和value向量
        # attn.shape: (batch_size, n_head, seq_len, seq_len)
        # v.shape: (batch_size, n_head, seq_len, d_v)
        # output.shape: (batch_size, n_head, seq_len, d_v)
        output = torch.matmul(attn, v)

        return output, attn  # 返回输出和注意力权重
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module 多头注意力机制模块 '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head    # 注意力头的数量
        self.d_k = d_k          # 每个注意力头的键/查询维度
        self.d_v = d_v          # 每个注意力头的值维度

        # 线性变换层，将输入投影到多个头的Q、K、V空间
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # 查询投影
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # 键投影  
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # 值投影
        
        # 输出投影层，将多个头的输出合并回原始维度
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 缩放点积注意力机制实例
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)        # 输出dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q  # 保存残差连接的原始输入

        # 第一步：线性投影并重塑为多头格式
        # q.shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_head * d_k)
        # view之后: (batch_size, seq_len, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 第二步：转置维度以便进行注意力计算
        # 从 (batch_size, seq_len, n_head, d_k) -> (batch_size, n_head, seq_len, d_k)
        # 这样每个头可以独立计算注意力
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 扩展mask维度以匹配多头格式
            # 从 (batch_size, seq_len) -> (batch_size, 1, seq_len) 或 (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1)   # 在头维度上广播

        # 第三步：计算缩放点积注意力
        q, attn = self.attention(q, k, v, mask=mask)

        # 第四步：转回原始维度格式并合并多头输出
        # 从 (batch_size, n_head, seq_len, d_v) -> (batch_size, seq_len, n_head, d_v)
        q = q.transpose(1, 2).contiguous()  # contiguous确保内存连续性
        # 合并所有头的输出: (batch_size, seq_len, n_head * d_v)
        q = q.view(sz_b, len_q, -1)
        
        # 第五步：通过输出投影层
        q = self.dropout(self.fc(q))
        
        # 第六步：残差连接
        q += residual
        
        # 第七步：层归一化
        q = self.layer_norm(q)

        return q, attn  # 返回输出和注意力权重
    
    
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
    

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn