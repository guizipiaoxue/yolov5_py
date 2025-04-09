import math

import torch
from torch import nn


class Encoder(nn.Module):
    """
    编码器的基类接口
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """
    解码器的基类接口
    """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """初始化隐状态"""
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """编码器-解码器基类接口"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args) # 更新隐状态
        return self.decoder(dec_X, dec_state)

class Seq2SeqEncoder(Encoder):
    """
    序列到序列编码器
    利用门循环神经网络
    """
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 用嵌入层降低维度
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这里可以用GRU或者LSTM
        self.rnn = nn.GRU(embed_size, num_hidden, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X.shape = [batch_size, num_steps, embed_size]
        X = self.embedding(X)
        # X.shape = [num_steps, batch_size, embed_size]
        X = X.permute(1, 0, 2) # 这个函数交换维度
        output, state = self.rnn(X)
        return output, state

class Seq2SeqDecoder(Decoder):
    """
    序列到序列解码器
    """
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hidden , num_hidden, num_layers, dropout=dropout)
        # 全连接层
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        # 广播机制让state有和X一样的大小
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

def sequence_mask(X, valid_len, value=0.):
    """屏蔽序列不相关项"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """掩蔽softmax操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """加性注意力机制"""
    def __inti__(self, key_size, query_size, num_hidden, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hidden, bias=False)
        self.W_q = nn.Linear(query_size, num_hidden, bias=False)
        self.W_v = nn.Linear(num_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        # 广播求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 这个函数添加维度
        features = torch.tanh(features)
        # 这里得删掉最后一维
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """缩放点注意力机制"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AttentionDecoder(Decoder):
    """带注意力机制解码器基类接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    # 修饰一下
    def attention_weights(self):
        return NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """序列到序列的注意力编码器"""
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hidden, vocab_size, embed_size, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hidden , num_hidden, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs = []
        self._attention_weight = []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], 1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat([context, torch.unsqueeze(x, dim=1)], dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weight.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weight(self):
        return self._attention_weight

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, key_size, query_size, value_size, num_hidden,
                 num_head, dropout,basic=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head = num_head
        # 这里选择缩放点可以减少计算量，但是也能变成加性注意力机制
        # self.attention = AdditiveAttention(query_size, key_size, num_hidden, dropout)
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hidden, bias=basic)
        self.W_k = nn.Linear(key_size, num_hidden, bias=basic)
        self.W_v = nn.Linear(value_size, num_hidden, bias=basic)
        self.W_o = nn.Linear(num_hidden, num_hidden, bias=basic)

    def forward(self, queries, keys, values, valid_lens):
        # 首先进行全连接层
        queries = transpose_qkv(self.W_q(queries), self.num_head)
        keys = transpose_qkv(self.W_k(keys), self.num_head)
        values = transpose_qkv(self.W_v(values), self.num_head)

        if valid_lens is not None:
            # 因为我们要对很多头进行操作，所以要重复几次
            valid_lens = torch.repeat_interleave(valid_lens, repeats = self.num_head, dim=0)

        outputs = self.attention(queries, keys, values, valid_lens)
        # 连接
        outputs_concat = transpose_qkv(outputs, self.num_head)
        return self.W_o(outputs_concat)


def transpose_qkv(X, num_heads):
    """把输入的张量变成我们多头注意力机制的头个数"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 把X的形状变成(batch_size，num_heads，查询或者“键－值”对的个数， num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最后输出，batch_size*num_head, q or k, num_hiddens/num_heads
    return X.resahpe(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """反解张量"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.rehsape(X.shape[0], X.shape[1], -1)