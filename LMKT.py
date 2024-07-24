import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
# from TCN.tcn import TemporalConvNet
import pprint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LMKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False,test=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.test=test
        embed_l = d_model
        self.embed_l = d_model
        self.softmax=nn.Softmax(dim=2)

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)

            self.q_embed_diff = nn.LSTM(embed_l,embed_l,batch_first=True)
            self.qa_embed_diff = nn.LSTM(embed_l,embed_l,batch_first=True)

            self.MengDui_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_MengDui = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_MengDui = nn.Embedding(2 * self.n_question + 1, embed_l)
            self.pid_data_emb=nn.Embedding(self.n_pid+1,embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        self.q_embed_linear=nn.Linear(embed_l*2,1)
        self.q_embed_dn = nn.Embedding(self.n_question+1, embed_l)
        self.q_embed_dn_linear=nn.Linear(embed_l,1)

        # self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            self.qa_embed_linear=nn.Linear(embed_l,1)
            self.qa_embed_fn = nn.Embedding(2*self.n_question+1, embed_l)
            self.qa_embed_fn_linear=nn.Linear(embed_l,1)
        else:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            self.qa_embed_linear=nn.Linear(embed_l,1)
            self.qa_embed_fn = nn.Embedding(2*self.n_question+1, embed_l)
            self.qa_embed_fn_linear=nn.Linear(embed_l,1)
            self.qa_embed_LSTM=nn.LSTM(embed_l*2,embed_l)

        self.self_attention_layer = SelfAttention(embed_l, 1)
        self.self_attention_layer_2 = SelfAttention(embed_l, 1)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.sta=embed_l*2
        self.sLinear=nn.Linear(self.sta,self.n_question)
        self.gLinear=nn.Linear(self.sta,self.n_question)
        self.lLinear=nn.LSTM(self.sta,self.n_question,batch_first=True)
        self.fLinear=nn.LSTM(embed_l,self.n_question,batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(self.n_question,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)


    def tags(self,q_data,qa_data,pid_data):
        data_array=q_data.cpu().numpy()
        # np.savetxt('data.txt', data_array.reshape(-1), delimiter=',')

        p_emb=self.pid_data_emb(pid_data)
        p_emb = p_emb.unsqueeze(2).expand(-1, -1, 7, -1)

        q_embed_data = self.q_embed(q_data)
        q_embed_data=self.self_attention_layer(p_emb,q_embed_data,q_embed_data)
        q_embed_data=torch.sum(q_embed_data,dim=2)#24 200 256

        qa_embed_data = self.qa_embed(qa_data)
        qa_embed_data=torch.cat((qa_embed_data,p_emb),-1)
        batch_size, seq_len, _, _ = qa_embed_data.size()
        qa_embed_data = qa_embed_data.view(batch_size * seq_len, -1, self.embed_l*2)

        qa_embed_data,_=self.qa_embed_LSTM(qa_embed_data)
        qa_embed_data = qa_embed_data.view(batch_size, seq_len, -1, qa_embed_data.size(-1))

        qa_embed_data=self.self_attention_layer_2(p_emb,qa_embed_data,qa_embed_data)

        qa_embed_data=torch.sum(qa_embed_data,dim=2)#24 200 256


        return q_embed_data,qa_embed_data

    def forward(self, q_data, qa_data, target, pid_data=None,test=False):
        # Batch First




        if self.n_pid > 0:
            p_emb = self.pid_data_emb(pid_data)
            q_embed_data,qa_embed_data=self.tags(q_data,qa_data,pid_data)

            c_reg_loss = 0
        else:
            c_reg_loss = 0

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)


        S=self.sLinear(concat_q)
        S=torch.sigmoid(S)
        G=self.gLinear(concat_q)
        G=torch.sigmoid(G)
        L,_=self.lLinear(concat_q)
        L=torch.sigmoid(L)
        F,_=self.fLinear(q_embed_data)
        F=torch.sigmoid(F)

        Lq=L*(1-F)
        h=(Lq*(1-S)+(1-Lq)*G)
        # h=Lq
        if test==True:
            np.save("../L.npy", L.cpu().detach().numpy())
            np.save("../S.npy", S.cpu().detach().numpy())
            np.save("../G.npy", G.cpu().detach().numpy())
            np.save("../F.npy", F.cpu().detach().numpy())
            np.save("../q.npy", q_data.cpu().detach().numpy())
            np.save("../a.npy", target.cpu().detach().numpy())
            np.save("../p.npy", pid_data.cpu().detach().numpy())
        output = self.out(h)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum(), m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = x
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class SelfAttention(nn.Module):
    def __init__(self, skill_embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.skill_embed_dim = skill_embed_dim
        self.num_heads = num_heads

        # 分别定义查询（query）、键（key）和值（value）的线性变换层
        self.query_linear = nn.Linear(skill_embed_dim, skill_embed_dim)
        self.key_linear = nn.Linear(skill_embed_dim, skill_embed_dim)
        self.value_linear = nn.Linear(skill_embed_dim, skill_embed_dim)

    def forward(self, x,y,v):
        batch_size, seq_len, skill_num, skill_embed = x.size()

        # 重塑输入张量以进行自注意力计算
        x = x.reshape(batch_size * seq_len * skill_num, skill_embed)

        # 使用线性变换层计算查询（query）、键（key）和值（value）
        query = self.query_linear(x)
        key = self.key_linear(y)
        value = self.value_linear(v)

        # 重塑查询（query）和键（key）张量以进行自注意力计算
        query = query.view(batch_size, seq_len, skill_num, self.skill_embed_dim)
        key = key.view(batch_size, seq_len, skill_num, self.skill_embed_dim)

        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1))

        # 对注意力分数进行缩放
        scaled_attn_scores = attn_scores / torch.sqrt(torch.tensor(self.skill_embed_dim, dtype=torch.float32))

        # 使用softmax计算注意力权重
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)

        # 重塑值（value）张量以进行自注意力计算
        value = value.view(batch_size, seq_len, skill_num, self.skill_embed_dim)

        # 将注意力权重应用于值（value）
        attended = torch.matmul(attn_weights, value)

        return attended

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
