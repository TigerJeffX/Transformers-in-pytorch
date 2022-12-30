# encoding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

NEG_INF = -1e9
POS_INF = 1e9

class EMB(nn.Module):
    ''' Embedding : value encoding & position encoding
    '''
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            max_len=60,
            cross_idx=True,
            dropout_rate=0.1):
        '''
        1. lut (lookup table) → value embedding
        2. pe_slots → positional encoding
        '''
        super(EMB, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.cross_idx = cross_idx
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lut = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        pe_slots = self.get_pe_slots(
            seq_len=self.max_len, feat_dim=self.embedding_dim, cross_idx=self.cross_idx)
        # cause pe_slots is persistent state (not model parameter could be learned)
        self.register_buffer("pe_slots", pe_slots)

    @staticmethod
    def get_pe_slots(seq_len, feat_dim, cross_idx):
        assert feat_dim%2==0, "feat dim must be even num but %s"%feat_dim
        pe_slots = torch.zeros(seq_len, feat_dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        even_dim = torch.arange(0, feat_dim, 2)
        power = -even_dim * math.log(10000) / feat_dim
        # either or not 'cross_idx' deserve same performance
        if cross_idx:
            pe_slots[:,0::2] = torch.sin(torch.mul(pos, torch.exp(power)))
            pe_slots[:,1::2] = torch.cos(torch.mul(pos, torch.exp(power)))
        else:
            pe_slots[:,0:feat_dim/2] = torch.sin(pos * torch.exp(power))
            pe_slots[:,feat_dim/2:] = torch.cos(pos * torch.exp(power))
        return pe_slots

    def value_encoding(self, x, use_dim_sqrt_weight=True):
        if not use_dim_sqrt_weight:
            return self.lut(x)
        return self.lut(x) * math.sqrt(self.embedding_dim)

    def positional_encoding(self, x):
        return x + self.pe_slots.unsqueeze(0)[:,:x.size(1), :].requires_grad_(False)

    def forward(self, x):
        x = self.value_encoding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x

class MHA(nn.Module):
    ''' Multi Head Attention
    '''
    def __init__(
            self, hidden_dim, head_num, dropout_rate=0.1):
        super(MHA, self).__init__()
        assert hidden_dim%head_num==0, \
            "hidden_dim %s is not divisible by head_num %s"%(hidden_dim, head_num)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None, use_dropout=True):
        '''
        0. short name explain
            bs : batch_size
            q_sl : query seq len
            kv_sl : key/value seq len
            mha : multi head attention
        1. tensor shape
            mask → [batch, seq_len, seq_len]
            query/key/value → [batch_size, q/k/v_seq_len, emb_dim]
            q/k/v → [batch_size, q/k/v_seq_len, hidden_dim]
            mha_q/k/v → [batch_size, head_num, q/k/v_seq_len, hidden_dim/head_num]
            mha_scores/attn → [batch_size, head_num, q_seq_len, k/v_seq_len]
            mha_o → [batch_size, head_num, q_seq_len, hidden_dim/head_num]
            o → [batch_size, q_seq_len, hidden_dim]
        '''
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        bs, q_sl = q.size(0), q.size(1)
        k_sl = v_sl = k.size(1)
        mha_q = q.view(bs, q_sl, self.head_num, -1).transpose(1, 2)
        mha_k = k.view(bs, k_sl, self.head_num, -1).transpose(1, 2)
        mha_v = v.view(bs, v_sl, self.head_num, -1).transpose(1, 2)
        mha_scores = mha_q.matmul(mha_k.transpose(-1, -2)) / math.sqrt(mha_q.size(-1))
        if mask is not None:
            mha_scores.masked_fill_(mask.unsqueeze(1)==True, NEG_INF)
        mha_attn = mha_scores.softmax(-1)
        if use_dropout:
            mha_attn = self.dropout(mha_attn)
        mha_o = mha_attn.matmul(mha_v)
        o = mha_o.transpose(1, 2).contiguous().view(bs, q_sl, self.hidden_dim)
        return self.w_o(o)

class FFN(nn.Module):
    '''
    Feed Forward Network → considered also attention structure
        q : query → input of ffn
        k : key →  first fc weights in ffn
        v : value → second fc weights in ffn
    '''
    def __init__(
            self, hidden_dim, ffn_dim, dropout_rate=0.1, fc_or_conv='fc'):
        super(FFN, self).__init__()
        self.fc_or_conv = fc_or_conv
        if self.fc_or_conv == 'fc':
            self.k = nn.Linear(hidden_dim, ffn_dim)
            self.v = nn.Linear(ffn_dim, hidden_dim)
        elif self.fc_or_conv == 'conv':
            self.k = nn.Conv1d(in_channels=hidden_dim, out_channels=ffn_dim, kernel_size=1)
            self.v = nn.Conv1d(in_channels=ffn_dim, out_channels=hidden_dim, kernel_size=1)
        else:
            raise ValueError('fc_or_conv must in ["fc", "conv"] but %s'%fc_or_conv)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q):
        if self.fc_or_conv == 'fc':
            attn = self.k(q)
            attn = self.dropout(self.act(attn))
            o = self.v(attn)
        elif self.fc_or_conv == 'conv':
            attn = self.k(q.transpose(-2, -1))
            attn = self.dropout(self.act(attn))
            o = self.v(attn).transpose(-2, -1)
        else:
            raise ValueError('fc_or_conv must in ["fc", "conv"] but %s'% self.fc_or_conv)

        return o

class RC(nn.Module):
    ''' Residual Connection : Add & Norm
    '''
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(RC, self).__init__()
        self.LN = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, output):
        return self.LN(query + self.dropout(output))

class Encoder(nn.Module):

    class Layer(nn.Module):
        def __init__(self, hidden_dim, head_num, ffn_dim):
            super(Encoder.Layer, self).__init__()
            self.mha_self = MHA(hidden_dim, head_num)
            self.rc1 = RC(hidden_dim)
            self.ffn = FFN(hidden_dim, ffn_dim)
            self.rc2 = RC(hidden_dim)

        def forward(self, x, enc_mask):
            '''
            x → [batch_size, seq_len, hidden_dim]
            enc_mask → [batch_size, seq_len, seq_len]
            o → [batch_size, seq_len, hidden_dim]
            '''
            q = k = v = x
            x = self.rc1(x, self.mha_self(q, k, v, enc_mask))
            o = self.rc2(x, self.ffn(x))
            return o

    def __init__(self, layer_num, hidden_dim, head_num, ffn_dim):
        super(Encoder, self).__init__()
        self.layer_modules = nn.ModuleList(
            [ Encoder.Layer(hidden_dim, head_num, ffn_dim) for _ in range(layer_num) ]
        )

    def forward(self, x, enc_mask):
        for layer_module in self.layer_modules:
            x = layer_module(x, enc_mask)
        return x

class Decoder(nn.Module):

    class Layer(nn.Module):
        def __init__(self, hidden_dim, head_num, ffn_dim):
            super(Decoder.Layer, self).__init__()
            self.mha_self = MHA(hidden_dim, head_num)
            self.rc1 = RC(hidden_dim)
            self.mha_cross = MHA(hidden_dim, head_num)
            self.rc2 = RC(hidden_dim)
            self.ffn = FFN(hidden_dim, ffn_dim)
            self.rc3 = RC(hidden_dim)

        def forward(self, enc_mem, x, dec_self_mask, dec_cross_mask):
            '''
            mem → [batch_size, enc_seq_len, hidden_dim]
            x → [batch_size, dec_seq_len, hidden_dim]
            dec_mask → [batch_size, dec_seq_len, dec_seq_len]
            enc_mask → [batch_size, enc_seq_len, enc_seq_len]
            o → [batch_size, dec_seq_len, hidden_dim]
            '''
            q = k = v = x
            x = self.rc1(x, self.mha_self(q, k, v, dec_self_mask))
            x = self.rc2(x, self.mha_cross(x, enc_mem, enc_mem, dec_cross_mask))
            o = self.rc3(x, self.ffn(x))
            return o

    def __init__(self, layer_num, hidden_dim, head_num, ffn_dim):
        super(Decoder, self).__init__()
        self.layer_num = layer_num
        self.layer_modules = nn.ModuleList(
            [ Decoder.Layer(hidden_dim, head_num, ffn_dim) for _ in range(layer_num) ]
        )

    def forward(self, encoder_mem, x, dec_self_mask, dec_cross_mask):
        for layer_module in self.layer_modules:
            x = layer_module(encoder_mem, x, dec_self_mask, dec_cross_mask)
        return x

class VanillaTransformer(nn.Module):
    def __init__(
            self,
            enc_num_embeddings,
            dec_num_embeddings,
            max_len,
            embedding_dim,
            hidden_dim,
            head_num,
            ffn_dim,
            enc_layer_num,
            dec_layer_num,
            share_emb=False
    ):
        super(VanillaTransformer, self).__init__()
        self.enc_emb = EMB(enc_num_embeddings, embedding_dim, max_len=max_len)
        if share_emb:
            self.dec_emb = self.enc_emb
        else:
            self.dec_emb = EMB(dec_num_embeddings, embedding_dim, max_len=max_len)
        self.enc = Encoder(enc_layer_num, hidden_dim, head_num, ffn_dim)
        self.dec = Decoder(dec_layer_num, hidden_dim, head_num, ffn_dim)
        self.gen = nn.Linear(hidden_dim, dec_num_embeddings)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_x, dec_x, enc_self_mask, dec_self_mask, dec_cross_mask):
        o_enc = self.encoder(enc_x, enc_self_mask)
        o_dec = self.decoder(o_enc, dec_x, dec_self_mask, dec_cross_mask)
        o_output = self.output(o_dec)
        return o_output

    def encoder(self, enc_x, enc_self_mask):
        o_emb = self.enc_emb(enc_x)
        return self.enc(o_emb, enc_self_mask)

    def decoder(self, o_enc, dec_x, dec_self_mask, dec_cross_mask):
        o_emb = self.dec_emb(dec_x)
        return self.dec(o_enc, o_emb, dec_self_mask, dec_cross_mask)

    def output(self, o_dec):
        return F.log_softmax(self.gen(o_dec), dim=-1)

class Loss(nn.Module):

    def __init__(self, padding_idx, smoothing_confidence=None):
        super(Loss, self).__init__()
        self.cret_kld = F.kl_div
        self.padding_idx = padding_idx
        self.smoothing_confidence = smoothing_confidence

    def forward(self, y_pred, y_true):
        '''
        Input tensor shape
            y_pred → [batch_size, seq_len, vocab]
            y_true → [batch_size, seq_len]

        Tensor before calculate loss
            y_pred → [batch_size*seq_len, vocab]
            y_true → [batch_size*seq_len, vocab]
        '''
        non_padding_sum = (y_true != self.padding_idx).sum().item()
        label_one, label_zero = 1., 0.
        if self.smoothing_confidence:
            cls_size = y_pred.size(-1)
            label_one = self.smoothing_confidence
            label_zero = (1. - self.smoothing_confidence) / (cls_size - 2)
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        y_true = y_true.contiguous().view(-1)
        y_target = torch.empty_like(y_pred)\
            .fill_(label_zero)\
            .scatter_(1, y_true.unsqueeze(-1), label_one)\
            .type_as(y_pred.data)
        y_target[:, self.padding_idx] == 0
        target_mask = torch.nonzero(y_true==self.padding_idx)
        y_target.index_fill_(0, target_mask.squeeze(), 0.)
        loss = self.cret_kld(y_pred, y_target, reduction='sum')
        return loss, non_padding_sum

class MASK(object):
    '''
    two mask mechanisms:
        1. padding mask → batch sequence length padding
        2. casual mask → mask future step info in sequence
    '''
    @staticmethod
    def create_mask(
            q, k, q_padding_idx, k_padding_idx,
            device_idx, use_causal_mask=False, padding_method=1):
        '''
        q → [batch_size, q_seq_len]
        k → [batch_size, k_seq_len]
        '''
        bs = q.size(0)
        q_sl = q.size(1)
        k_sl = k.size(1)
        ## Padding Mask
        if padding_method == 1:
            k_padding_mask = k==k_padding_idx
            padding_mask = k_padding_mask.unsqueeze(1).expand(bs, q_sl, k_sl)
        elif padding_method == 2:
            q_padding_mask = q!=q_padding_idx
            k_padding_mask = k!=k_padding_idx
            padding_mask = torch.matmul(
                q_padding_mask.unsqueeze(-1).float(),
                k_padding_mask.unsqueeze(-1).transpose(-1, -2).float()
            )
            padding_mask = padding_mask==0.
        else:
            raise ValueError("padding method must in [1,2] but %s"%padding_method)
        mask = padding_mask.cuda(device_idx)
        ## Causal Mask
        if use_causal_mask:
            causal_mask = torch.triu(torch.ones(q_sl, k_sl), diagonal=1)==1
            mask = mask | causal_mask.unsqueeze(0).cuda(device_idx)
        return mask.int()
