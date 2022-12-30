# encoding=utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vanilla_transformer import FFN, MHA
from ml_collections import ConfigDict


n2t_types = ConfigDict()
n2t_types.conv2d = 'conv2d'
n2t_types.qkv_w = 'qkv_w'
n2t_types.qkv_b ='qkv_b'
n2t_types.o_w = 'o_w'
n2t_types.o_b = 'o_b'
n2t_types.ffn_w = 'ffn_w'
n2t_types.cls_fc_w = 'cls_fc_w'
n2t_types.others = 'others'

def n2t(pretrained_model, k, n2t_type=n2t_types.others):
    '''
    From Google pretrained ViT on ImageNet21k to PyTorch Tenor
    '''
    tensor = torch.from_numpy(pretrained_model[k])
    if n2t_type==n2t_types.conv2d:
        tensor = tensor.permute(3, 2, 0, 1)
    elif n2t_type==n2t_types.qkv_w:
        tensor = tensor.view(tensor.shape[0], -1).transpose(0, 1)
    elif n2t_type==n2t_types.o_w:
        tensor =  tensor.view(tensor.shape[-1], -1).transpose(0, 1)
    elif n2t_type in (n2t_types.qkv_b, n2t_types.o_b):
        tensor = tensor.view(-1)
    elif n2t_type in (n2t_types.ffn_w, n2t_types.cls_fc_w):
        tensor = tensor.transpose(0, 1)
    return tensor

class EMB(nn.Module):
    def __init__(
            self, img_size, patch_size, channel_num, hidden_size, dropout_rate=0.0, pretrained_model=None):
        super(EMB, self).__init__()
        assert img_size%patch_size==0, \
            "image_size %s must be divisible by patch_dim %s"%(img_size, patch_size)
        self.patch_seq_len =  int(pow(img_size/patch_size, 2))
        self.hidden_size = hidden_size
        self.patch_val_emb = nn.Conv2d(
            in_channels=channel_num, out_channels=self.hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_val_emb = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_seq_len+1, self.hidden_size))
        self.dropout = nn.Dropout(p=dropout_rate)
        if pretrained_model:
            self.load_from_pretrained(pretrained_model)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        '''
        x → [batch_size, channel, height, width]
        o_emb → [batch_size, patch_seq_len+1, hidden_size]
        '''
        bs = x.shape[0]
        val_emb = torch.cat(
            [
                self.cls_val_emb.expand(bs, -1, -1),
                self.patch_val_emb(x).view(bs, self.hidden_size, -1).transpose(-2, -1)
            ],
            dim=1
        )
        pos_emb = self.pos_emb.expand(bs, -1, -1)
        o_emb = self.dropout(val_emb + pos_emb)
        return o_emb

    def load_from_pretrained(self, pretrained_model):
        with torch.no_grad():
            self.patch_val_emb.weight.copy_(n2t(pretrained_model, 'embedding/kernel', n2t_type=n2t_types.conv2d))
            self.patch_val_emb.bias.copy_(n2t(pretrained_model, 'embedding/bias'))
            self.cls_val_emb.copy_(n2t(pretrained_model, 'cls'))
            self.pos_emb.copy_(n2t(pretrained_model, 'Transformer/posembed_input/pos_embedding'))

class Encoder(nn.Module):

    class Layer(nn.Module):

        def __init__(self, hidden_size, head_num, ffn_hidden_size):
            super(Encoder.Layer, self).__init__()
            self.ln_prev_attn = nn.LayerNorm(hidden_size, eps=1e-6)
            self.attn = MHA(hidden_size, head_num, dropout_rate=0.0)
            self.ln_prev_ffn = nn.LayerNorm(hidden_size, eps=1e-6)
            self.ffn = FFN(hidden_size, ffn_hidden_size, dropout_rate=0.0)

        def forward(self, x):
            '''
            forward pass
            1. attn : x → ln(x) → attn(ln(x)) → attn(ln(x))+x
            2. ffn : x → ln(x) → ffn(ln(x)) → ffn(ln(x))+x
            '''
            # residual block1
            ori = x
            x = self.ln_prev_attn(x)
            o_attn = self.attn(x, x, x)
            x = o_attn + ori

            # residual block2
            ori = x
            x = self.ln_prev_ffn(x)
            o_ffn = self.ffn(x)
            o_out = o_ffn + ori
            return o_out

    def __init__(self, layer_num, hidden_size, head_num, ffn_hidden_size, pretrained_model=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.ffn_hidden_size = ffn_hidden_size
        self.layers = nn.ModuleList(
                [Encoder.Layer(self.hidden_size, self.head_num, self.ffn_hidden_size) for _ in range(layer_num)]
        )
        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)
        if pretrained_model:
            self.load_from_pretrained(pretrained_model)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        o_enc = self.ln(x)
        return o_enc

    def load_from_pretrained(self, pretrained_model):
        ## loader encoder layer parameters
        MHDPA = 'MultiHeadDotProductAttention'
        LN = 'LayerNorm'
        MLP = 'MlpBlock'
        with torch.no_grad():
            for idx, layer in enumerate(self.layers):
                PRE = 'Transformer/encoderblock_%s'%idx
                # ln_prev_attn
                layer.ln_prev_attn.weight.copy_(n2t(pretrained_model, '/'.join([PRE, LN+'_0', 'scale'])))
                layer.ln_prev_attn.bias.copy_(n2t(pretrained_model, '/'.join([PRE, LN+'_0', 'bias'])))
                # attn
                layer.attn.w_q.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/query', 'kernel']), n2t_type=n2t_types.qkv_w))
                layer.attn.w_q.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/query', 'bias']), n2t_type=n2t_types.qkv_b))
                layer.attn.w_k.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/key', 'kernel']), n2t_type=n2t_types.qkv_w))
                layer.attn.w_k.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/key', 'bias']), n2t_type=n2t_types.qkv_b))
                layer.attn.w_v.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/value', 'kernel']), n2t_type=n2t_types.qkv_w))
                layer.attn.w_v.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/value', 'bias']), n2t_type=n2t_types.qkv_b))
                layer.attn.w_o.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/out', 'kernel']), n2t_type=n2t_types.o_w))
                layer.attn.w_o.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MHDPA+'_1/out', 'bias']), n2t_type=n2t_types.o_b))
                # ln_prev_ffn
                layer.ln_prev_ffn.weight.copy_(n2t(pretrained_model, '/'.join([PRE, LN+'_2', 'scale'])))
                layer.ln_prev_ffn.bias.copy_(n2t(pretrained_model, '/'.join([PRE, LN+'_2', 'bias'])))
                # ffn
                layer.ffn.k.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MLP+'_3/Dense_0', 'kernel']), n2t_type=n2t_types.ffn_w))
                layer.ffn.k.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MLP+'_3/Dense_0', 'bias'])))
                layer.ffn.v.weight.copy_(n2t(pretrained_model, '/'.join([PRE, MLP+'_3/Dense_1', 'kernel']), n2t_type=n2t_types.ffn_w))
                layer.ffn.v.bias.copy_(n2t(pretrained_model, '/'.join([PRE, MLP+'_3/Dense_1', 'bias'])))
            ## load output ln parameters
            self.ln.weight.copy_(n2t(pretrained_model, 'Transformer/encoder_norm/scale'))
            self.ln.bias.copy_(n2t(pretrained_model, 'Transformer/encoder_norm/bias'))

class Head(nn.Module):

    def __init__(self, hidden_size, output_cls_num, pretrained_model=None):
        super(Head, self).__init__()
        self.head_fc = nn.Linear(hidden_size, output_cls_num)
        if pretrained_model:
            self.load_from_pretrained(pretrained_model)

    def forward(self, x):
        return self.head_fc(x)

    def load_from_pretrained(self, pretrained_model):
        with torch.no_grad():
            nn.init.zeros_(self.head_fc.weight)
            nn.init.zeros_(self.head_fc.bias)

class Loss(nn.Module):

    def __init__(self, smoothing_confidence=None, num_cls=1000):
        super(Loss, self).__init__()
        self.smoothing_confidence = smoothing_confidence
        if self.smoothing_confidence:
            self.cret = F.kl_div
            self.label_one = self.smoothing_confidence
            self.label_zero = (1. - self.smoothing_confidence) / (num_cls - 1)
        else:
            self.cret = F.cross_entropy

    def forward(self, y_pred, y_true):
        if self.smoothing_confidence:
            y_pred = F.log_softmax(y_pred, dim=-1).contiguous().view(-1, y_pred.shape[-1])
            y_true = y_true.contiguous().view(-1)
            y_target = torch.empty_like(y_pred)\
                .fill_(self.label_zero)\
                .scatter_(1, y_true.unsqueeze(-1), self.label_one)\
                .type_as(y_pred.data)
            loss = self.cret(y_pred, y_target, reduction='batchmean')
        else:
            loss = self.cret(y_pred, y_true)
        return loss

class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size, patch_size, channel_num,
                 hidden_size, layer_num, head_num, ffn_hidden_size,
                 output_cls_num,
                 pretrained_model=None):
        super(VisionTransformer, self).__init__()
        self.emb = EMB(
            img_size, patch_size, channel_num, hidden_size, pretrained_model=pretrained_model)
        self.enc = Encoder(
            layer_num, hidden_size, head_num, ffn_hidden_size, pretrained_model=pretrained_model)
        self.head = Head(
            hidden_size, output_cls_num, pretrained_model=pretrained_model)

    def forward(self, x):
        o_emb = self.emb(x)
        o_enc = self.enc(o_emb)
        o_head = self.head(o_enc[:,0,:])
        return o_head

if __name__ == '__main__':
    args = ConfigDict()
    args.layer_num = 12
    args.img_size = 224
    args.batch_size = 8
    args.patch_size = 16
    args.output_cls_num = 10
    args.hidden_size = 768
    args.head_num = 12
    args.ffn_hidden_size = 768 * 4

    vit_model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        channel_num=3,
        hidden_size=args.hidden_size,
        layer_num=args.layer_num,
        head_num=args.head_num,
        ffn_hidden_size=args.ffn_hidden_size,
        output_cls_num=args.output_cls_num,
        pretrained_model=None
    )
