import torch
import torch.nn as nn
from encoder import Encoder, EncoderLayer, ConvLayer
from attn import FullAttention, ProbAttention, AttentionLayer
from embed import DataEmbedding

class InformerForDDPM(nn.Module):
    def __init__(self, enc_in, seq_len, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                 dropout=0.0, attn='prob', activation='gelu', 
                 output_attention=False, distil=True, device=torch.device('cuda:0')):
        super(InformerForDDPM, self).__init__()
        self.attn = attn
        self.output_attention = output_attention
        self.seq_len = seq_len

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, 5, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(1)  # 只使用一层卷积层
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, enc_in, bias=True)
        
    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        # print(f'After embedding: {enc_out.shape}')  # 添加调试信息
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(f'After encoder: {enc_out.shape}')  # 添加调试信息
        
        enc_out = self.projection(enc_out)
        # print(f'After projection: {enc_out.shape}')  # 添加调试信息
        
        enc_out = enc_out[:, :self.seq_len, :]  # 调整输出形状
        # print(f'After shape adjustment: {enc_out.shape}')  # 添加调试信息
        
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out
