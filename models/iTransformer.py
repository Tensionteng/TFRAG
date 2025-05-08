import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.dist_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=False)
            self.action_mean = nn.Sequential(
                nn.Linear(configs.pred_len, configs.pred_len, bias=True),
                # nn.Tanh(),
            )
            self.action_logstd = nn.Sequential(
                nn.Linear(configs.pred_len, configs.pred_len, bias=True),
                nn.Softplus(),
            )
            self.action_mean_cat = nn.Sequential(
                nn.Linear(configs.pred_len * 2, configs.pred_len, bias=True),
                nn.Tanh(),
                nn.Linear(configs.pred_len, configs.pred_len, bias=True),
                nn.Tanh(),
            )
            self.action_logstd_cat = nn.Sequential(
                nn.Linear(configs.pred_len * 2, configs.pred_len, bias=False),
                nn.Softplus(),
            )
            # 均值初始均值为output和gt之间的差值
            # self.action_mean[0].weight = nn.Parameter(torch.eye(configs.pred_len))
            # self.action_mean[0].bias = nn.Parameter(torch.zeros(configs.pred_len))
            # self.action_mean_cat[0].weight = nn.Parameter(
            #     torch.cat(
            #         [
            #             torch.eye(configs.pred_len) * -1.0,
            #             torch.eye(configs.pred_len),
            #         ],
            #         dim=1,
            #     )
            # )
            # 初始标准差不能太大
            # self.action_logstd_cat = nn.Parameter(
            #     torch.zeros(1, configs.pred_len, configs.enc_in) - 1.0
            # )

        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.enc_in, configs.num_class
            )

    def get_dist(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]

        # x_enc = self.encode_out
        action_mean = self.action_mean(x_enc).permute(0, 2, 1)[:, :, :N]
        action_logstd = self.action_logstd(x_enc).permute(0, 2, 1)[:, :, :N]
        action_logstd = torch.clamp(action_logstd, min=-5, max=-1)
        action_std = torch.exp(action_logstd)  # [B, L, D]
        # action_std = torch.clamp(action_std, min=0.1, max=10.0)

        return torch.distributions.Normal(action_mean, action_std)

    def get_dist_cat(self, x):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x = x / stdev
        _, _, N = x.shape
        x = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        action_mean = self.action_mean_cat(x).permute(0, 2, 1)[:, :, :N]
        action_logstd = self.action_logstd_cat(x).permute(0, 2, 1)[:, :, :N]
        # action_logstd = torch.clamp(action_logstd, min=-5, max=-1)
        action_std = torch.exp(action_logstd)
        action_std = torch.clamp(action_std, max=0.3)

        # action_mean = action_mean * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # action_mean = action_mean + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return torch.distributions.Normal(action_mean, action_std)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        self.encode_out = enc_out

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
