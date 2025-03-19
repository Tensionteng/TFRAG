import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from models.Memory import MemoryBankWithRetrieval


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.fusion_mode = configs.fusion_mode
        
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

        # Decoder
        if self.task_name in [
            "long_term_forecast",
            "short_term_forecast",
            "imputation",
            "anomaly_detection",
        ]:
            self.projection = nn.Linear(
                configs.d_model,
                configs.pred_len if self.task_name != "imputation" else configs.seq_len,
                bias=True,
            )
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.enc_in, configs.num_class
            )

        # RAG components
        self.memory_bank = MemoryBankWithRetrieval(
            seq_len=configs.pred_len, feature_dim=configs.d_model, use_gpu=True
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        _, _, N = x_enc.shape

        # Original prediction
        enc_embed = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_embed, attn_mask=None)
        pred_seq = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # Update memory bank with current prediction
        self.memory_bank.update(pred_seq.detach())

        # RAG: Retrieve similar sequences
        similar_seqs = self.memory_bank.retrieve_similar(pred_seq, k=10)

        # Fuse sequences
        fused_seq = self.memory_bank.fuse_sequences(
            pred_seq, similar_seqs, fusion_mode=self.fusion_mode
        )

        # Return fused prediction
        return fused_seq
