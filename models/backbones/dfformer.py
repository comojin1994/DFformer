import math
import torch
import torch.nn as nn
from einops import rearrange
from easydict import EasyDict
from typing import List, Tuple


def get_sincos_pos_embed(dim: int, seq_len: int, cls_token: bool = False):
    if cls_token:
        pe = torch.zeros(seq_len + 1, dim)
        position = torch.arange(0, seq_len + 1, dtype=torch.float).unsqueeze(1)
    else:
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    return pe


def calculate_output_size(input_size, layers):
    output_size = input_size
    for layer in layers:
        kernel_size, stride = layer[2], layer[3]
        output_size = math.floor((output_size - kernel_size) / stride) + 1
    return output_size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None
        self.value_gradients = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def save_value_gradients(self, value_gradients):
        self.value_gradients = value_gradients

    def get_value_gradients(self):
        return self.value_gradients

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        self.save_attention_map(attn)
        if register_hook:
            v.register_hook(self.save_value_gradients)
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-2, -1)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        layers: List[Tuple[int, int, int, int]],
        bias: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=kernel,
                        stride=stride,
                        bias=bias,
                    ),
                    TransposeLast(),
                    nn.LayerNorm(out_dim),
                    TransposeLast(),
                    nn.GELU(),
                )
                for (in_dim, out_dim, kernel, stride) in layers
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalSpatialEncoder(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout_rate: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        self.temporal_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.spatial_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_hook=False):
        B, C, D, T = x.shape

        # Temporal Block
        x = x.reshape(B * C, D, T)  # BC x D x T
        x = x.transpose(1, 2)  # BC x T x D
        x = self.temporal_block(x, register_hook=register_hook)
        x = x.reshape(B, C, T, D)  # B x C x T x D
        x = x.transpose(1, 2)  # B x T x C x D

        # Spatial Block
        x = x.reshape(B * T, C, D)  # BT x C x D
        x = self.spatial_block(x, register_hook=register_hook)
        x = x.reshape(B, T, C, D)  # B x T x C x D
        x = x.permute(0, 2, 3, 1)  # B x C x D x T

        return x


class Embedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        nhead,
        spatial_len,
        input_size,
        cnn_layers,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = calculate_output_size(input_size, cnn_layers)
        self.spatial_len = spatial_len

        self.patch_embed = PatchEmbed(cnn_layers)

        self.temporal_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.spatial_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len + 1, embed_dim), requires_grad=False
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.spatial_len + 1, self.embed_dim),
            requires_grad=False,
        )

        self.temporal_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.spatial_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        temporal_pos_embed = get_sincos_pos_embed(
            dim=self.embed_dim, seq_len=self.seq_len, cls_token=True
        )
        self.temporal_pos_embed.data.copy_(temporal_pos_embed)

        spatial_pos_embed = get_sincos_pos_embed(
            dim=self.embed_dim,
            seq_len=self.spatial_len,
            cls_token=True,
        )
        self.spatial_pos_embed.data.copy_(spatial_pos_embed)

        torch.nn.init.normal_(self.temporal_token, std=0.02)
        torch.nn.init.normal_(self.spatial_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_hook=False):
        B, C, D, T = x.shape

        # Path embedding
        x = x.reshape(B * C, D, T)  # BC x D x T
        x = self.patch_embed(x)

        # Temporal position embedding & block
        x = x.transpose(1, 2)  # BC x T x D
        x = x + self.temporal_pos_embed[:, 1:, :]
        token = self.temporal_token + self.temporal_pos_embed[:, :1, :]
        token = token.expand(B * C, -1, -1)
        x = torch.cat((token, x), dim=1)
        x = self.temporal_block(x, register_hook=register_hook)
        x = x.reshape(B, C, -1, self.embed_dim)  # B x C x T x D
        x = x.transpose(1, 2)  # B x T x C x D

        # Spatial position embedding & block
        B, T, C, D = x.shape
        x = x.reshape(B * T, C, D)  # BT x C x D
        x = x + self.spatial_pos_embed[:, 1:, :]
        token = self.spatial_token + self.spatial_pos_embed[:, :1, :]
        token = token.expand(B * T, -1, -1)
        x = torch.cat((token, x), dim=1)
        x = self.spatial_block(x, register_hook=register_hook)
        x = x.reshape(B, -1, C + 1, self.embed_dim)  # B x T x C x D
        x = x.permute(0, 2, 3, 1)  # B x C x D x T

        return x


class ClassifierHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        num_channels,
        seq_len,
        db_name,
        use_token,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.db_name = db_name
        self.embed_dim = embed_dim
        self.use_token = use_token

        self._init_mlp_head(embed_dim, num_classes, num_channels, seq_len, dropout_rate)

    def _init_mlp_head(
        self, embed_dim, num_classes, num_channels, seq_len, dropout_rate
    ):
        if self.use_token:
            self.mlp_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear((num_channels + 1 + seq_len) * embed_dim, num_classes),
            )
        else:
            if self.db_name == "BCIC2a" or self.db_name == "BCIC2b":
                self.mlp_head = nn.Sequential(
                    nn.Conv2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=(num_channels, 1),
                        bias=False,
                    ),
                    nn.ELU(),
                    nn.BatchNorm2d(embed_dim),
                    nn.Dropout2d(dropout_rate),
                    nn.Flatten(),
                    nn.Linear(seq_len * embed_dim, num_classes),
                )
            elif self.db_name == "SleepEDF" or self.db_name == "SHHS":
                self.mlp_head = nn.Sequential(
                    nn.BatchNorm1d(embed_dim),
                    nn.ELU(),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    nn.Dropout1d(dropout_rate),
                    nn.Flatten(),
                    nn.Linear(embed_dim, num_classes),
                )
            else:
                raise NotImplementedError


    def forward(self, x):
        if not self.use_token:
            B, C, D, T = x.shape
            if self.db_name == "BCIC2a" or self.db_name == "BCIC2b":
                x = x.transpose(1, 2)  # B x D x C x T
            elif self.db_name == "SleepEDF" or self.db_name == "SHHS":
                x = x.reshape(B * C, self.embed_dim, -1)  # BC x D x T
        else:
            if self.db_name == "SleepEDF" or self.db_name == "SHHS":
                B, C, D = x.shape
                x = x.reshape(B * C, self.embed_dim)

        x = self.mlp_head(x)
        return x


class DFformer(nn.Module):
    def __init__(self, args: EasyDict):
        super().__init__()

        self.embed_dim = args.dim
        self.use_token = args.use_token
        self.apply_cls_head = args.apply_cls_head

        self.embedding = Embedding(
            self.embed_dim,
            args.nhead,
            args.inter_information_length,
            args.origin_ival[-1],
            args.cnn_layers,
        )

        self.blocks = nn.ModuleList(
            [
                TemporalSpatialEncoder(self.embed_dim, args.nhead)
                for _ in range(args.nlayer)
            ]
        )

        if self.apply_cls_head:
            self.classifier_head = ClassifierHead(
                self.embed_dim,
                args.num_classes,
                args.inter_information_length,
                self.embedding.seq_len,
                args.db_name,
                args.use_token,
            )

    def forward(self, x, register_hook=False):
        x = self.embedding(x, register_hook=register_hook)

        for block in self.blocks:
            x = x + block(x, register_hook=register_hook)  # B x C x D x T

        if self.apply_cls_head:
            x = self.classifier_head(x[:, 1:, :, 1:])

        return x
