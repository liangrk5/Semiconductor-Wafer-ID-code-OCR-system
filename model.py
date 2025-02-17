import torch
import torch.nn as nn
import math

class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activation='relu'):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.activation = self.get_activation(activation)
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            self.activation
        )

        self.pointwise = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            self.activation
        )

    def get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),  # 修正拼写错误
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        if isinstance(name, nn.Module):
            return name
        name = name.lower()
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}.")
        return activations[name]
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)

class SACNN(nn.Module):
    def __init__(self, 
                 num_classes, 
                 max_len, 
                 attention_heads, 
                 activation='relu',
                 # 新增可配置参数
                 branch_channels=[32, 64, 128],
                 branch_kernel_sizes=[3, 5, 7],
                 branch_pools=[True, True, True],
                 use_positional_encoding=True,
                 transformer_layers=2,
                 transformer_dim_feedforward=512,
                 transformer_dropout=0.1,
                 transformer_activation='gelu',
                 adaptive_pool_size='auto',
                 classifier_hidden_dims=[512, 256],
                 classifier_dropout=0.0,
                 use_layer_norm=True):
        super().__init__()

        # 参数校验
        assert len(branch_channels) == len(branch_kernel_sizes) == len(branch_pools), \
            "分支参数长度必须一致"

        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList()
        for ch, ksize, pool in zip(branch_channels, branch_kernel_sizes, branch_pools):
            layers = [
                DepthwiseConvBlock(1, ch, ksize, activation=activation),
                nn.MaxPool1d(2) if pool else nn.Identity(),
                DepthwiseConvBlock(ch, ch, ksize, activation=activation)
            ]
            self.conv_branches.append(nn.Sequential(*layers))

        # 特征维度计算
        combined_dim = sum(branch_channels)
        
        # 自适应池化
        if adaptive_pool_size == 'auto':
            self.adaptive_pool_size = math.ceil(max_len / 4)  # 两次池化后的长度
        else:
            self.adaptive_pool_size = adaptive_pool_size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.adaptive_pool_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(combined_dim, self.adaptive_pool_size) \
            if use_positional_encoding else nn.Identity()

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=combined_dim,
                nhead=attention_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                activation=transformer_activation,
                batch_first=True
            ),
            num_layers=transformer_layers
        )

        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 分类器
        classifier_layers = []
        input_dim = combined_dim * 2
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                classifier_layers.append(nn.LayerNorm(hidden_dim))
            classifier_layers.append(nn.ReLU())
            if classifier_dropout > 0:
                classifier_layers.append(nn.Dropout(classifier_dropout))
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        # 输入形状调整
        x = x.permute(0, 2, 1)  # [B, 1, L]

        # 多尺度特征提取
        branch_features = []
        for branch in self.conv_branches:
            feat = branch(x)
            feat = self.adaptive_pool(feat)
            branch_features.append(feat)
        x = torch.cat(branch_features, dim=1)  # [B, C, L']

        # Transformer处理
        x = x.permute(2, 0, 1)  # [L', B, C]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # [B, C, L']

        # 双池化
        avg = self.avg_pool(x).squeeze(-1)
        max_ = self.max_pool(x).squeeze(-1)
        combined = torch.cat([avg, max_], dim=1)

        # 分类
        return self.classifier(combined)