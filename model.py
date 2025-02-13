# model.py 修改后的模型
import torch 
import torch.nn as nn

class SACNN(nn.Module):
    def __init__(self, num_classes, maxlen, attention_heads=4, dropout_rate=0.5):
        super(SACNN, self).__init__()
        
        # 卷积层添加BatchNorm
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding=2),  # 修改kernel_size增加多样性
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        combined_channels = 32 + 64 + 128
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=combined_channels,
            num_heads=attention_heads,
            dropout=0.3
        )
        
        # 添加全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(combined_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, 1)
        x = x.permute(0, 2, 1)  # -> (batch, 1, seq_len)
        
        # 并行卷积
        x1 = self.conv1(x)  # (batch, 32, seq_len)
        x2 = self.conv2(x)  # (batch, 64, seq_len)
        x3 = self.conv3(x)  # (batch, 128, seq_len)
        
        # 通道拼接
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 224, seq_len)
        
        # 自注意力需要(seq_len, batch, features)
        x_attn = x.permute(2, 0, 1)  # (seq_len, batch, 224)
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        
        # 恢复形状并池化
        x = attn_out.permute(1, 2, 0)  # (batch, 224, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, 224)
        
        return self.classifier(x)