import torch
import torch.nn as nn


class SACNN(nn.Module):
    def __init__(self, num_classes, maxlen):
        super(SACNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1)

        # 计算全连接层输入维度（拼接后的通道数 * 序列长度）
        combined_channels = 32 + 64 + 128
        self.fc1 = nn.Linear(combined_channels * maxlen, 384)
        self.fc2 = nn.Linear(384, num_classes)

        # 自注意力层的embed_dim应与输入特征维度一致
        self.attention = nn.MultiheadAttention(embed_dim=combined_channels, num_heads=1)

    def forward(self, x):
        # 输入形状: (batch_size, sequence_length, 1)
        # 调整维度为Conv1d需要的格式：(batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # 并行卷积处理
        x1 = torch.relu(self.conv1(x))  # (batch, 32, maxlen)
        x2 = torch.relu(self.conv2(x))  # (batch, 64, maxlen)
        x3 = torch.relu(self.conv3(x))  # (batch, 128, maxlen)

        # 在通道维度拼接结果
        x = torch.cat((x1, x2, x3), dim=1)  # (batch, 32+64+128=224, maxlen)

        # 调整维度为自注意力需要的格式：(sequence_length, batch_size, embed_dim)
        x_attn = x.permute(2, 0, 1)
        
        # 自注意力处理
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)  # (maxlen, batch, 224)
        
        # 恢复维度并展平
        x = attn_output.permute(1, 0, 2)  # (batch, maxlen, 224)
        x = x.contiguous().view(x.size(0), -1)  # (batch, maxlen*224)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
        