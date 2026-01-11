"""
深度学习模型定义
LSTM和Transformer模型用于网球比赛预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMModel(nn.Module):
    """
    LSTM模型：LSTM + Dropout + Dense
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)
        
        return output

class TransformerModel(nn.Module):
    """
    Transformer模型用于对比
    """
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # 投影到d_model维度
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出
        last_output = x[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)
        
        return output

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def create_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    创建模型
    
    Args:
        model_type: 模型类型 ('lstm', 'transformer')
        input_dim: 输入特征维度
        **kwargs: 模型超参数
        
    Returns:
        模型实例
    """
    if model_type.lower() == 'lstm':
        return LSTMModel(input_dim, **kwargs)
    elif model_type.lower() == 'transformer':
        return TransformerModel(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    seq_len = 10
    input_dim = 20
    
    # 测试LSTM
    lstm_model = LSTMModel(input_dim, hidden_dim=128, num_layers=2)
    x = torch.randn(batch_size, seq_len, input_dim)
    output = lstm_model(x)
    print(f"LSTM输出形状: {output.shape}")
    
    # 测试Transformer
    transformer_model = TransformerModel(input_dim, d_model=128, nhead=8, num_layers=2)
    output = transformer_model(x)
    print(f"Transformer输出形状: {output.shape}")
