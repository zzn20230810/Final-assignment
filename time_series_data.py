"""
时间序列数据构建模块
构建逐分得失序列用于LSTM/Transformer模型训练
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# 抑制pandas的SettingWithCopyWarning（已通过使用copy()避免）
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

class TennisSequenceDataset(Dataset):
    """网球比赛时间序列数据集"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: 时间序列数据 (N, seq_len, feature_dim)
            labels: 标签数据 (N,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def build_point_sequence(df: pd.DataFrame, match_id: str, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    为单场比赛构建时间序列数据
    
    Args:
        df: 比赛数据
        match_id: 比赛ID
        sequence_length: 序列长度
        
    Returns:
        (sequences, labels): 序列数据和标签
    """
    match_data = df[df['match_id'] == match_id].copy()
    match_data = match_data.sort_values('point_no').reset_index(drop=True)
    
    sequences = []
    labels = []
    
    # 构建特征向量（每一分的特征）
    # 注意：所有特征都是"当前点"的特征，但我们在构建序列时只使用历史数据
    # p1_points_won, p2_points_won: 累积得分（到当前点为止）
    # p1_ace, p2_ace等: 当前点的事件（0或1）
    # 使用iloc[i-1]获取这些特征时，得到的是时刻i-1的特征，不包含时刻i的信息
    feature_cols = [
        'p1_points_won', 'p2_points_won',  # 累积得分（到当前点为止）
        'p1_sets', 'p2_sets',              # 累积盘数
        'p1_games', 'p2_games',            # 当前盘的累积局数
        'server', 'serve_no',              # 当前点的发球信息
        'p1_ace', 'p2_ace',                # 当前点是否有Ace
        'p1_winner', 'p2_winner',          # 当前点是否有制胜分
        'p1_unf_err', 'p2_unf_err',        # 当前点是否有非受迫性失误
        'p1_break_pt', 'p2_break_pt',      # 当前点是否是破发点机会
        'rally_count', 'speed_mph',        # 当前点的回合数和发球速度
        'p1_distance_run', 'p2_distance_run'  # 当前点的跑动距离
    ]
    
    # 只保留存在的列
    available_cols = [col for col in feature_cols if col in match_data.columns]
    
    # 填充缺失值（使用copy避免SettingWithCopyWarning）
    match_data = match_data.copy()
    for col in available_cols:
        match_data[col] = match_data[col].fillna(0)
    
    # 归一化数值特征
    numeric_cols = match_data[available_cols].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if match_data[col].std() > 0:
            mean_val = match_data[col].mean()
            std_val = match_data[col].std()
            match_data[col] = (match_data[col] - mean_val) / std_val
    
    # 构建序列
    # 重要：只使用历史数据预测未来，避免数据泄露
    # 使用时刻 i-sequence_length 到 i-1 的特征预测时刻 i 的结果
    for i in range(sequence_length, len(match_data)):
        # 获取序列窗口：使用历史 sequence_length 个时间步的特征
        # iloc[i-sequence_length:i] 表示从 i-sequence_length 到 i-1（不包括i）
        sequence = match_data[available_cols].iloc[i-sequence_length:i].values
        
        # 标签：时刻 i 的获胜者 (1: player1, 2: player2)
        # 注意：这里使用时刻 i 的结果作为标签，特征只用到时刻 i-1
        label = match_data.iloc[i]['point_victor'] - 1  # 转换为0/1
        
        sequences.append(sequence)
        labels.append(label)
    
    if len(sequences) == 0:
        return np.array([]), np.array([])
    
    return np.array(sequences), np.array(labels)

def build_all_sequences(df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    为所有比赛构建时间序列数据
    
    Args:
        df: 所有比赛数据
        sequence_length: 序列长度
        
    Returns:
        (sequences, labels): 所有序列数据和标签
    """
    all_sequences = []
    all_labels = []
    
    for match_id in df['match_id'].unique():
        seq, lab = build_point_sequence(df, match_id, sequence_length)
        if len(seq) > 0:
            all_sequences.append(seq)
            all_labels.append(lab)
    
    if len(all_sequences) == 0:
        return np.array([]), np.array([])
    
    # 合并所有序列
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return sequences, labels

def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建动量特征用于时间序列
    
    Args:
        df: 原始数据
        
    Returns:
        添加了动量特征的数据
    """
    df_momentum = df.copy()
    df_momentum = df_momentum.sort_values(['match_id', 'point_no']).reset_index(drop=True)
    
    # 为每个比赛计算动量特征
    for match_id in df_momentum['match_id'].unique():
        match_mask = df_momentum['match_id'] == match_id
        match_data = df_momentum[match_mask].copy()
        match_indices = match_data.index.tolist()
        
        # 得分差距（使用copy避免警告）
        score_diff = (match_data['p1_points_won'] - match_data['p2_points_won']).values
        df_momentum.loc[match_indices, 'score_diff'] = score_diff
        
        # 盘数差距
        set_diff = (match_data['p1_sets'] - match_data['p2_sets']).values
        df_momentum.loc[match_indices, 'set_diff'] = set_diff
        
        # 局数差距
        game_diff = (match_data['p1_games'] - match_data['p2_games']).values
        df_momentum.loc[match_indices, 'game_diff'] = game_diff
        
        # 最近N点得分率（滑动窗口）
        window_size = 5
        p1_recent_rates = []
        p2_recent_rates = []
        
        for i in range(len(match_data)):
            start_idx = max(0, i - window_size + 1)
            # 使用copy()避免SettingWithCopyWarning
            window_data = match_data.iloc[start_idx:i+1].copy()
            
            p1_wins = (window_data['point_victor'] == 1).sum()
            total = len(window_data)
            
            if total > 0:
                p1_recent_rates.append(p1_wins / total)
                p2_recent_rates.append(1 - p1_wins / total)
            else:
                p1_recent_rates.append(0.5)
                p2_recent_rates.append(0.5)
        
        # 使用列表赋值避免警告
        df_momentum.loc[match_indices, 'p1_recent_win_rate'] = p1_recent_rates
        df_momentum.loc[match_indices, 'p2_recent_win_rate'] = p2_recent_rates
        
        # 连胜次数
        p1_streaks = []
        p2_streaks = []
        p1_streak = 0
        p2_streak = 0
        
        for i, idx in enumerate(match_indices):
            point_victor = match_data.iloc[i]['point_victor']
            if point_victor == 1:
                p1_streak += 1
                p2_streak = 0
            elif point_victor == 2:
                p2_streak += 1
                p1_streak = 0
            
            p1_streaks.append(p1_streak)
            p2_streaks.append(p2_streak)
        
        df_momentum.loc[match_indices, 'p1_streak'] = p1_streaks
        df_momentum.loc[match_indices, 'p2_streak'] = p2_streaks
    
    return df_momentum

def prepare_time_series_data(df: pd.DataFrame, sequence_length: int = 10,
                            train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    准备时间序列数据用于训练
    
    Args:
        df: 原始数据
        sequence_length: 序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (train_loader, val_loader, test_loader, feature_dim)
    """
    # 创建动量特征
    print("正在创建动量特征...")
    df_momentum = create_momentum_features(df)
    
    # 构建时间序列
    print("正在构建时间序列...")
    sequences, labels = build_all_sequences(df_momentum, sequence_length)
    
    if len(sequences) == 0:
        raise ValueError("无法构建时间序列数据，请检查数据")
    
    print(f"序列形状: {sequences.shape}, 标签形状: {labels.shape}")
    
    # 划分数据集
    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_sequences = sequences[train_indices]
    train_labels = labels[train_indices]
    val_sequences = sequences[val_indices]
    val_labels = labels[val_indices]
    test_sequences = sequences[test_indices]
    test_labels = labels[test_indices]
    
    # 创建数据集和数据加载器
    train_dataset = TennisSequenceDataset(train_sequences, train_labels)
    val_dataset = TennisSequenceDataset(val_sequences, val_labels)
    test_dataset = TennisSequenceDataset(test_sequences, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    feature_dim = sequences.shape[2]
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    print(f"特征维度: {feature_dim}")
    
    return train_loader, val_loader, test_loader, feature_dim

if __name__ == "__main__":
    from data_preprocessing import prepare_data
    
    # 测试时间序列构建
    file_path = "选题三_Data/2024_Wimbledon_featured_matches.csv"
    df_clean, _ = prepare_data(file_path)
    
    train_loader, val_loader, test_loader, feature_dim = prepare_time_series_data(
        df_clean, sequence_length=10
    )
    
    print("\n时间序列数据准备完成！")
