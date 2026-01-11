"""
数据预处理模块
处理原始网球比赛数据，清洗和准备数据用于后续分析
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings

# 抑制pandas的SettingWithCopyWarning（已通过使用copy()避免）
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        加载的DataFrame
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗数据：处理缺失值、异常值等
    
    Args:
        df: 原始数据
        
    Returns:
        清洗后的数据
    """
    df_clean = df.copy()
    
    # 处理时间格式
    if 'elapsed_time' in df_clean.columns:
        df_clean['elapsed_time'] = pd.to_datetime(df_clean['elapsed_time'], format='%H:%M:%S', errors='coerce')
    
    # 处理数值型缺失值（使用copy避免警告）
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            # 对于距离、速度等特征，用0填充
            if 'distance' in col.lower() or 'speed' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            # 对于其他数值特征，用中位数填充
            else:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # 处理分类特征缺失值
    categorical_cols = ['serve_width', 'serve_depth', 'return_depth', 'winner_shot_type']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # 确保关键列没有缺失值
    essential_cols = ['match_id', 'point_victor', 'p1_points_won', 'p2_points_won']
    for col in essential_cols:
        if col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def get_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取每场比赛的最终结果
    
    Args:
        df: 数据
        
    Returns:
        包含比赛结果的DataFrame
    """
    match_results = []
    
    for match_id in df['match_id'].unique():
        match_data = df[df['match_id'] == match_id].copy()
        
        # 获取最后一行数据（比赛结束时的状态）
        last_row = match_data.iloc[-1]
        
        # 确定获胜者
        p1_final_points = last_row['p1_points_won']
        p2_final_points = last_row['p2_points_won']
        
        if p1_final_points > p2_final_points:
            winner = 1
        elif p2_final_points > p1_final_points:
            winner = 2
        else:
            # 如果点数相同，查看盘数
            if last_row['p1_sets'] > last_row['p2_sets']:
                winner = 1
            elif last_row['p2_sets'] > last_row['p1_sets']:
                winner = 2
            else:
                winner = None  # 平局（理论上不应该发生）
        
        match_results.append({
            'match_id': match_id,
            'player1': last_row['player1'],
            'player2': last_row['player2'],
            'p1_final_points': p1_final_points,
            'p2_final_points': p2_final_points,
            'p1_final_sets': last_row['p1_sets'],
            'p2_final_sets': last_row['p2_sets'],
            'winner': winner,
            'total_points': p1_final_points + p2_final_points
        })
    
    return pd.DataFrame(match_results)

def prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    完整的数据预处理流程
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        (清洗后的数据, 比赛结果)
    """
    print("正在加载数据...")
    df = load_data(file_path)
    print(f"原始数据形状: {df.shape}")
    
    print("正在清洗数据...")
    df_clean = clean_data(df)
    print(f"清洗后数据形状: {df_clean.shape}")
    
    print("正在提取比赛结果...")
    match_results = get_match_results(df_clean)
    print(f"比赛数量: {len(match_results)}")
    
    return df_clean, match_results

if __name__ == "__main__":
    # 测试数据预处理
    file_path = "选题三_Data/2024_Wimbledon_featured_matches.csv"
    df_clean, match_results = prepare_data(file_path)
    
    print("\n数据预处理完成！")
    print("\n比赛结果统计:")
    print(match_results['winner'].value_counts())
    print("\n前5场比赛结果:")
    print(match_results.head())
