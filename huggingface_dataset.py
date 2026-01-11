"""
Hugging Face Datasets集成
将预处理后的数据上传到Hugging Face Datasets
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import os

def create_huggingface_dataset(df: pd.DataFrame, sequences: np.ndarray, 
                               labels: np.ndarray, train_ratio: float = 0.7,
                               val_ratio: float = 0.15) -> DatasetDict:
    """
    创建Hugging Face数据集
    
    Args:
        df: 原始数据
        sequences: 时间序列数据
        labels: 标签数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        DatasetDict对象
    """
    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # 划分数据集
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # 创建数据集字典
    dataset_dict = {
        'train': Dataset.from_dict({
            'sequences': sequences[train_indices].tolist(),
            'labels': labels[train_indices].tolist()
        }),
        'validation': Dataset.from_dict({
            'sequences': sequences[val_indices].tolist(),
            'labels': labels[val_indices].tolist()
        }),
        'test': Dataset.from_dict({
            'sequences': sequences[test_indices].tolist(),
            'labels': labels[test_indices].tolist()
        })
    }
    
    return DatasetDict(dataset_dict)

def save_to_huggingface(dataset_dict: DatasetDict, dataset_name: str,
                       username: str = None, private: bool = False):
    """
    保存数据集到Hugging Face Hub
    
    Args:
        dataset_dict: 数据集字典
        dataset_name: 数据集名称
        username: Hugging Face用户名（可选）
        private: 是否私有
    """
    if username:
        full_name = f"{username}/{dataset_name}"
    else:
        full_name = dataset_name
    
    print(f"正在上传数据集到 Hugging Face: {full_name}")
    dataset_dict.push_to_hub(full_name, private=private)
    print(f"数据集已上传: https://huggingface.co/datasets/{full_name}")

def load_from_huggingface(dataset_name: str) -> DatasetDict:
    """
    从Hugging Face Hub加载数据集
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        DatasetDict对象
    """
    from datasets import load_dataset
    dataset_dict = load_dataset(dataset_name)
    return dataset_dict

def save_preprocessed_data(df: pd.DataFrame, output_dir: str = "processed_data"):
    """
    保存预处理后的数据到本地（CSV/Parquet格式）
    
    Args:
        df: 预处理后的数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, "processed_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存为CSV: {csv_path}")
    
    # 保存为Parquet
    parquet_path = os.path.join(output_dir, "processed_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"数据已保存为Parquet: {parquet_path}")
    
    # 创建数据集描述文档
    description = f"""
# 网球比赛动量分析数据集

## 数据集描述

本数据集包含2024年温布尔登网球公开赛的比赛数据，经过预处理和特征工程。

## 数据统计

- 总样本数: {len(df)}
- 比赛数: {df['match_id'].nunique() if 'match_id' in df.columns else 'N/A'}
- 特征数: {len(df.columns)}

## 特征说明

主要特征包括：
- 比分特征：p1_points_won, p2_points_won, p1_sets, p2_sets等
- 动量特征：score_diff, p1_recent_win_rate, p1_streak等
- 技术统计：p1_ace, p2_ace, p1_winner, p2_winner等
- 体能特征：p1_distance_run, p2_distance_run等

## 数据格式

- CSV格式：使用UTF-8编码
- Parquet格式：使用Parquet格式存储，支持高效读取

## 使用说明

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('processed_data.csv')

# 读取Parquet
df = pd.read_parquet('processed_data.parquet')
```

## 引用

如果使用本数据集，请引用：
- 数据来源：2024年温布尔登网球公开赛
- 处理时间：2024年
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(description)
    print(f"数据集描述文档已保存: {readme_path}")

if __name__ == "__main__":
    from data_preprocessing import prepare_data
    from time_series_data import create_momentum_features, build_all_sequences
    
    # 加载和预处理数据
    file_path = "选题三_Data/2024_Wimbledon_featured_matches.csv"
    df_clean, _ = prepare_data(file_path)
    
    # 创建动量特征
    df_momentum = create_momentum_features(df_clean)
    
    # 构建时间序列
    sequences, labels = build_all_sequences(df_momentum, sequence_length=10)
    
    # 保存预处理后的数据
    save_preprocessed_data(df_momentum, "processed_data")
    
    # 创建Hugging Face数据集（本地）
    dataset_dict = create_huggingface_dataset(df_momentum, sequences, labels)
    print(f"\n数据集创建完成:")
    print(f"  训练集: {len(dataset_dict['train'])}")
    print(f"  验证集: {len(dataset_dict['validation'])}")
    print(f"  测试集: {len(dataset_dict['test'])}")
    
    # 如需上传到Hugging Face Hub，取消下面的注释
    # save_to_huggingface(dataset_dict, "tennis-momentum-2024", username="your_username")
