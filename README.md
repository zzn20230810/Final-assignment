# 网球比赛动量分析与结果预测

## 项目简介

本项目基于2024年温布尔登网球公开赛的比赛数据，使用深度学习模型（LSTM、Transformer）进行网球比赛的动量分析和结果预测。项目实现了完整的时间序列建模流程，包括数据预处理、特征工程、模型训练、评估和可视化。

**注意**：本项目使用PyTorch实现LSTM和Transformer模型。

## 研究问题

1. **动量分析**：如何量化比赛中的动量变化？哪些因素影响比赛动量？
2. **结果预测**：能否基于比赛过程中的动量特征预测最终结果？
3. **关键因素**：哪些特征对比赛结果影响最大？

## 研究任务

1. **数据预处理**：清洗和处理原始比赛数据
2. **动量特征工程**：计算各种动量指标（得分差距、连胜、关键分等）
3. **模型训练**：使用多种机器学习模型进行预测
4. **模型评估**：通过交叉验证和多种指标评估模型性能
5. **可视化分析**：生成动量轨迹、特征重要性等可视化结果

## 技术实现

### 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- wandb (实验记录)
- datasets, huggingface-hub (数据集管理)
- matplotlib, seaborn (可视化)

### 项目结构

```
.
├── main_deep.py                      # 主程序
├── data_preprocessing.py             # 数据预处理模块
├── time_series_data.py               # 时间序列数据构建模块
├── models.py                         # 模型定义（LSTM/Transformer）
├── train.py                          # 模型训练模块
├── visualization_deep.py             # 可视化模块
├── huggingface_dataset.py            # Hugging Face数据集集成
├── test_models.py                    # 单元测试
├── tennis_momentum_analysis.ipynb     # Jupyter Notebook
├── README.md                          # 项目说明
├── 实验报告_MCM格式.md                # MCM格式实验报告
├── requirements.txt                  # 依赖包列表
├── 选题三_Data/                      # 数据目录
│   ├── 2024_data_dictionary.csv     # 数据字典
│   └── 2024_Wimbledon_featured_matches.csv  # 比赛数据
├── results/                          # 结果目录（运行后生成）
└── visualizations/                   # 可视化图表（运行后生成）
```

### 运行方式

**方法1：运行深度学习版本主程序**
```bash
python main_deep.py
```

**方法2：使用Jupyter Notebook**
```bash
jupyter notebook tennis_momentum_analysis.ipynb
```


### 主要功能模块

#### 1. 数据预处理 (`data_preprocessing.py`)
- 加载和清洗数据
- 处理缺失值和异常值
- 提取比赛结果

#### 2. 动量特征工程 (`momentum_features.py`)
- **得分差距**：当前点数、盘数、局数差距
- **近期表现**：最近N个点的得分率
- **连胜次数**：连续得分次数
- **关键分**：破发点机会和成功率
- **发球优势**：发球局得分率
- **技术统计**：Ace、制胜分、失误等累积统计
- **体能指标**：跑动距离等

#### 3. 模型训练 (`train.py`)
- **时间序列预测**：使用历史N个点预测下一个点的胜负
- **模型类型**：
  - LSTM (长短期记忆网络) - 参考模型
  - Transformer (注意力机制)
- **评估指标**：MAE、RMSE、F1-score、AUC、准确率
- **实验记录**：使用WandB记录所有实验

#### 4. 可视化 (`visualization.py`)
- 比赛动量轨迹
- 模型性能对比
- 特征重要性分析
- 混淆矩阵
- 比赛统计对比

## 实验结果

运行程序后，会在 `results/` 和 `visualizations/` 目录下生成：

1. **模型结果**：
   - `model_summary.json`：模型性能摘要
   - `point_feature_importance.csv`：点级别特征重要性
   - `match_feature_importance.csv`：比赛级别特征重要性

2. **可视化图表**：
   - 动量轨迹图
   - 模型性能对比图
   - 特征重要性图
   - 混淆矩阵
   - 比赛统计对比图

## 参考模型

### LSTM模型结构（参考模型）

```
LSTM + Dropout + Dense
- 输入：时间序列数据 (seq_len, feature_dim)
- LSTM层：hidden_dim=128, num_layers=2
- Dropout：0.3
- 输出：预测概率 (num_classes=2)
```

### 模型对比

- **LSTM**：长短期记忆，适合捕捉长期依赖
- **Transformer**：注意力机制，捕捉全局依赖

## 技术特点

1. **时间序列建模**：构建逐分得失序列，利用历史信息预测
2. **深度学习**：使用PyTorch实现LSTM和Transformer模型
3. **实验管理**：集成WandB进行实验记录和可视化
4. **数据集管理**：使用Hugging Face Datasets管理数据
5. **完整流程**：从数据预处理到模型评估的完整流程

## 提交内容

### 1. 代码工程
- ✅ 模块化源代码（Python文件）
- ✅ Jupyter Notebook实验脚本
- ✅ 单元测试脚本 (`test_models.py`)

### 2. 数据集
- ✅ 原始数据：`选题三_Data/`
- ✅ 预处理数据：`processed_data/` (CSV/Parquet格式)
- ✅ 数据集描述文档
- ✅ Hugging Face数据集集成

### 3. 实验报告
- ✅ MCM格式报告：`实验报告_MCM格式.md`
- ✅ 包含研究局限与未来改进方向
- ✅ GitHub代码链接（待填入）

## GitHub仓库

代码已上传至GitHub：[待填入链接]

## 作者

大数据系统原理与应用 - 期末作业二

## 日期

2024年
