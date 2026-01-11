"""
深度学习模型可视化模块
生成Loss曲线、预测结果趋势图、动量变化热力图等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from matplotlib import font_manager
import os
import pandas as pd

# 设置中文字体
import matplotlib
import platform
import os

# 全局字体属性变量
_chinese_font_prop = None

def setup_chinese_font():
    """设置中文字体，返回FontProperties对象"""
    global _chinese_font_prop
    
    if _chinese_font_prop is not None:
        return _chinese_font_prop
    
    try:
        if platform.system() == 'Windows':
            # Windows系统字体路径（按优先级）
            font_paths = [
                r'C:\Windows\Fonts\msyh.ttc',      # Microsoft YaHei
                r'C:\Windows\Fonts\msyhbd.ttc',     # Microsoft YaHei Bold
                r'C:\Windows\Fonts\simhei.ttf',     # SimHei (黑体)
                r'C:\Windows\Fonts\simsun.ttc',      # SimSun (宋体)
                r'C:\Windows\Fonts\simkai.ttf',     # KaiTi (楷体)
            ]
            
            # 尝试直接加载字体文件
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        _chinese_font_prop = font_manager.FontProperties(fname=font_path)
                        font_name = _chinese_font_prop.get_name()
                        # 同时设置rcParams
                        plt.rcParams['font.sans-serif'] = [font_name] + ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans', 'sans-serif']
                        plt.rcParams['axes.unicode_minus'] = False
                        print(f"成功加载字体: {font_name} ({font_path})")
                        return _chinese_font_prop
                    except Exception as e:
                        continue
            
            # 如果直接加载失败，尝试从系统字体列表查找
            chinese_font_names = []
            for font in font_manager.fontManager.ttflist:
                font_name = font.name
                if any(keyword in font_name for keyword in 
                      ['Microsoft YaHei', 'YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', '微软雅黑', '黑体', '宋体', '楷体']):
                    chinese_font_names.append(font_name)
            
            if chinese_font_names:
                # 去重并保持顺序
                seen = set()
                unique_fonts = []
                for font in chinese_font_names:
                    if font not in seen:
                        seen.add(font)
                        unique_fonts.append(font)
                
                # 优先使用Microsoft YaHei
                preferred = [f for f in unique_fonts if 'Microsoft YaHei' in f or 'YaHei' in f or '微软雅黑' in f]
                if preferred:
                    font_name = preferred[0]
                else:
                    font_name = unique_fonts[0]
                
                plt.rcParams['font.sans-serif'] = [font_name] + ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                _chinese_font_prop = font_manager.FontProperties(family=font_name)
                print(f"使用字体: {font_name}")
                return _chinese_font_prop
            else:
                print("警告: 未找到中文字体，中文可能显示为方框")
                # 使用默认设置
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                _chinese_font_prop = font_manager.FontProperties()
        else:
            # 非Windows系统
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            _chinese_font_prop = font_manager.FontProperties()
    except Exception as e:
        print(f"字体设置失败: {e}，使用默认设置")
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        _chinese_font_prop = font_manager.FontProperties()
    
    return _chinese_font_prop

# 执行字体设置
setup_chinese_font()

# 清除matplotlib字体缓存（如果字体不显示，可以尝试清除缓存）
# import matplotlib
# matplotlib.font_manager._rebuild()

sns.set_style("whitegrid")

def plot_training_curves(history: dict, model_name: str, save_path: str = None):
    """
    绘制训练曲线（Loss曲线）
    
    Args:
        history: 训练历史字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # 获取中文字体属性
    font_prop = setup_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} 训练曲线', fontsize=16, fontweight='bold', fontproperties=font_prop)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('损失曲线', fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_accuracy'], 'g-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率曲线', fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.grid(True, alpha=0.3)
    
    # F1-Score曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_f1'], 'm-', label='验证F1分数', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1分数曲线', fontproperties=font_prop)
    ax3.legend(prop=font_prop)
    ax3.grid(True, alpha=0.3)
    
    # AUC曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_auc'], 'c-', label='验证AUC', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC')
    ax4.set_title('AUC曲线', fontproperties=font_prop)
    ax4.legend(prop=font_prop)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_prediction_trend(model, data_loader, device, match_id: str = None,
                          save_path: str = None):
    """
    绘制预测结果趋势图
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        match_id: 比赛ID（可选）
        save_path: 保存路径
    """
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(labels.numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    
    # 获取中文字体属性
    font_prop = setup_chinese_font()
    
    # 绘制趋势图
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('预测结果趋势图', fontsize=16, fontweight='bold', fontproperties=font_prop)
    
    # 预测概率趋势
    ax1 = axes[0]
    x = range(len(probabilities))
    ax1.plot(x, probabilities, 'b-', alpha=0.6, label='预测概率', linewidth=1)
    ax1.plot(x, true_labels, 'r-', alpha=0.8, label='真实标签', linewidth=1)
    ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='阈值(0.5)')
    ax1.set_xlabel('样本序号', fontproperties=font_prop)
    ax1.set_ylabel('概率/标签', fontproperties=font_prop)
    ax1.set_title('预测概率趋势', fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    ax1.grid(True, alpha=0.3)
    
    # 预测准确率滑动窗口
    ax2 = axes[1]
    window_size = 50
    accuracies = []
    for i in range(len(predictions)):
        start = max(0, i - window_size + 1)
        window_preds = predictions[start:i+1]
        window_labels = true_labels[start:i+1]
        acc = (window_preds == window_labels).mean()
        accuracies.append(acc)
    
    ax2.plot(x, accuracies, 'g-', linewidth=2, label=f'滑动窗口准确率 (窗口={window_size})')
    ax2.set_xlabel('样本序号', fontproperties=font_prop)
    ax2.set_ylabel('准确率', fontproperties=font_prop)
    ax2.set_title('预测准确率趋势', fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测趋势图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_momentum_heatmap(df: pd.DataFrame, match_id: str, save_path: str = None):
    """
    绘制动量变化热力图
    
    Args:
        df: 包含动量特征的数据
        match_id: 比赛ID
        save_path: 保存路径
    """
    match_data = df[df['match_id'] == match_id].copy()
    match_data = match_data.sort_values('point_no').reset_index(drop=True)
    
    # 选择动量特征
    momentum_features = [
        'score_diff', 'set_diff', 'game_diff',
        'p1_recent_win_rate', 'p2_recent_win_rate',
        'p1_streak', 'p2_streak'
    ]
    
    # 只保留存在的特征
    available_features = [f for f in momentum_features if f in match_data.columns]
    
    if len(available_features) == 0:
        print("未找到动量特征")
        return
    
    # 提取特征矩阵
    feature_matrix = match_data[available_features].values.T
    
    # 获取中文字体属性
    font_prop = setup_chinese_font()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(15, 6))
    
    im = ax.imshow(feature_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    ax.set_xticks(range(0, len(match_data), max(1, len(match_data)//10)))
    ax.set_xticklabels([match_data.iloc[i]['point_no'] 
                        for i in range(0, len(match_data), max(1, len(match_data)//10))])
    ax.set_yticks(range(len(available_features)))
    ax.set_yticklabels(available_features, fontproperties=font_prop)
    ax.set_xlabel('点数', fontproperties=font_prop)
    ax.set_ylabel('动量特征', fontproperties=font_prop)
    ax.set_title(f'比赛 {match_id} 动量变化热力图', fontproperties=font_prop)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_model_comparison(results: dict, save_path: str = None):
    """
    绘制模型对比图
    
    Args:
        results: 模型结果字典 {model_name: {test_metrics}}
        save_path: 保存路径
    """
    # 获取中文字体属性
    font_prop = setup_chinese_font()
    
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1', 'auc', 'mae', 'rmse']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('模型性能对比', fontsize=16, fontweight='bold', fontproperties=font_prop)
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[name]['test_metrics'][metric] for name in model_names]
        
        if metric in ['mae', 'rmse']:
            # MAE和RMSE越小越好
            bars = ax.bar(model_names, values, color='coral')
            ax.set_ylabel(metric.upper(), fontproperties=font_prop)
            ax.set_title(f'{metric.upper()} (越小越好)', fontproperties=font_prop)
        else:
            # 其他指标越大越好
            bars = ax.bar(model_names, values, color='skyblue')
            ax.set_ylabel(metric.upper(), fontproperties=font_prop)
            ax.set_title(f'{metric.upper()} (越大越好)', fontproperties=font_prop)
        
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontproperties=font_prop)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签（数值不需要中文字体）
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
    
    # 隐藏最后一个子图
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_all_visualizations_deep(results: dict, model, test_loader, device,
                                  df_momentum: pd.DataFrame = None):
    """
    创建所有可视化图表
    
    Args:
        results: 模型结果字典
        model: 最佳模型
        test_loader: 测试数据加载器
        device: 设备
        df_momentum: 动量特征数据（可选）
    """
    os.makedirs('visualizations', exist_ok=True)
    
    print("正在生成可视化图表...")
    
    # 1. 训练曲线
    for model_name, result in results.items():
        plot_training_curves(
            result['history'],
            model_name,
            f'visualizations/training_curves_{model_name}.png'
        )
    
    # 2. 预测趋势图
    plot_prediction_trend(
        model, test_loader, device,
        save_path='visualizations/prediction_trend.png'
    )
    
    # 3. 模型对比
    plot_model_comparison(results, 'visualizations/model_comparison.png')
    
    # 4. 动量热力图（如果有数据）
    if df_momentum is not None:
        sample_matches = df_momentum['match_id'].unique()[:3]
        for i, match_id in enumerate(sample_matches):
            plot_momentum_heatmap(
                df_momentum, match_id,
                f'visualizations/momentum_heatmap_{i+1}.png'
            )
    
    print("\n所有可视化图表已生成完成！")

if __name__ == "__main__":
    print("可视化模块测试完成")
