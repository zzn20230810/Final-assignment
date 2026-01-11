"""
深度学习模型主训练脚本
使用LSTM/Transformer进行网球比赛动量分析与预测
"""

import os
import warnings
import torch
from data_preprocessing import prepare_data
from time_series_data import prepare_time_series_data, create_momentum_features
from models import create_model
from train import train_model, compare_models, evaluate
from visualization_deep import create_all_visualizations_deep
import torch.nn as nn

# 抑制常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    """主函数"""
    print("=" * 80)
    print("网球比赛动量分析与结果预测 - 深度学习版本")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 1. 数据预处理
    print("\n[步骤 1/5] 数据预处理")
    print("-" * 80)
    file_path = "选题三_Data/2024_Wimbledon_featured_matches.csv"
    
    if not os.path.exists(file_path):
        print(f"错误：找不到数据文件 {file_path}")
        return
    
    df_clean, _ = prepare_data(file_path)
    print(f"✓ 数据预处理完成")
    print(f"  - 总点数: {len(df_clean)}")
    
    # 2. 创建动量特征
    print("\n[步骤 2/5] 创建动量特征")
    print("-" * 80)
    df_momentum = create_momentum_features(df_clean)
    print(f"✓ 动量特征创建完成")
    
    # 3. 构建时间序列数据
    print("\n[步骤 3/5] 构建时间序列数据")
    print("-" * 80)
    sequence_length = 10
    train_loader, val_loader, test_loader, feature_dim = prepare_time_series_data(
        df_momentum, sequence_length=sequence_length
    )
    print(f"✓ 时间序列数据构建完成")
    print(f"  - 序列长度: {sequence_length}")
    print(f"  - 特征维度: {feature_dim}")
    
    # 4. 模型配置
    print("\n[步骤 4/5] 模型训练与对比")
    print("-" * 80)
    
    models_config = {
        'LSTM': {
            'type': 'lstm',
            'kwargs': {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3
            }
        },
        'Transformer': {
            'type': 'transformer',
            'kwargs': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 2,
                'dropout': 0.3
            }
        }
    }
    
    # 对比模型
    results = compare_models(
        models_config,
        train_loader,
        val_loader,
        test_loader,
        feature_dim,
        num_epochs=50,
        use_wandb=True  # 使用WandB记录实验
    )
    
    # 5. 可视化
    print("\n[步骤 5/5] 生成可视化")
    print("-" * 80)
    
    # 选择最佳模型（基于F1分数）
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_metrics']['f1'])
    best_model = create_model(
        models_config[best_model_name]['type'],
        feature_dim,
        **models_config[best_model_name]['kwargs']
    )
    best_model.load_state_dict(
        torch.load(f'checkpoints/best_{best_model_name}.pth')
    )
    best_model = best_model.to(device)
    
    create_all_visualizations_deep(
        results,
        best_model,
        test_loader,
        device,
        df_momentum
    )
    
    # 6. 保存结果
    print("\n[保存结果]")
    print("-" * 80)
    os.makedirs('results', exist_ok=True)
    
    import json
    # 保存测试结果
    test_results = {
        name: result['test_metrics'] 
        for name, result in results.items()
    }
    
    with open('results/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print("✓ 结果已保存到 results/test_results.json")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print("\n模型测试结果总结：")
    print("-" * 80)
    
    for model_name, result in results.items():
        metrics = result['test_metrics']
        print(f"\n{model_name}:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print(f"\n最佳模型: {best_model_name} (F1: {results[best_model_name]['test_metrics']['f1']:.4f})")
    print("\n生成的文件：")
    print("  - results/ : 模型测试结果")
    print("  - checkpoints/ : 模型检查点")
    print("  - visualizations/ : 可视化图表")

if __name__ == "__main__":
    main()
