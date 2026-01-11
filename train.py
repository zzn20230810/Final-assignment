"""
模型训练模块
使用PyTorch训练LSTM/Transformer模型
集成WandB进行实验记录
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score
import wandb
from typing import Dict, Tuple
import os

# 抑制常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches

def evaluate(model: nn.Module, data_loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 获取预测
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = (all_preds == all_labels).mean()
    
    # MAE和RMSE（对于概率预测）
    mae = mean_absolute_error(all_labels, all_probs)
    rmse = np.sqrt(mean_squared_error(all_labels, all_probs))
    
    # F1-score
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'f1': f1,
        'auc': auc
    }

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: torch.device = None, use_wandb: bool = True,
                project_name: str = "tennis-momentum", model_name: str = "lstm") -> Dict:
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        use_wandb: 是否使用WandB
        project_name: WandB项目名
        model_name: 模型名称
        
    Returns:
        训练历史
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 初始化WandB
    if use_wandb:
        wandb.init(
            project=project_name,
            name=model_name,
            config={
                'model': model_name,
                'epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': train_loader.batch_size,
            }
        )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_mae': [],
        'val_rmse': [],
        'val_f1': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"开始训练 {model_name} 模型...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # 记录到WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc']
            })
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val AUC: {val_metrics['auc']:.4f}")
        
        # 早停
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # 保存最佳模型
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于第 {epoch+1} 轮")
                break
    
    if use_wandb:
        wandb.finish()
    
    return history

def compare_models(models_config: Dict, train_loader: DataLoader, 
                  val_loader: DataLoader, test_loader: DataLoader,
                  input_dim: int, num_epochs: int = 50, use_wandb: bool = True) -> Dict:
    """
    对比多个模型
    
    Args:
        models_config: 模型配置字典 {model_name: {type, kwargs}}
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        input_dim: 输入特征维度
        num_epochs: 训练轮数
        use_wandb: 是否使用WandB
        
    Returns:
        所有模型的测试结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name}")
        print(f"{'='*60}")
        
        # 创建模型
        from models import create_model
        model = create_model(config['type'], input_dim, **config.get('kwargs', {}))
        
        # 训练模型
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            device=device,
            use_wandb=use_wandb,
            model_name=model_name
        )
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'checkpoints/best_{model_name}.pth'))
        
        # 测试
        test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        
        results[model_name] = {
            'history': history,
            'test_metrics': test_metrics
        }
        
        print(f"\n{model_name} 测试结果:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
    
    return results

if __name__ == "__main__":
    # 测试训练流程
    from time_series_data import prepare_time_series_data
    from data_preprocessing import prepare_data
    from models import create_model
    
    # 加载数据
    file_path = "选题三_Data/2024_Wimbledon_featured_matches.csv"
    df_clean, _ = prepare_data(file_path)
    
    # 准备时间序列数据
    train_loader, val_loader, test_loader, feature_dim = prepare_time_series_data(
        df_clean, sequence_length=10
    )
    
    # 创建模型
    model = create_model('lstm', feature_dim, hidden_dim=128, num_layers=2)
    
    # 训练模型
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=20,
        use_wandb=False  # 测试时不使用WandB
    )
