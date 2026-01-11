"""
单元测试脚本
测试各个模块的功能
"""

import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入要测试的模块
from models import LSTMModel, TransformerModel, create_model
from time_series_data import TennisSequenceDataset, build_point_sequence, build_all_sequences
from data_preprocessing import load_data, clean_data, get_match_results

class TestModels(unittest.TestCase):
    """测试模型定义"""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 20
        self.hidden_dim = 64
        self.num_classes = 2
        
    def test_lstm_model(self):
        """测试LSTM模型"""
        model = LSTMModel(self.input_dim, self.hidden_dim, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_transformer_model(self):
        """测试Transformer模型"""
        model = TransformerModel(self.input_dim, d_model=64, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_create_model(self):
        """测试模型创建函数"""
        lstm = create_model('lstm', self.input_dim, hidden_dim=self.hidden_dim)
        transformer = create_model('transformer', self.input_dim, d_model=64)
        
        self.assertIsInstance(lstm, LSTMModel)
        self.assertIsInstance(transformer, TransformerModel)

class TestTimeSeriesData(unittest.TestCase):
    """测试时间序列数据构建"""
    
    def setUp(self):
        # 创建测试数据
        self.df = pd.DataFrame({
            'match_id': ['match1'] * 20,
            'point_no': range(1, 21),
            'p1_points_won': range(1, 21),
            'p2_points_won': range(0, 20),
            'p1_sets': [0] * 20,
            'p2_sets': [0] * 20,
            'p1_games': [0] * 20,
            'p2_games': [0] * 20,
            'server': [1, 2] * 10,
            'serve_no': [1] * 20,
            'point_victor': [1, 2] * 10,
            'p1_ace': [0] * 20,
            'p2_ace': [0] * 20,
            'p1_winner': [0] * 20,
            'p2_winner': [0] * 20,
            'p1_unf_err': [0] * 20,
            'p2_unf_err': [0] * 20,
            'p1_break_pt': [0] * 20,
            'p2_break_pt': [0] * 20,
            'rally_count': [5] * 20,
            'speed_mph': [100] * 20,
            'p1_distance_run': [10] * 20,
            'p2_distance_run': [10] * 20
        })
        
    def test_build_point_sequence(self):
        """测试构建单场比赛序列"""
        sequences, labels = build_point_sequence(self.df, 'match1', sequence_length=5)
        self.assertGreater(len(sequences), 0)
        self.assertEqual(len(sequences), len(labels))
        self.assertEqual(sequences.shape[1], 5)  # 序列长度
        
    def test_tennis_sequence_dataset(self):
        """测试数据集类"""
        sequences = np.random.randn(10, 5, 20)
        labels = np.random.randint(0, 2, 10)
        dataset = TennisSequenceDataset(sequences, labels)
        
        self.assertEqual(len(dataset), 10)
        seq, label = dataset[0]
        self.assertEqual(seq.shape, (5, 20))
        self.assertIsInstance(label, torch.Tensor)

class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理"""
    
    def test_load_data(self):
        """测试数据加载"""
        # 这里需要实际的数据文件，所以跳过或使用模拟数据
        pass
        
    def test_clean_data(self):
        """测试数据清洗"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['a', 'b', 'c', np.nan],
            'match_id': ['m1', 'm1', 'm1', 'm1'],
            'point_victor': [1, 2, 1, 2],
            'p1_points_won': [1, 2, 3, 4],
            'p2_points_won': [0, 1, 2, 3]
        })
        df_clean = clean_data(df)
        self.assertFalse(df_clean['col1'].isnull().any())

class TestTraining(unittest.TestCase):
    """测试训练流程"""
    
    def test_model_forward(self):
        """测试模型前向传播"""
        model = LSTMModel(20, hidden_dim=64)
        x = torch.randn(4, 10, 20)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))
        
    def test_loss_computation(self):
        """测试损失计算"""
        model = LSTMModel(20, hidden_dim=64)
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 10, 20)
        labels = torch.randint(0, 2, (4,))
        output = model(x)
        loss = criterion(output, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)

def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeSeriesData))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")
