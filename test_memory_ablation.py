# =====================================================
# test_memory_ablation.py - 消融测试脚本
# =====================================================
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# 导入模块
from network.MainNet import create_memory_forensics_net
from network.data import TestDataset
from network.utils import cal_metrics, plot_ROC
from network.transform import Data_Transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MemoryForensicsAblationTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """根据模型配置设置模型"""
        print(f"Setting up model for testing...")
        
        # 首先尝试从检查点加载配置
        model_config = self.load_model_config()
        
        if model_config:
            print(f"Loaded model configuration from checkpoint:")
            print(f"  - DSTM: {model_config['enable_dstm']}")
            print(f"  - MQFF: {model_config['enable_mqff']}")
            print(f"  - ACGM: {model_config['enable_acgm']}")
            
            enable_dstm = model_config['enable_dstm']
            enable_mqff = model_config['enable_mqff']
            enable_acgm = model_config['enable_acgm']
        else:
            print(f"Using command line configuration:")
            print(f"  - DSTM: {self.args.enable_dstm}")
            print(f"  - MQFF: {self.args.enable_mqff}")
            print(f"  - ACGM: {self.args.enable_acgm}")
            
            enable_dstm = self.args.enable_dstm
            enable_mqff = self.args.enable_mqff
            enable_acgm = self.args.enable_acgm
        
        # 创建模型
        self.model = create_memory_forensics_net(
            num_classes=self.args.num_classes,
            drop_rate=0.0,  # 测试时不使用dropout
            enable_dstm=enable_dstm,
            enable_mqff=enable_mqff,
            enable_acgm=enable_acgm
        )
        
        # 加载模型权重
        if os.path.exists(self.args.model_path):
            print(f"Loading model from: {self.args.model_path}")
            self.load_model()
        else:
            raise FileNotFoundError(f"Model not found: {self.args.model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        
    def load_model_config(self):
        """从检查点加载模型配置"""
        try:
            if self.args.model_path.endswith('.pth'):
                checkpoint = torch.load(self.args.model_path, map_location='cpu')
                if 'ablation_config' in checkpoint:
                    return checkpoint['ablation_config']
            
            model_dir = os.path.dirname(self.args.model_path)
            config_file = os.path.join(model_dir, 'ablation_config.json')
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return {
                    'enable_dstm': config.get('enable_dstm', False),
                    'enable_mqff': config.get('enable_mqff', False),
                    'enable_acgm': config.get('enable_acgm', False)
                }
            
            return None
            
        except Exception as e:
            print(f"Could not load model config: {e}")
            return None
    
    def load_model(self):
        """加载模型权重"""
        try:
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理DataParallel保存的模型
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_data(self):
        """设置测试数据加载器"""
        print("Setting up test data...")
        
        test_dataset = TestDataset(
            txt_path=self.args.test_txt_path,
            test_transform=Data_Transforms['test']
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        print(f"Test data loaded: {len(test_dataset)} samples")
    
    def test(self):
        """运行测试"""
        print("Running test...")
        
        all_labels = []
        all_probs = []
        all_preds = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.num_classes == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
                if batch_idx % 100 == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(f"Batch [{batch_idx:4d}/{len(self.test_loader)}] Acc: {current_acc:.2%}")
        
        return self.calculate_metrics(all_labels, all_probs, all_preds, correct, total)
    
    def calculate_metrics(self, all_labels, all_probs, all_preds, correct, total):
        """计算详细指标"""
        results = {
            'accuracy': correct / total if total > 0 else 0,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        if self.args.num_classes == 2 and len(all_labels) > 0:
            try:
                ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(all_labels, all_probs)
                
                results.update({
                    'auc': auc_score,
                    'ap': ap_score,
                    'eer': eer,
                    'tpr_at_fpr_1e-2': TPR_2,
                    'tpr_at_fpr_1e-3': TPR_3,
                    'tpr_at_fpr_1e-4': TPR_4
                })
                
                # 绘制ROC曲线
                plot_eer, plot_auc = plot_ROC(all_labels, all_probs)
                
                cm = confusion_matrix(all_labels, all_preds)
                results['confusion_matrix'] = cm.tolist()
                
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    results.update({
                        'true_negative': int(tn),
                        'false_positive': int(fp),
                        'false_negative': int(fn),
                        'true_positive': int(tp),
                        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
                    })
                    
                    precision = results['precision']
                    recall = results['recall']
                    results['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
            except Exception as e:
                print(f"Error calculating detailed metrics: {e}")
        
        return results
    
    def save_results(self, results):
        """保存测试结果"""
        output_dir = os.path.dirname(self.args.model_path)
        if not output_dir:
            output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_file = os.path.join(output_dir, f'test_results_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(f"MemoryForensics Ablation Test Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model path: {self.args.model_path}\n")
            f.write(f"Test data: {self.args.test_txt_path}\n")
            f.write(f"Enhancement modules: DSTM={self.args.enable_dstm}, MQFF={self.args.enable_mqff}, ACGM={self.args.enable_acgm}\n")
            f.write(f"\nResults:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            
            if 'auc' in results:
                f.write(f"AUC: {results['auc']:.4f}\n")
                f.write(f"AP: {results['ap']:.4f}\n")
                f.write(f"EER: {results['eer']:.4f}\n")
                f.write(f"F1 Score: {results['f1_score']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"Specificity: {results['specificity']:.4f}\n")
                f.write(f"TPR@FPR=1e-2: {results['tpr_at_fpr_1e-2']:.4f}\n")
                f.write(f"TPR@FPR=1e-3: {results['tpr_at_fpr_1e-3']:.4f}\n")
                f.write(f"TPR@FPR=1e-4: {results['tpr_at_fpr_1e-4']:.4f}\n")
        
        # 保存JSON格式结果
        json_results = {k: v for k, v in results.items() if k != 'confusion_matrix'}
        json_file = os.path.join(output_dir, f'test_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        return result_file
    
    def run_test(self):
        """运行测试"""
        print(f"\n🚀 Testing MemoryForensics Ablation Model")
        print("="*60)
        
        results = self.test()
        self.save_results(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='MemoryForensics Ablation Model Testing')
    
    # 基本参数
    parser.add_argument('--test_txt_path', type=str, 
                       default='/home/zqc/DFR/test.txt',
                       help='Test data txt path')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # 消融实验参数（可选，会尝试从模型配置文件自动读取）
    parser.add_argument('--enable_dstm', action='store_true', default=False,
                       help='Enable DSTM module (auto-detected if not specified)')
    parser.add_argument('--enable_mqff', action='store_true', default=False,
                       help='Enable MQFF module (auto-detected if not specified)')
    parser.add_argument('--enable_acgm', action='store_true', default=False,
                       help='Enable ACGM module (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("🧪 MemoryForensics Ablation Model Testing")
    print("="*40)
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_txt_path}")
    print("="*40)
    
    try:
        tester = MemoryForensicsAblationTester(args)
        results = tester.run_test()
        
        print("\n🎉 Testing completed successfully!")
        print(f"Final Accuracy: {results['accuracy']:.4f}")
        if 'auc' in results:
            print(f"Final AUC: {results['auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())