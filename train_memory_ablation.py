# train_memory_ablation.py - 改进的训练脚本，解决模块协同问题
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import json
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

class ImprovedMemoryForensicsTrainer:
    """改进的MemoryForensics训练器 - 解决模块协同问题"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_metrics = {'acc': 0.0, 'auc': 0.0, 'epoch': 0}
        
        # 设置随机种子
        self.setup_seed(args.seed)
        
        # 生成实验名称
        self.experiment_name = self._generate_experiment_name()
        
        # 初始化组件
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_logging()
        
        # 改进的训练策略
        self.training_strategy = self._setup_training_strategy()
        
        print(f"🚀 Improved MemoryForensics Trainer initialized!")
        print(f"Experiment: {self.experiment_name}")
        print(f"Training strategy: Progressive with specialization constraint")
        
    def setup_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def _generate_experiment_name(self):
        """生成实验名称"""
        modules = []
        if self.args.enable_dstm:
            modules.append("DSTM")
        if self.args.enable_mqff:
            modules.append("MQFF")
        if self.args.enable_acgm:
            modules.append("ACGM")
        
        if not modules:
            return f"{self.args.name}_baseline"
        else:
            return f"{self.args.name}_improved_{'_'.join(modules)}"
    
    def setup_model(self):
        """设置改进的模型"""
        print("Setting up improved MemoryForensics model...")
        print(f"Configuration:")
        print(f"  - DSTM: {self.args.enable_dstm}")
        print(f"  - MQFF: {self.args.enable_mqff}")
        print(f"  - ACGM: {self.args.enable_acgm}")
        
        # 导入改进的模型
        from network.MainNet import create_memory_forensics_net
        
        # 创建模型
        self.model = create_memory_forensics_net(
            num_classes=self.args.num_classes,
            drop_rate=self.args.drop_rate,
            enable_dstm=self.args.enable_dstm,
            enable_mqff=self.args.enable_mqff,
            enable_acgm=self.args.enable_acgm
        )
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            
        self.model = self.model.to(self.device)
        
        # 测试前向传播
        self._test_forward_pass()
        
    def _test_forward_pass(self):
        """测试前向传播"""
        print("Testing forward pass...")
        try:
            test_input = torch.randn(2, 3, 224, 224).to(self.device)
            self.model.eval()
            
            with torch.no_grad():
                # 测试基本前向传播
                test_output = self.model(test_input)
                print(f"✅ Basic forward pass: {test_output.shape}")
                
                # 测试训练模式
                self.model.train()
                main_out, aux_out, spec_loss = self.model(test_input, return_aux_outputs=True)
                print(f"✅ Training mode - Main: {main_out.shape}, Aux: {list(aux_out.keys())}")
                print(f"✅ Specialization loss: {spec_loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def setup_data(self):
        """设置数据加载器"""
        print("Setting up data loaders...")
        
        from network.data import SingleInputDataset
        from network.transform import Data_Transforms
        
        # 训练数据
        train_dataset = SingleInputDataset(
            txt_path=self.args.train_txt_path,
            train_transform=Data_Transforms['train']
        )
        
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        
        # 验证数据
        if os.path.exists(self.args.valid_txt_path):
            val_dataset = SingleInputDataset(
                txt_path=self.args.valid_txt_path,
                valid_transform=Data_Transforms['val']
            )
            self.valid_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            print(f"Warning: Validation data not found: {self.args.valid_txt_path}")
            self.valid_loader = None
            
        print(f"Train samples: {len(train_dataset)}, batches: {len(self.train_loader)}")
        if self.valid_loader:
            print(f"Valid samples: {len(val_dataset)}, batches: {len(self.valid_loader)}")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("Setting up optimizer...")
        
        # 分组参数 - 对不同模块使用不同学习率
        param_groups = []
        
        # 骨干网络参数
        backbone_params = []
        module_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif any(module in name for module in ['dstm', 'mqff', 'acgm']):
                module_params.append(param)
            else:
                classifier_params.append(param)
        
        # 不同组使用不同学习率
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.args.lr * 0.1,  # 骨干网络使用较小学习率
                'weight_decay': self.args.weight_decay
            })
        
        if module_params:
            param_groups.append({
                'params': module_params,
                'lr': self.args.lr,  # 模块使用标准学习率
                'weight_decay': self.args.weight_decay * 0.5
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.args.lr * 1.5,  # 分类器使用较大学习率
                'weight_decay': self.args.weight_decay
            })
        
        # 优化器
        self.optimizer = optim.AdamW(
            param_groups if param_groups else self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.lr * 0.01
        )
        
        print(f"Optimizer: AdamW with {len(param_groups)} parameter groups")
        
    def setup_logging(self):
        """设置日志"""
        self.output_path = os.path.join('./output', self.experiment_name)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存配置
        config = {
            'experiment_name': self.experiment_name,
            'enable_dstm': self.args.enable_dstm,
            'enable_mqff': self.args.enable_mqff,
            'enable_acgm': self.args.enable_acgm,
            'training_strategy': 'improved_progressive',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'args': vars(self.args)
        }
        
        with open(os.path.join(self.output_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Outputs will be saved to: {self.output_path}")
    
    def _setup_training_strategy(self):
        """设置训练策略"""
        
        # 创建一个简化的训练策略，避免循环导入
        class SimpleTrainingStrategy:
            def __init__(self, model, optimizer, device):
                self.model = model
                self.optimizer = optimizer
                self.device = device
                self.classification_criterion = nn.CrossEntropyLoss()
                
            def compute_loss(self, inputs, targets, epoch, total_epochs):
                """计算损失"""
                # 前向传播
                if self.model.training:
                    results = self.model(inputs, return_aux_outputs=True)
                    if len(results) == 3:  # main_output, aux_outputs, specialization_loss
                        main_output, aux_outputs, specialization_loss = results
                    else:  # main_output, aux_outputs
                        main_output, aux_outputs = results
                        specialization_loss = torch.tensor(0.0, device=inputs.device)
                else:
                    main_output = self.model(inputs)
                    aux_outputs = {}
                    specialization_loss = torch.tensor(0.0, device=inputs.device)
                
                # 主分类损失
                main_loss = self.classification_criterion(main_output, targets)
                
                # 辅助分类损失
                aux_loss = 0.0
                if aux_outputs:
                    for aux_output in aux_outputs.values():
                        aux_loss += self.classification_criterion(aux_output, targets)
                    aux_loss /= len(aux_outputs)
                
                # 动态权重调整
                progress = epoch / total_epochs
                
                # 渐进式训练策略
                if epoch < total_epochs * 0.3:
                    training_stage = 'specialization'
                    spec_weight = 0.2
                    aux_weight = 0.3
                elif epoch < total_epochs * 0.7:
                    training_stage = 'communication'
                    spec_weight = 0.1
                    aux_weight = 0.4
                else:
                    training_stage = 'full'
                    spec_weight = 0.05
                    aux_weight = 0.3
                
                # 总损失
                total_loss = (
                    main_loss + 
                    aux_weight * aux_loss + 
                    spec_weight * specialization_loss
                )
                
                return total_loss, {
                    'main_loss': main_loss.item(),
                    'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                    'spec_loss': specialization_loss.item(),
                    'training_stage': training_stage
                }
        
        return SimpleTrainingStrategy(self.model, self.optimizer, self.device)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        running_spec_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Starting epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            try:
                # 使用改进的训练策略
                self.optimizer.zero_grad()
                
                total_loss, loss_dict = self.training_strategy.compute_loss(
                    images, labels, epoch, self.args.epochs
                )
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化步骤
                self.optimizer.step()
                
                # 统计
                running_loss += total_loss.item()
                running_main_loss += loss_dict['main_loss']
                running_aux_loss += loss_dict['aux_loss']
                running_spec_loss += loss_dict['spec_loss']
                
                # 计算准确率
                with torch.no_grad():
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # 定期输出
                if batch_idx % 50 == 0:
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = 100. * correct / total
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"Epoch[{epoch+1:2d}/{self.args.epochs}] "
                          f"Batch[{batch_idx:4d}/{len(self.train_loader)}] "
                          f"Loss:{current_loss:.4f} Acc:{current_acc:.2f}% "
                          f"LR:{current_lr:.6f} Stage:{loss_dict['training_stage']}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # 计算平均值
        num_batches = len(self.train_loader)
        epoch_loss = running_loss / num_batches
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1} Training Summary:")
        print(f"  - Total Loss: {epoch_loss:.4f}")
        print(f"  - Main Loss: {running_main_loss/num_batches:.4f}")
        print(f"  - Aux Loss: {running_aux_loss/num_batches:.4f}")
        print(f"  - Spec Loss: {running_spec_loss/num_batches:.4f}")
        print(f"  - Accuracy: {epoch_acc:.4f}")
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        if self.valid_loader is None:
            return 0.0, 0.0, 0.0, {}
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    outputs = self.model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.num_classes == 2:
                        probs = torch.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        
                except Exception as e:
                    print(f"Validation error in batch {batch_idx}: {e}")
                    continue
        
        val_loss = running_loss / len(self.valid_loader)
        val_acc = correct / total
        val_auc = 0.0
        detailed_metrics = {}
        
        # 计算详细指标
        if self.args.num_classes == 2 and len(set(all_labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                from network.utils import cal_metrics
                from network.log_record import save_acc
                
                val_auc = roc_auc_score(all_labels, all_probs)
                ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(all_labels, all_probs)
                
                detailed_metrics = {
                    'auc': auc_score,
                    'ap': ap_score,
                    'eer': eer,
                    'tpr_2': TPR_2,
                    'tpr_3': TPR_3,
                    'tpr_4': TPR_4
                }
                
                # 保存结果
                save_acc(val_acc, ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4, epoch, self.output_path)
                
            except Exception as e:
                print(f"Metrics calculation error: {e}")
        
        print(f"Validation Summary:")
        print(f"  - Loss: {val_loss:.4f}")
        print(f"  - Accuracy: {val_acc:.4f}")
        print(f"  - AUC: {val_auc:.4f}")
        
        return val_loss, val_acc, val_auc, detailed_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'ablation_config': {
                'enable_dstm': self.args.enable_dstm,
                'enable_mqff': self.args.enable_mqff,
                'enable_acgm': self.args.enable_acgm
            }
        }
        
        torch.save(checkpoint, os.path.join(self.output_path, 'latest.pth'))
        
        if is_best:
            torch.save(model_to_save.state_dict(), os.path.join(self.output_path, 'best.pkl'))
            torch.save(checkpoint, os.path.join(self.output_path, 'best.pth'))
            print(f"🌟 Best model saved!")
    
    def train(self):
        """主训练循环"""
        print(f"🚀 Starting improved MemoryForensics training: {self.experiment_name}")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'='*60}")
            
            try:
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_acc, val_auc, detailed_metrics = self.validate_epoch(epoch)
                
                # 检查是否为最佳模型
                is_best = False
                current_metric = val_auc if self.args.metric == 'auc' else val_acc
                
                if current_metric > self.best_metrics[self.args.metric]:
                    self.best_metrics = {
                        'acc': val_acc,
                        'auc': val_auc,
                        'epoch': epoch,
                        **detailed_metrics
                    }
                    is_best = True
                    print(f"🌟 New best model! {self.args.metric.upper()}: {current_metric:.4f}")
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best=is_best)
                
                # 学习率调度
                self.scheduler.step()
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                print("\n⏸️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算总时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        # 保存最终结果
        try:
            from network.pipeline import params_count, cal_params_thop
            from network.log_record import save_final_results
            
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = params_count(model_to_save)
            
            test_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            flops, params_str = cal_params_thop(model_to_save, test_tensor)
            
            save_final_results(flops, total_params, f"{hours}h {minutes}m", 
                              self.best_metrics['acc'], self.best_metrics['auc'], self.output_path)
        except Exception as e:
            print(f"Warning: Could not save final results: {e}")
        
        print(f"\n{'='*60}")
        print(f"🎉 Improved MemoryForensics training completed!")
        print(f"Experiment: {self.experiment_name}")
        print(f"Best epoch: {self.best_metrics['epoch'] + 1}")
        print(f"Best accuracy: {self.best_metrics['acc']:.4f}")
        print(f"Best AUC: {self.best_metrics['auc']:.4f}")
        print(f"Total time: {hours}h {minutes}m")
        print(f"{'='*60}")
        
        return self.best_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved MemoryForensics Training')
    
    # 基本参数
    parser.add_argument('--name', type=str, default='memory_forensics',
                       help='Base experiment name')
    parser.add_argument('--train_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/train.txt',
                       help='Training data txt path')
    parser.add_argument('--valid_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/val.txt',
                       help='Validation data txt path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 模型参数
    parser.add_argument('--drop_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # 消融实验参数
    parser.add_argument('--enable-dstm', action='store_true', default=False,
                       help='Enable DSTM module')
    parser.add_argument('--enable-mqff', action='store_true', default=False,
                       help='Enable MQFF module')
    parser.add_argument('--enable-acgm', action='store_true', default=False,
                       help='Enable ACGM module')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # 验证参数
    parser.add_argument('--metric', type=str, default='auc',
                       choices=['acc', 'auc'],
                       help='Metric for model selection')
    
    args = parser.parse_args()
    
    print("🚀 Improved MemoryForensics Training")
    print("="*50)
    print(f"🔧 Key Improvements:")
    print(f"  - Parallel module processing (避免累积误差)")
    print(f"  - Module communication hub (模块间协作)")
    print(f"  - Specialization constraint (强制功能分化)")
    print(f"  - Dynamic weight fusion (自适应权重)")
    print(f"  - Progressive training strategy (渐进式训练)")
    print("="*50)
    print(f"Ablation Configuration:")
    print(f"  - DSTM: {args.enable_dstm}")
    print(f"  - MQFF: {args.enable_mqff}")
    print(f"  - ACGM: {args.enable_acgm}")
    print(f"Training Parameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print("="*50)
    
    try:
        trainer = ImprovedMemoryForensicsTrainer(args)
        best_metrics = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Final best metrics: {best_metrics}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())