# visualize_attention.py - 注意力机制可视化（修复版）
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# 导入模型和数据处理
from network.MainNet import create_memory_forensics_net
from network.transform import Data_Transforms
from network.data import TestDataset
from torch.utils.data import DataLoader, Subset

# 设置CUDA调试模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AttentionVisualizer:
    """注意力机制可视化器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 获取实际的查询数量
        if hasattr(self.model, 'mqff'):
            self.num_queries = self.model.mqff.num_queries
        else:
            self.num_queries = 8
            
        print(f"Model loaded with {self.num_queries} queries")
        
        # 根据实际查询数量调整查询类型定义
        self.query_types = {}
        if self.num_queries >= 8:
            self.query_types = {
                'edge_queries': [0, 4],      # 边缘/轮廓检测
                'texture_queries': [1, 5],   # 纹理一致性检测
                'color_queries': [2, 6],     # 光照/颜色检测
                'semantic_queries': [3, 7]   # 全局语义检测
            }
        elif self.num_queries >= 4:
            self.query_types = {
                'edge_queries': [0],
                'texture_queries': [1],
                'color_queries': [2],
                'semantic_queries': [3]
            }
        else:
            # 如果查询数量更少，动态分配
            for i in range(self.num_queries):
                self.query_types[f'query_{i}'] = [i]
        
        self.query_colors = {
            'edge_queries': '#FF6B6B',      # 红色系
            'texture_queries': '#4ECDC4',   # 青色系
            'color_queries': '#45B7D1',     # 蓝色系
            'semantic_queries': '#96CEB4',  # 绿色系
            'query_0': '#FF6B6B',
            'query_1': '#4ECDC4',
            'query_2': '#45B7D1',
            'query_3': '#96CEB4',
            'query_4': '#F7DC6F',
            'query_5': '#BB8FCE',
            'query_6': '#85C1E2',
            'query_7': '#F8C471'
        }
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"Loading model from: {model_path}")
        
        # 先尝试加载checkpoint以获取配置
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 获取模型配置
        if isinstance(checkpoint, dict) and 'ablation_config' in checkpoint:
            config = checkpoint['ablation_config']
            enable_dstm = config.get('enable_dstm', True)
            enable_mqff = config.get('enable_mqff', True)
            enable_acgm = config.get('enable_acgm', True)
        else:
            enable_dstm = enable_mqff = enable_acgm = True
        
        print(f"Model config - DSTM: {enable_dstm}, MQFF: {enable_mqff}, ACGM: {enable_acgm}")
        
        # 创建模型
        model = create_memory_forensics_net(
            num_classes=2,
            enable_dstm=enable_dstm,
            enable_mqff=enable_mqff,
            enable_acgm=enable_acgm
        )
        
        # 加载权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 处理DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        
        return model
    
    def extract_attention_maps(self, image_tensor):
        """提取注意力图"""
        attention_maps = {}
        
        try:
            with torch.no_grad():
                # 获取中间特征
                backbone_features = self.model.backbone.extract_features(image_tensor)
                
                # 1. 提取DSTM记忆注意力
                if hasattr(self.model, 'dstm') and self.model.enable_dstm:
                    print("Extracting DSTM attention...")
                    memory_attention_weights = None
                    
                    def hook_fn(module, input, output):
                        nonlocal memory_attention_weights
                        if isinstance(output, tuple) and len(output) > 1:
                            # 获取注意力权重
                            attn_output, attn_weights = output
                            if attn_weights is not None:
                                memory_attention_weights = attn_weights.detach()
                    
                    # 注册hook
                    handle = self.model.dstm.memory_attention.register_forward_hook(hook_fn)
                    
                    try:
                        # 前向传播
                        _ = self.model.dstm(backbone_features)
                    finally:
                        # 移除hook
                        handle.remove()
                    
                    if memory_attention_weights is not None:
                        attention_maps['dstm_memory'] = memory_attention_weights
                        print(f"DSTM attention shape: {memory_attention_weights.shape}")
                
                # 2. 提取MQFF查询响应
                if hasattr(self.model, 'mqff') and self.model.enable_mqff:
                    print("Extracting MQFF attention...")
                    query_responses = {}
                    attention_weights = None
                    
                    def mqff_hook(module, input, output):
                        nonlocal attention_weights
                        if isinstance(output, tuple) and len(output) > 1:
                            attn_output, attn_weights = output
                            if attn_weights is not None:
                                attention_weights = attn_weights.detach()
                    
                    # 注册hook
                    handle = self.model.mqff.cross_attention.register_forward_hook(mqff_hook)
                    
                    try:
                        # 前向传播
                        _ = self.model.mqff(backbone_features)
                    finally:
                        # 移除hook
                        handle.remove()
                    
                    if attention_weights is not None:
                        print(f"MQFF attention shape: {attention_weights.shape}")
                        
                        # 安全地处理注意力权重
                        B = attention_weights.shape[0]
                        
                        # 检查注意力权重的维度
                        if len(attention_weights.shape) == 4:  # (B, num_heads, num_queries, HW)
                            # 平均所有注意力头
                            attention_weights = attention_weights.mean(dim=1)  # (B, num_queries, HW)
                        
                        if len(attention_weights.shape) == 3:
                            num_queries_actual = attention_weights.shape[1]
                            HW = attention_weights.shape[2]
                            
                            # 计算H和W
                            H = W = int(np.sqrt(HW))
                            if H * W != HW:
                                # 如果不是完全平方数，尝试其他方法
                                H = W = 7  # EfficientNet-B4的默认输出大小
                            
                            print(f"Reshaping attention: queries={num_queries_actual}, H={H}, W={W}")
                            
                            # 分离不同类型的查询响应
                            for query_type, query_indices in self.query_types.items():
                                # 过滤掉超出范围的索引
                                valid_indices = [idx for idx in query_indices if idx < num_queries_actual]
                                if valid_indices:
                                    query_attention = attention_weights[:, valid_indices, :].mean(dim=1)
                                    query_attention = query_attention.view(B, H, W)
                                    query_responses[query_type] = query_attention
                            
                            attention_maps['mqff_queries'] = query_responses
                
        except Exception as e:
            print(f"Error extracting attention maps: {e}")
            import traceback
            traceback.print_exc()
            
        return attention_maps
    
    def visualize_dstm_memory_attention(self, image, attention_weights, save_path=None):
        """可视化DSTM记忆注意力"""
        if attention_weights is None:
            print("No DSTM attention weights to visualize")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        try:
            # 处理注意力权重
            if len(attention_weights.shape) > 2:
                # 如果是多维的，取平均
                avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()
            else:
                avg_attention = attention_weights[0].cpu().numpy()
            
            # 确保是2D
            if len(avg_attention.shape) == 1:
                # 尝试重塑为方形
                size = int(np.sqrt(avg_attention.shape[0]))
                if size * size == avg_attention.shape[0]:
                    avg_attention = avg_attention.reshape(size, size)
                else:
                    # 使用最接近的大小
                    avg_attention = avg_attention[:49].reshape(7, 7)
            
            # 归一化
            if avg_attention.max() > avg_attention.min():
                avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())
            
            # 上采样到图像大小
            avg_attention_resized = cv2.resize(avg_attention, (image.shape[1], image.shape[0]))
            
            # 创建热力图
            im = axes[1].imshow(avg_attention_resized, cmap='hot', interpolation='bilinear')
            axes[1].set_title('Memory Attention Weights', fontsize=14)
            axes[1].axis('off')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            
            # 叠加显示
            axes[2].imshow(image)
            axes[2].imshow(avg_attention_resized, cmap='hot', alpha=0.5, interpolation='bilinear')
            axes[2].set_title('Attention Overlay', fontsize=14)
            axes[2].axis('off')
            
        except Exception as e:
            print(f"Error visualizing DSTM attention: {e}")
            axes[1].text(0.5, 0.5, 'Error visualizing attention', ha='center', va='center')
            axes[1].axis('off')
            axes[2].axis('off')
        
        plt.suptitle('DSTM Memory Attention Visualization', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_mqff_query_responses(self, image, query_responses, save_path=None):
        """可视化MQFF查询响应"""
        if not query_responses:
            print("No MQFF query responses to visualize")
            return
            
        num_queries = len(query_responses)
        fig, axes = plt.subplots(2, max(5, (num_queries + 1) // 2), figsize=(20, 8))
        axes = axes.flatten()
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 为每种查询类型创建子图
        idx = 1
        for query_type, responses in query_responses.items():
            if responses is not None and idx < len(axes) // 2:
                try:
                    # 上采样到原图大小
                    response_map = responses[0].cpu().numpy()
                    response_map = cv2.resize(response_map, (image.shape[1], image.shape[0]))
                    
                    # 归一化
                    if response_map.max() > response_map.min():
                        response_map = (response_map - response_map.min()) / (response_map.max() - response_map.min())
                    
                    # 显示响应图
                    im = axes[idx].imshow(response_map, cmap='viridis')
                    axes[idx].set_title(f'{query_type.replace("_", " ").title()}', 
                                       fontsize=11, color=self.query_colors.get(query_type, '#000000'))
                    axes[idx].axis('off')
                    
                    # 叠加显示
                    overlay_idx = idx + len(axes) // 2
                    if overlay_idx < len(axes):
                        axes[overlay_idx].imshow(image)
                        axes[overlay_idx].imshow(response_map, cmap='viridis', alpha=0.4)
                        axes[overlay_idx].set_title(f'Overlay: {query_type.replace("_", " ").title()}', 
                                             fontsize=11)
                        axes[overlay_idx].axis('off')
                    
                except Exception as e:
                    print(f"Error visualizing query {query_type}: {e}")
                    
                idx += 1
        
        # 隐藏未使用的子图
        for i in range(idx, len(axes) // 2):
            axes[i].axis('off')
        for i in range(idx + len(axes) // 2, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('MQFF Multi-Query Feature Responses', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_combined_attention_figure(self, image, attention_maps, save_path=None):
        """创建组合的注意力可视化图"""
        fig = plt.figure(figsize=(20, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # DSTM记忆注意力（上部）
        if 'dstm_memory' in attention_maps and attention_maps['dstm_memory'] is not None:
            try:
                memory_att = attention_maps['dstm_memory'][0]
                if len(memory_att.shape) > 1:
                    memory_att = memory_att.mean(dim=0)
                memory_att = memory_att.cpu().numpy()
                
                # 重塑如果需要
                if len(memory_att.shape) == 1:
                    size = int(np.sqrt(memory_att.shape[0]))
                    if size * size == memory_att.shape[0]:
                        memory_att = memory_att.reshape(size, size)
                    else:
                        memory_att = memory_att[:49].reshape(7, 7)
                
                # 归一化
                if memory_att.max() > memory_att.min():
                    memory_att = (memory_att - memory_att.min()) / (memory_att.max() - memory_att.min())
                
                # 上采样
                memory_att = cv2.resize(memory_att, (image.shape[1], image.shape[0]))
                
                # 创建子图
                for i in range(3):
                    ax = fig.add_subplot(gs[0, i])
                    
                    if i == 0:
                        ax.imshow(image)
                        ax.set_title('Input Image', fontsize=12)
                    elif i == 1:
                        im = ax.imshow(memory_att, cmap='hot')
                        ax.set_title('DSTM Memory Attention', fontsize=12)
                    else:
                        ax.imshow(image)
                        ax.imshow(memory_att, cmap='hot', alpha=0.5)
                        ax.set_title('Attention Overlay', fontsize=12)
                    
                    ax.axis('off')
                    
            except Exception as e:
                print(f"Error in combined DSTM visualization: {e}")
        
        # MQFF查询响应（下部）
        if 'mqff_queries' in attention_maps:
            query_idx = 0
            for query_type, responses in attention_maps['mqff_queries'].items():
                if responses is not None and query_idx < 8:
                    try:
                        row = 1 + query_idx // 4
                        col = query_idx % 4
                        
                        ax = fig.add_subplot(gs[row, col])
                        
                        # 处理响应图
                        response_map = responses[0].cpu().numpy()
                        response_map = cv2.resize(response_map, (image.shape[1], image.shape[0]))
                        
                        if response_map.max() > response_map.min():
                            response_map = (response_map - response_map.min()) / (response_map.max() - response_map.min())
                        
                        # 叠加显示
                        ax.imshow(image)
                        ax.imshow(response_map, cmap='viridis', alpha=0.5)
                        ax.set_title(f'{query_type.replace("_", " ").title()}', 
                                   fontsize=11, color=self.query_colors.get(query_type, '#000000'))
                        ax.axis('off')
                        
                    except Exception as e:
                        print(f"Error visualizing query {query_type} in combined view: {e}")
                        
                    query_idx += 1
        
        plt.suptitle('MemoryForensics Attention Visualization', fontsize=18, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def process_dataset_samples(self, dataset_path, output_dir, start_idx=1000, step=500, num_samples=8):
        """处理数据集样本"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        test_dataset = TestDataset(
            txt_path=dataset_path,
            test_transform=Data_Transforms['test']
        )
        
        # 选择样本索引
        indices = []
        current_idx = start_idx
        while len(indices) < num_samples and current_idx < len(test_dataset):
            indices.append(current_idx)
            current_idx += step
        
        print(f"Processing {len(indices)} samples: {indices}")
        
        # 处理每个样本
        for i, idx in enumerate(indices):
            print(f"\nProcessing sample {i+1}/{len(indices)} (index: {idx})")
            
            try:
                # 获取图像
                image_tensor, label = test_dataset[idx]
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # 获取原始图像用于显示
                with open(dataset_path, 'r') as f:
                    lines = f.readlines()
                    img_path = lines[idx].strip().split()[0]
                
                original_image = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))
                
                # 提取注意力图
                attention_maps = self.extract_attention_maps(image_tensor)
                
                # 保存可视化结果
                label_str = 'real' if label == 0 else 'fake'
                
                # DSTM可视化
                if 'dstm_memory' in attention_maps:
                    save_path = os.path.join(output_dir, f'sample_{idx}_{label_str}_dstm.png')
                    self.visualize_dstm_memory_attention(original_image, 
                                                       attention_maps['dstm_memory'], 
                                                       save_path)
                
                # MQFF可视化
                if 'mqff_queries' in attention_maps:
                    save_path = os.path.join(output_dir, f'sample_{idx}_{label_str}_mqff.png')
                    self.visualize_mqff_query_responses(original_image, 
                                                      attention_maps['mqff_queries'], 
                                                      save_path)
                
                # 组合可视化
                if attention_maps:
                    save_path = os.path.join(output_dir, f'sample_{idx}_{label_str}_combined.png')
                    self.create_combined_attention_figure(original_image, attention_maps, save_path)
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue


def main():
    # 配置参数
    model_path = './output/full_model_improved_DSTM_MQFF_ACGM/best.pth'  # 修改为您的模型路径
    dataset_path = '/home/zqc/FaceForensics++/c23/train.txt'  # 修改为您的数据集路径
    output_dir = './visualizations/attention_maps'
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model_path)
    
    # 处理数据集样本
    visualizer.process_dataset_samples(
        dataset_path=dataset_path,
        output_dir=output_dir,
        start_idx=2000,  # 从第1000张开始
        step=100,        # 每隔500张
        num_samples=8    # 总共8个样本
    )
    
    print(f"\nVisualization completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()