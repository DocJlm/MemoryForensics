# network/MainNet.py - 改进的三创新点集成版本
# 解决模块组合负效应问题，实现真正的协同工作

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from .enhanced_modules_memory import (
    DynamicSpatioTemporalMemory, 
    MultiQueryFeatureFusion, 
    AdaptiveChannelGroupingMechanism,
    SimplifiedAdaptiveChannelGroupingMechanism
)

class MemoryForensicsNet(nn.Module):
    """
    改进的MemoryForensics主网络 - 解决模块协同问题
    
    主要改进：
    1. 并行模块设计 - 所有模块并行处理，避免累积误差
    2. 模块间通信机制 - 引入交叉注意力实现模块协作
    3. 专门化约束 - 强制模块学习不同特征，避免冗余
    4. 动态权重融合 - 根据模块置信度动态调整权重
    5. 渐进式训练支持 - 支持分阶段训练策略
    """
    
    def __init__(self, num_classes=2, drop_rate=0.3, 
                 enable_dstm=True, enable_mqff=True, enable_acgm=True,
                 dstm_config=None, mqff_config=None, acgm_config=None):
        super(MemoryForensicsNet, self).__init__()
        
        self.num_classes = num_classes
        self.enable_dstm = enable_dstm
        self.enable_mqff = enable_mqff
        self.enable_acgm = enable_acgm
        
        # 默认配置
        self.dstm_config = dstm_config or {
            'memory_size': 128,  # 减少记忆大小提高专门化
            'memory_dim': 256,   # 减少维度避免过拟合
            'update_rate': 0.1,
            'temperature': 1.0,
            'update_threshold': 0.8
        }
        
        self.mqff_config = mqff_config or {
            'num_queries': 6,    # 减少查询数量提高专门化
            'query_dim': 256,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        self.acgm_config = acgm_config or {
            'num_groups': 8,
            'reduction_ratio': 16,
            'routing_iterations': 3,
            'use_simplified': False
        }
        
        # EfficientNet-B4 骨干网络
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.feature_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        # 获取卷积特征图的维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            conv_features = self.backbone.extract_features(test_input)
            self.conv_features_dim = conv_features.shape[1]
        
        print(f"EfficientNet-B4 feature dimensions: {self.conv_features_dim}")
        
        # 创新模块初始化
        self._initialize_innovation_modules()
        
        # 模块间通信中心
        self.communication_hub = ModuleCommunicationHub(
            self.conv_features_dim,
            enabled_modules=[enable_dstm, enable_mqff, enable_acgm]
        )
        
        # 专门化约束模块
        self.specialization_constraint = SpecializationConstraint(
            self.conv_features_dim
        )
        
        # 动态权重融合器
        self.dynamic_fusion = DynamicWeightFusion(
            self.conv_features_dim,
            num_modules=sum([enable_dstm, enable_mqff, enable_acgm])
        )
        
        # 改进的智能分类器
        self.intelligent_classifier = ImprovedIntelligentClassifier(
            input_channels=self.conv_features_dim,
            num_classes=num_classes,
            drop_rate=drop_rate,
            num_modules=sum([enable_dstm, enable_mqff, enable_acgm])
        )
        
        # 辅助分类器（用于多任务学习）
        self.auxiliary_classifiers = nn.ModuleDict()
        if enable_dstm:
            self.auxiliary_classifiers['dstm'] = self._create_auxiliary_classifier(drop_rate)
        if enable_mqff:
            self.auxiliary_classifiers['mqff'] = self._create_auxiliary_classifier(drop_rate)
        if enable_acgm:
            self.auxiliary_classifiers['acgm'] = self._create_auxiliary_classifier(drop_rate)
        
        # 特征分析器
        self.feature_analyzer = FeatureAnalyzer(self.conv_features_dim)
        
        print(f"Improved MemoryForensics Network Initialized:")
        print(f"  - Innovation Modules: DSTM={enable_dstm}, MQFF={enable_mqff}, ACGM={enable_acgm}")
        print(f"  - Auxiliary Classifiers: {len(self.auxiliary_classifiers)}")
        print(f"  - Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_innovation_modules(self):
        """初始化创新模块"""
        
        # 1. 专门化的DSTM - 专注于时序记忆
        if self.enable_dstm:
            self.dstm = DynamicSpatioTemporalMemory(
                input_channels=self.conv_features_dim,
                **self.dstm_config
            )
            print(f"✅ DSTM initialized (专注时序记忆)")
        
        # 2. 专门化的MQFF - 专注于多尺度查询
        if self.enable_mqff:
            self.mqff = MultiQueryFeatureFusion(
                input_channels=self.conv_features_dim,
                **self.mqff_config
            )
            print(f"✅ MQFF initialized (专注多尺度查询)")
        
        # 3. 专门化的ACGM - 专注于通道自适应
        if self.enable_acgm:
            if self.acgm_config.get('use_simplified', False):
                self.acgm = SimplifiedAdaptiveChannelGroupingMechanism(
                    input_channels=self.conv_features_dim,
                    num_groups=self.acgm_config['num_groups'],
                    reduction_ratio=self.acgm_config['reduction_ratio']
                )
                print(f"✅ Simplified ACGM initialized (专注通道自适应)")
            else:
                try:
                    self.acgm = AdaptiveChannelGroupingMechanism(
                        input_channels=self.conv_features_dim,
                        num_groups=self.acgm_config['num_groups'],
                        reduction_ratio=self.acgm_config['reduction_ratio'],
                        routing_iterations=self.acgm_config['routing_iterations']
                    )
                    print(f"✅ Full ACGM initialized (专注通道自适应)")
                except Exception as e:
                    print(f"⚠️ Full ACGM failed, using simplified version: {e}")
                    self.acgm = SimplifiedAdaptiveChannelGroupingMechanism(
                        input_channels=self.conv_features_dim,
                        num_groups=self.acgm_config['num_groups'],
                        reduction_ratio=self.acgm_config['reduction_ratio']
                    )
    
    def _create_auxiliary_classifier(self, drop_rate):
        """创建辅助分类器"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(self.conv_features_dim, self.conv_features_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(self.conv_features_dim // 2, self.num_classes)
        )
    
    def forward(self, x, training_stage='full', return_aux_outputs=False, return_analysis=False):
        """
        改进的前向传播 - 兼容原有接口
        
        Args:
            x: 输入图像 (B, 3, 224, 224)
            training_stage: 训练阶段 ('single', 'progressive', 'full')
            return_aux_outputs: 是否返回辅助输出
            return_analysis: 是否返回特征分析
            
        Returns:
            根据参数返回不同的结果组合
        """
        # 1. 骨干网络特征提取
        backbone_features = self.backbone.extract_features(x)  # (B, C, H, W)
        
        # 2. 记录原始特征
        original_features = backbone_features.clone() if return_analysis else None
        
        # 3. 并行模块处理 - 避免累积误差
        module_features = {}
        aux_outputs = {}
        
        # 3.1 并行处理各模块
        if self.enable_dstm:
            dstm_features = self.dstm(backbone_features)  # 直接从骨干特征处理
            module_features['dstm'] = dstm_features
            if return_aux_outputs:
                aux_outputs['dstm'] = self.auxiliary_classifiers['dstm'](dstm_features)
        
        if self.enable_mqff:
            mqff_features = self.mqff(backbone_features)  # 直接从骨干特征处理
            module_features['mqff'] = mqff_features
            if return_aux_outputs:
                aux_outputs['mqff'] = self.auxiliary_classifiers['mqff'](mqff_features)
        
        if self.enable_acgm:
            acgm_features = self.acgm(backbone_features)  # 直接从骨干特征处理
            module_features['acgm'] = acgm_features
            if return_aux_outputs:
                aux_outputs['acgm'] = self.auxiliary_classifiers['acgm'](acgm_features)
        
        # 4. 模块间通信 - 实现协作
        if len(module_features) > 1:
            communicated_features = self.communication_hub(module_features)
        else:
            communicated_features = module_features
        
        # 5. 计算专门化约束损失
        specialization_loss = self.specialization_constraint(module_features)
        
        # 6. 动态权重融合
        fused_features = self.dynamic_fusion(
            backbone_features, communicated_features, training_stage
        )
        
        # 7. 智能分类
        main_output = self.intelligent_classifier(fused_features, communicated_features)
        
        # 8. 特征分析
        analysis = None
        if return_analysis:
            analysis = self.feature_analyzer(
                original_features, fused_features, communicated_features
            )
        
        # 9. 返回结果 - 兼容原有接口
        if return_aux_outputs and return_analysis:
            if self.training:
                return main_output, aux_outputs, analysis, specialization_loss
            else:
                return main_output, aux_outputs, analysis
        elif return_aux_outputs:
            if self.training:
                return main_output, aux_outputs, specialization_loss
            else:
                return main_output, aux_outputs
        elif return_analysis:
            if self.training:
                return main_output, analysis, specialization_loss
            else:
                return main_output, analysis
        else:
            # 最常见的情况 - 只返回主输出（测试时使用）
            return main_output
    
    # 保持原有的其他方法
    def extract_features(self, x):
        """提取多层特征用于分析和可视化"""
        with torch.no_grad():
            backbone_features = self.backbone.extract_features(x)
            
            features = {
                'backbone': backbone_features,
                'modules': {}
            }
            
            # 并行提取各模块特征
            if self.enable_dstm:
                features['modules']['dstm'] = self.dstm(backbone_features)
            if self.enable_mqff:
                features['modules']['mqff'] = self.mqff(backbone_features)
            if self.enable_acgm:
                features['modules']['acgm'] = self.acgm(backbone_features)
            
            # 通信后特征
            if len(features['modules']) > 1:
                features['communicated'] = self.communication_hub(features['modules'])
            else:
                features['communicated'] = features['modules']
            
            # 融合特征
            features['fused'] = self.dynamic_fusion(
                backbone_features, features['communicated'], 'full'
            )
            
            return features
    
    def get_memory_status(self):
        """获取记忆库状态"""
        if self.enable_dstm and hasattr(self.dstm, 'memory_bank'):
            with torch.no_grad():
                return {
                    'memory_bank': self.dstm.memory_bank.data.clone(),
                    'memory_age': self.dstm.memory_age.data.clone(),
                    'memory_quality': self.dstm.memory_quality.data.clone(),
                    'memory_activation': self.dstm.memory_activation.data.clone()
                }
        return None
    
    def reset_memory(self):
        """重置记忆库"""
        if self.enable_dstm and hasattr(self.dstm, '_initialize_memory'):
            self.dstm._initialize_memory()
            print("✅ Memory bank reset successfully")


class ModuleCommunicationHub(nn.Module):
    """模块间通信中心 - 实现模块协作"""
    
    def __init__(self, feature_dim, enabled_modules):
        super(ModuleCommunicationHub, self).__init__()
        
        self.feature_dim = feature_dim
        self.enabled_modules = enabled_modules
        self.num_modules = sum(enabled_modules)
        
        if self.num_modules <= 1:
            return
        
        # 交叉注意力机制 - 实现模块间信息交换
        self.cross_attention = nn.ModuleDict()
        module_names = []
        if enabled_modules[0]:  # DSTM
            module_names.append('dstm')
        if enabled_modules[1]:  # MQFF
            module_names.append('mqff')
        if enabled_modules[2]:  # ACGM
            module_names.append('acgm')
        
        # 为每对模块创建交叉注意力
        for i, name_i in enumerate(module_names):
            for j, name_j in enumerate(module_names):
                if i != j:
                    self.cross_attention[f'{name_i}_to_{name_j}'] = nn.MultiheadAttention(
                        embed_dim=feature_dim,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1
                    )
        
        # 通信权重矩阵
        self.communication_weights = nn.Parameter(
            torch.eye(self.num_modules) + 0.1 * torch.randn(self.num_modules, self.num_modules)
        )
        
        # 信息门控
        self.info_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, module_features):
        """模块间通信"""
        if len(module_features) <= 1:
            return module_features
        
        B, C, H, W = next(iter(module_features.values())).shape
        
        # 转换为序列格式用于注意力计算
        feature_tokens = {}
        for name, features in module_features.items():
            # 使用自适应池化减少计算量
            pooled = F.adaptive_avg_pool2d(features, (4, 4))  # (B, C, 4, 4)
            tokens = pooled.flatten(2).transpose(1, 2)  # (B, 16, C)
            feature_tokens[name] = tokens
        
        # 交叉注意力通信
        communicated_features = {}
        module_names = list(module_features.keys())
        
        for i, name_i in enumerate(module_names):
            enhanced_token = feature_tokens[name_i]
            
            # 与其他模块进行交叉注意力
            for j, name_j in enumerate(module_names):
                if i != j:
                    key = f'{name_i}_to_{name_j}'
                    if key in self.cross_attention:
                        attended, _ = self.cross_attention[key](
                            enhanced_token, feature_tokens[name_j], feature_tokens[name_j]
                        )
                        # 加权融合
                        enhanced_token = enhanced_token + 0.1 * attended
            
            # 转换回空间格式
            enhanced_spatial = enhanced_token.transpose(1, 2).view(B, C, 4, 4)
            enhanced_spatial = F.interpolate(enhanced_spatial, size=(H, W), mode='bilinear', align_corners=False)
            
            # 信息门控
            gate = self.info_gate(F.adaptive_avg_pool2d(enhanced_spatial, 1).flatten(1))
            gate = gate.unsqueeze(-1).unsqueeze(-1)
            
            # 门控融合
            communicated_features[name_i] = gate * enhanced_spatial + (1 - gate) * module_features[name_i]
        
        return communicated_features


class SpecializationConstraint(nn.Module):
    """专门化约束模块 - 强制模块学习不同特征"""
    
    def __init__(self, feature_dim):
        super(SpecializationConstraint, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, module_features):
        """计算专门化约束损失"""
        if len(module_features) <= 1:
            return torch.tensor(0.0, device=next(iter(module_features.values())).device)
        
        # 计算特征表示的正交性
        feature_vectors = []
        for name, features in module_features.items():
            # 转换为全局特征向量
            global_vector = F.adaptive_avg_pool2d(features, 1).flatten(1)
            global_vector = F.normalize(global_vector, dim=1)
            feature_vectors.append(global_vector)
        
        # 计算余弦相似度矩阵
        similarity_loss = 0.0
        count = 0
        
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                # 计算批次内的平均相似度
                similarity = F.cosine_similarity(feature_vectors[i], feature_vectors[j], dim=1)
                similarity_loss += similarity.abs().mean()
                count += 1
        
        return similarity_loss / count if count > 0 else torch.tensor(0.0, device=feature_vectors[0].device)


class DynamicWeightFusion(nn.Module):
    """动态权重融合器 - 自适应调整模块重要性"""
    
    def __init__(self, feature_dim, num_modules):
        super(DynamicWeightFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_modules = num_modules
        
        if num_modules == 0:
            return
        
        # 输入难度评估器
        self.difficulty_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 模块置信度评估器
        self.confidence_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(num_modules + 1, num_modules * 2),
            nn.ReLU(),
            nn.Linear(num_modules * 2, num_modules),
            nn.Softmax(dim=1)
        )
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim)
        )
        
        # 残差门控
        self.residual_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, backbone_features, module_features, training_stage='full'):
        """动态权重融合"""
        if not module_features or self.num_modules == 0:
            return backbone_features
        
        B, C, H, W = backbone_features.shape
        
        # 1. 评估输入难度
        difficulty = self.difficulty_estimator(backbone_features)
        
        # 2. 评估各模块置信度
        confidences = []
        for name, features in module_features.items():
            conf = self.confidence_estimator(features)
            confidences.append(conf)
        
        confidences = torch.cat(confidences, dim=1)
        
        # 3. 生成动态权重
        weight_input = torch.cat([difficulty, confidences], dim=1)
        dynamic_weights = self.weight_generator(weight_input)
        
        # 4. 加权融合模块特征
        fused_features = backbone_features.clone()
        
        for i, (name, features) in enumerate(module_features.items()):
            weight = dynamic_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
            fused_features = fused_features + weight * features
        
        # 5. 特征融合处理
        enhanced_features = self.fusion_conv(fused_features)
        
        # 6. 残差门控
        gate = self.residual_gate(enhanced_features)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        
        final_features = gate * enhanced_features + (1 - gate) * backbone_features
        
        return final_features


class ImprovedIntelligentClassifier(nn.Module):
    """改进的智能分类器"""
    
    def __init__(self, input_channels, num_classes, drop_rate=0.3, num_modules=3):
        super(ImprovedIntelligentClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_modules = num_modules
        
        # 多尺度特征提取
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4)
        ])
        
        # 特征融合
        multi_scale_dim = input_channels * (1 + 4 + 16)
        self.feature_fusion = nn.Sequential(
            nn.Linear(multi_scale_dim, input_channels),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # 模块感知注意力
        if num_modules > 1:
            self.module_attention = nn.Sequential(
                nn.Linear(input_channels, input_channels // 2),
                nn.ReLU(),
                nn.Linear(input_channels // 2, num_modules),
                nn.Softmax(dim=1)
            )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.BatchNorm1d(input_channels // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(input_channels // 2, input_channels // 4),
            nn.BatchNorm1d(input_channels // 4),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.25),
            nn.Linear(input_channels // 4, num_classes)
        )
    
    def forward(self, features, module_features=None):
        """前向传播"""
        # 多尺度特征提取
        multi_scale_features = []
        for pool in self.multi_scale_pools:
            pooled = pool(features)
            flattened = pooled.flatten(1)
            multi_scale_features.append(flattened)
        
        # 特征融合
        combined_features = torch.cat(multi_scale_features, dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 模块感知注意力
        if hasattr(self, 'module_attention') and module_features and len(module_features) > 1:
            attention_weights = self.module_attention(fused_features)
            # 注意力权重可以用于进一步的特征调制
            
        # 分类
        output = self.classifier(fused_features)
        return output


class FeatureAnalyzer(nn.Module):
    """特征分析器"""
    
    def __init__(self, feature_dim):
        super(FeatureAnalyzer, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 特征统计分析
        self.stats_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 4)
        )
        
        # 特征相似性分析
        self.similarity_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, original_features, enhanced_features, module_features):
        """分析特征"""
        analysis = {
            'original_stats': self.stats_analyzer(original_features),
            'enhanced_stats': self.stats_analyzer(enhanced_features),
            'enhancement_similarity': self.similarity_analyzer(enhanced_features - original_features)
        }
        
        for module_name, features in module_features.items():
            analysis[f'{module_name}_stats'] = self.stats_analyzer(features)
        
        return analysis


# 工厂函数和配置保持不变
def create_memory_forensics_net(num_classes=2, drop_rate=0.3,
                               enable_dstm=True, enable_mqff=True, enable_acgm=True,
                               dstm_config=None, mqff_config=None, acgm_config=None):
    """创建改进的MemoryForensics网络"""
    return MemoryForensicsNet(
        num_classes=num_classes,
        drop_rate=drop_rate,
        enable_dstm=enable_dstm,
        enable_mqff=enable_mqff,
        enable_acgm=enable_acgm,
        dstm_config=dstm_config,
        mqff_config=mqff_config,
        acgm_config=acgm_config
    )


# 保持原有的预定义配置
PRESET_CONFIGS = {
    'lightweight': {
        'dstm_config': {
            'memory_size': 64,
            'memory_dim': 128,
            'update_rate': 0.1,
            'temperature': 1.0,
            'update_threshold': 0.8
        },
        'mqff_config': {
            'num_queries': 4,
            'query_dim': 128,
            'num_heads': 4,
            'dropout': 0.1
        },
        'acgm_config': {
            'num_groups': 4,
            'reduction_ratio': 16,
            'routing_iterations': 2,
            'use_simplified': True
        }
    },
    'standard': {
        'dstm_config': {
            'memory_size': 128,
            'memory_dim': 256,
            'update_rate': 0.1,
            'temperature': 1.0,
            'update_threshold': 0.8
        },
        'mqff_config': {
            'num_queries': 6,
            'query_dim': 256,
            'num_heads': 8,
            'dropout': 0.1
        },
        'acgm_config': {
            'num_groups': 8,
            'reduction_ratio': 16,
            'routing_iterations': 3,
            'use_simplified': False
        }
    },
    'high_performance': {
        'dstm_config': {
            'memory_size': 256,
            'memory_dim': 512,
            'update_rate': 0.05,
            'temperature': 0.5,
            'update_threshold': 0.9
        },
        'mqff_config': {
            'num_queries': 8,
            'query_dim': 512,
            'num_heads': 16,
            'dropout': 0.05
        },
        'acgm_config': {
            'num_groups': 16,
            'reduction_ratio': 8,
            'routing_iterations': 5,
            'use_simplified': False
        }
    }
}


def create_preset_model(preset='standard', num_classes=2, drop_rate=0.3,
                       enable_dstm=True, enable_mqff=True, enable_acgm=True):
    """创建预设配置的模型"""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset]
    
    return create_memory_forensics_net(
        num_classes=num_classes,
        drop_rate=drop_rate,
        enable_dstm=enable_dstm,
        enable_mqff=enable_mqff,
        enable_acgm=enable_acgm,
        **config
    )


# 改进的训练策略类
class ImprovedTrainingStrategy:
    """改进的训练策略 - 解决模块协同问题"""
    
    def __init__(self, model, optimizer, device, training_config=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # 训练配置
        self.config = training_config or {
            'specialization_weight': 0.2,
            'auxiliary_weight': 0.3,
            'weight_decay_schedule': True,
            'progressive_training': True
        }
        
        # 损失权重调度
        self.loss_weights = {
            'classification': 1.0,
            'auxiliary': self.config['auxiliary_weight'],
            'specialization': self.config['specialization_weight']
        }
    
    def compute_loss(self, inputs, targets, epoch, total_epochs):
        """计算改进的损失函数"""
        
        # 渐进式训练策略
        if self.config['progressive_training']:
            if epoch < total_epochs * 0.3:
                training_stage = 'specialization'  # 前30%专注专门化
            elif epoch < total_epochs * 0.7:
                training_stage = 'communication'   # 中40%启用通信
            else:
                training_stage = 'full'           # 后30%全面优化
        else:
            training_stage = 'full'
        
        # 前向传播
        if self.model.training:
            main_output, aux_outputs, specialization_loss = self.model(
                inputs, training_stage=training_stage, return_aux_outputs=True
            )
        else:
            main_output = self.model(inputs, training_stage=training_stage)
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
        
        # 专门化权重随训练进程衰减
        if self.config['weight_decay_schedule']:
            dynamic_spec_weight = self.loss_weights['specialization'] * (1 - progress)
        else:
            dynamic_spec_weight = self.loss_weights['specialization']
        
        # 辅助权重在中期最大
        if training_stage == 'communication':
            dynamic_aux_weight = self.loss_weights['auxiliary'] * 1.2
        else:
            dynamic_aux_weight = self.loss_weights['auxiliary']
        
        # 总损失
        total_loss = (
            self.loss_weights['classification'] * main_loss +
            dynamic_aux_weight * aux_loss +
            dynamic_spec_weight * specialization_loss
        )
        
        return total_loss, {
            'main_loss': main_loss.item(),
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            'spec_loss': specialization_loss.item(),
            'training_stage': training_stage,
            'spec_weight': dynamic_spec_weight,
            'aux_weight': dynamic_aux_weight
        }
    
    def should_update_memory(self, epoch, total_epochs):
        """判断是否应该更新记忆库"""
        # 在训练前期更频繁地更新记忆库
        if epoch < total_epochs * 0.5:
            return True
        else:
            return epoch % 2 == 0  # 后期每隔一个epoch更新


# 保持原有的特征融合模块，但简化实现
class FeatureFusionModule(nn.Module):
    """简化的特征融合模块"""
    
    def __init__(self, input_channels, enable_dstm=True, enable_mqff=True, enable_acgm=True):
        super(FeatureFusionModule, self).__init__()
        
        self.enable_dstm = enable_dstm
        self.enable_mqff = enable_mqff
        self.enable_acgm = enable_acgm
        
        # 简化的融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )
        
        # 残差门控
        self.residual_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, enhanced_features, original_features, module_features=None):
        """特征融合"""
        fused = self.fusion_conv(enhanced_features)
        gate = self.residual_gate(fused)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        
        output = gate * fused + (1 - gate) * original_features
        return output


# 测试函数
def test_improved_model():
    """测试改进的模型"""
    print("🧪 Testing Improved MemoryForensics Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试标准配置
    print("\n🔬 Testing Improved Configuration...")
    try:
        model = create_preset_model('standard', num_classes=2).to(device)
        
        # 测试输入
        x = torch.randn(2, 3, 224, 224).to(device)
        
        # 测试基本前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)
            print(f"✅ Basic forward pass: {output.shape}")
        
        # 测试训练模式
        model.train()
        main_output, aux_outputs, spec_loss = model(x, return_aux_outputs=True)
        print(f"✅ Training mode - Main: {main_output.shape}, Aux: {list(aux_outputs.keys())}")
        print(f"✅ Specialization loss: {spec_loss.item():.4f}")
        
        # 测试改进的训练策略
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = ImprovedTrainingStrategy(model, optimizer, device)
        
        targets = torch.randint(0, 2, (2,)).to(device)
        total_loss, loss_dict = trainer.compute_loss(x, targets, epoch=0, total_epochs=100)
        
        print(f"✅ Improved training strategy:")
        print(f"  - Total loss: {total_loss.item():.4f}")
        print(f"  - Main loss: {loss_dict['main_loss']:.4f}")
        print(f"  - Aux loss: {loss_dict['aux_loss']:.4f}")
        print(f"  - Spec loss: {loss_dict['spec_loss']:.4f}")
        print(f"  - Training stage: {loss_dict['training_stage']}")
        
        # 测试反向传播
        total_loss.backward()
        optimizer.step()
        print("✅ Backward pass successful!")
        
        # 测试特征提取
        model.eval()
        with torch.no_grad():
            features = model.extract_features(x)
            print(f"✅ Feature extraction: {list(features.keys())}")
        
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Improved model testing completed!")


if __name__ == '__main__':
    test_improved_model()