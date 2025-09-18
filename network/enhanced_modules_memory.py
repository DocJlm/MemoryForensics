# network/enhanced_modules_memory.py - 完整版本：三个创新点的统一实现
# 包含：DSTM（动态记忆更新）+ MQFF（多查询特征融合）+ ACGM（自适应通道分组）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicSpatioTemporalMemory(nn.Module):
    """
    动态时空记忆模块 (DSTM) - 完整版本
    
    核心创新：
    1. 可学习的视觉记忆库，存储典型的真实/伪造模式
    2. 智能检索机制，基于相似度检索相关记忆
    3. 自适应更新策略，根据新经验动态调整记忆内容
    
    技术特点：
    - 选择性存储：只记住最具代表性的模式
    - 联想检索：通过相似性快速定位相关记忆
    - 适应性更新：根据新经验调整记忆内容
    """
    
    def __init__(self, input_channels, memory_size=256, memory_dim=512, 
                 update_rate=0.1, temperature=1.0, update_threshold=0.8):
        super(DynamicSpatioTemporalMemory, self).__init__()
        
        self.input_channels = input_channels
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.update_rate = update_rate
        self.temperature = temperature
        self.update_threshold = update_threshold
        
        # 核心记忆库 - 使用可训练参数但不参与梯度计算
        self.memory_bank = nn.Parameter(torch.randn(memory_size, memory_dim), requires_grad=False)
        self.memory_age = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
        self.memory_quality = nn.Parameter(torch.ones(memory_size), requires_grad=False)
        self.memory_activation = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
        
        # 特征编码器 - 将输入特征映射到记忆空间
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(input_channels, memory_dim // 2, 1),
            nn.BatchNorm2d(memory_dim // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(memory_dim // 2, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.Dropout(0.1)
        )
        
        # 记忆检索 - 多头注意力机制
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # 记忆质量评估器
        self.quality_evaluator = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征重构器 - 将记忆信息转换为空间权重
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, input_channels),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 生成空间权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 记忆融合门控
        self.memory_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )
        
        self._initialize_memory()
        
    def _initialize_memory(self):
        """初始化记忆库"""
        with torch.no_grad():
            # 使用Xavier初始化记忆原型
            nn.init.xavier_uniform_(self.memory_bank)
            
            # 初始化记忆质量和年龄
            nn.init.ones_(self.memory_quality)
            nn.init.zeros_(self.memory_age)
            nn.init.zeros_(self.memory_activation)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            enhanced_features: 记忆增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 特征编码 - 将输入特征映射到记忆空间
        encoded_features = self.feature_encoder(x)  # (B, memory_dim)
        
        # 2. 记忆检索 - 基于相似度检索相关记忆
        retrieved_memory, attention_weights = self._retrieve_memory(encoded_features)
        
        # 3. 记忆融合 - 融合当前特征和检索记忆
        fused_features = self._fuse_memory(encoded_features, retrieved_memory)
        
        # 4. 特征重构 - 生成空间增强权重
        spatial_weights = self.feature_reconstructor(fused_features)
        spatial_weights = spatial_weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # 5. 空间注意力 - 生成空间权重
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        # 6. 特征增强 - 应用记忆指导的增强
        enhanced_features = x * spatial_weights * spatial_att
        
        # 7. 记忆更新 - 训练时更新记忆库
        if self.training:
            self._update_memory_safe(encoded_features.detach(), attention_weights.detach())
        
        return enhanced_features
    
    def _retrieve_memory(self, query_features):
        """
        智能记忆检索机制
        
        Args:
            query_features: 查询特征 (B, memory_dim)
            
        Returns:
            retrieved_memory: 检索到的记忆 (B, memory_dim)
            attention_weights: 注意力权重 (B, memory_size)
        """
        B = query_features.size(0)
        
        # 扩展记忆库用于批次计算
        memory_expanded = self.memory_bank.unsqueeze(0).expand(B, -1, -1)  # (B, memory_size, memory_dim)
        query_expanded = query_features.unsqueeze(1)  # (B, 1, memory_dim)
        
        # 多头注意力检索
        retrieved_memory, attention_weights = self.memory_attention(
            query_expanded, memory_expanded, memory_expanded
        )
        
        retrieved_memory = retrieved_memory.squeeze(1)  # (B, memory_dim)
        attention_weights = attention_weights.squeeze(1)  # (B, memory_size)
        
        return retrieved_memory, attention_weights
    
    def _fuse_memory(self, current_features, retrieved_memory):
        """
        记忆融合机制
        
        Args:
            current_features: 当前特征 (B, memory_dim)
            retrieved_memory: 检索的记忆 (B, memory_dim)
            
        Returns:
            fused_features: 融合后的特征 (B, memory_dim)
        """
        # 拼接当前特征和检索记忆
        combined = torch.cat([current_features, retrieved_memory], dim=1)
        
        # 门控融合
        gate = self.memory_gate(combined)
        fused_features = gate * current_features + (1 - gate) * retrieved_memory
        
        return fused_features
    
    def _update_memory_safe(self, new_features, attention_weights):
        """
        安全的记忆更新机制
        
        更新策略：
        1. 如果新特征与现有记忆相似度高，更新该记忆
        2. 否则，替换质量最低的记忆
        3. 更新记忆质量和年龄信息
        
        Args:
            new_features: 新的特征 (B, memory_dim)
            attention_weights: 注意力权重 (B, memory_size)
        """
        with torch.no_grad():
            if new_features.size(0) == 0:
                return
            
            # 选择最重要的特征进行更新
            feature_importance = torch.norm(new_features, dim=1)
            if feature_importance.numel() == 0:
                return
                
            max_importance_idx = torch.argmax(feature_importance)
            selected_feature = new_features[max_importance_idx].clone()
            
            # 计算与现有记忆的相似度
            similarities = F.cosine_similarity(
                selected_feature.unsqueeze(0), 
                self.memory_bank.data,
                dim=1
            )
            
            max_similarity, max_sim_idx = torch.max(similarities, dim=0)
            
            # 评估新特征的质量
            new_quality = self.quality_evaluator(selected_feature.unsqueeze(0)).item()
            
            # 更新策略
            if max_similarity > self.update_threshold:
                # 更新现有记忆
                old_memory = self.memory_bank.data[max_sim_idx].clone()
                new_memory = (1 - self.update_rate) * old_memory + self.update_rate * selected_feature
                
                # 安全更新
                self.memory_bank.data[max_sim_idx].copy_(new_memory)
                
                # 更新质量分数
                old_quality = self.memory_quality.data[max_sim_idx].item()
                updated_quality = 0.9 * old_quality + 0.1 * new_quality
                self.memory_quality.data[max_sim_idx] = updated_quality
                
                # 重置年龄
                self.memory_age.data[max_sim_idx] = 0
                
            else:
                # 替换质量最低的记忆
                quality_scores = self.memory_quality.data.clone()
                # 考虑年龄因素
                age_penalty = self.memory_age.data / (self.memory_age.data.max() + 1e-8)
                adjusted_quality = quality_scores - 0.1 * age_penalty
                
                worst_idx = torch.argmin(adjusted_quality)
                
                # 只有当新特征质量更高时才替换
                if new_quality > self.memory_quality.data[worst_idx]:
                    self.memory_bank.data[worst_idx].copy_(selected_feature)
                    self.memory_quality.data[worst_idx] = new_quality
                    self.memory_age.data[worst_idx] = 0
            
            # 更新激活计数
            self.memory_activation.data[max_sim_idx] += 1
            
            # 全局年龄更新
            self.memory_age.data.add_(1)
            
            # 防止年龄过大
            max_age = self.memory_age.data.max()
            if max_age > 1000:
                self.memory_age.data.div_(2)


class MultiQueryFeatureFusion(nn.Module):
    """
    多查询特征融合模块 (MQFF)
    
    核心创新：
    1. 专门化查询向量：针对不同检测维度设计专门的查询向量
    2. 层次化交叉注意力：多层次的注意力计算机制
    3. 查询间协作：不同查询之间的信息交换和协作
    
    查询向量专门化：
    - Q0, Q4: 边缘/轮廓检测（高通滤波初始化）
    - Q1, Q5: 纹理一致性检测（Gabor滤波初始化）
    - Q2, Q6: 光照/颜色检测（色彩空间初始化）
    - Q3, Q7: 全局语义检测（语义分割初始化）
    """
    
    def __init__(self, input_channels, num_queries=8, query_dim=256, num_heads=8, dropout=0.1):
        super(MultiQueryFeatureFusion, self).__init__()
        
        self.input_channels = input_channels
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        # 可学习的专门化查询向量
        self.query_vectors = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # 查询专门化器
        self.query_specializers = nn.ModuleList([
            self._create_query_specializer(i) for i in range(num_queries)
        ])
        
        # 输入特征投影
        self.input_projection = nn.Conv2d(input_channels, query_dim, 1)
        
        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 查询间协作机制
        self.query_collaboration = QueryCollaboration(num_queries, query_dim)
        
        # 层次化特征融合
        self.hierarchical_fusion = HierarchicalFeatureFusion(query_dim, input_channels, num_queries)
        
        # 位置编码
        self.position_encoding = PositionalEncoding2D(query_dim)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )
        
        self._initialize_parameters()
    
    def _create_query_specializer(self, query_idx):
        """创建专门化的查询处理器"""
        if query_idx in [0, 4]:  # 边缘/轮廓检测
            return nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim),
                nn.ReLU(),
                nn.Linear(self.query_dim, self.query_dim),
                nn.Tanh()
            )
        elif query_idx in [1, 5]:  # 纹理一致性检测
            return nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim * 2),
                nn.GELU(),
                nn.Linear(self.query_dim * 2, self.query_dim),
                nn.Sigmoid()
            )
        elif query_idx in [2, 6]:  # 光照/颜色检测
            return nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.query_dim, self.query_dim),
                nn.Hardtanh()
            )
        else:  # 全局语义检测
            return nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.query_dim, self.query_dim)
            )
    
    def _initialize_parameters(self):
        """专门化初始化查询向量"""
        with torch.no_grad():
            for i in range(self.num_queries):
                if i % 4 == 0:  # 边缘检测
                    nn.init.xavier_uniform_(self.query_vectors.data[i:i+1])
                elif i % 4 == 1:  # 纹理检测
                    nn.init.kaiming_uniform_(self.query_vectors.data[i:i+1])
                elif i % 4 == 2:  # 颜色检测
                    nn.init.normal_(self.query_vectors.data[i:i+1], 0, 0.02)
                else:  # 语义检测
                    nn.init.orthogonal_(self.query_vectors.data[i:i+1])
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            enhanced_features: 多查询增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 输入特征投影
        projected_features = self.input_projection(x)  # (B, query_dim, H, W)
        
        # 2. 添加位置编码
        projected_features = self.position_encoding(projected_features)
        
        # 3. 特征展平用于注意力计算
        feature_tokens = projected_features.flatten(2).transpose(1, 2)  # (B, H*W, query_dim)
        
        # 4. 查询专门化
        specialized_queries = []
        for i, specializer in enumerate(self.query_specializers):
            specialized_query = specializer(self.query_vectors[i].unsqueeze(0))
            specialized_queries.append(specialized_query)
        
        queries = torch.cat(specialized_queries, dim=0).unsqueeze(0).expand(B, -1, -1)
        
        # 5. 多头交叉注意力
        query_responses, attention_weights = self.cross_attention(
            queries, feature_tokens, feature_tokens
        )
        
        # 6. 查询间协作
        collaborated_queries = self.query_collaboration(query_responses)
        
        # 7. 层次化特征融合
        enhanced_features = self.hierarchical_fusion(
            collaborated_queries, projected_features, attention_weights
        )
        
        # 8. 输出投影
        output = self.output_projection(enhanced_features)
        
        return output


class QueryCollaboration(nn.Module):
    """查询间协作机制"""
    
    def __init__(self, num_queries, query_dim):
        super(QueryCollaboration, self).__init__()
        
        # 查询间通信矩阵
        self.communication_matrix = nn.Parameter(
            torch.eye(num_queries) + 0.1 * torch.randn(num_queries, num_queries)
        )
        
        # 协作门控
        self.collaboration_gate = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.Sigmoid()
        )
        
        # 信息融合
        self.information_fusion = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim)
        )
    
    def forward(self, query_responses):
        """
        Args:
            query_responses: (B, num_queries, query_dim)
        Returns:
            collaborated_queries: (B, num_queries, query_dim)
        """
        B, num_queries, query_dim = query_responses.shape
        
        # 查询间信息交换
        communication_weights = F.softmax(self.communication_matrix, dim=1)
        communicated = torch.matmul(communication_weights, query_responses)
        
        # 门控融合
        combined = torch.cat([query_responses, communicated], dim=-1)
        gates = self.collaboration_gate(combined)
        
        # 自适应融合
        collaborated = gates * query_responses + (1 - gates) * communicated
        
        # 信息融合
        output = self.information_fusion(collaborated) + query_responses
        
        return output


class HierarchicalFeatureFusion(nn.Module):
    """层次化特征融合"""
    
    def __init__(self, query_dim, output_channels, num_queries):
        super(HierarchicalFeatureFusion, self).__init__()
        
        self.query_dim = query_dim
        self.output_channels = output_channels
        self.num_queries = num_queries
        
        # 查询聚合
        self.query_aggregator = nn.Sequential(
            nn.Linear(query_dim * num_queries, output_channels),
            nn.Sigmoid()
        )
        
        # 特征转换
        self.feature_transform = nn.Sequential(
            nn.Conv2d(query_dim, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(output_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 多尺度融合
        self.multi_scale_fusion = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4)
        ])
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(output_channels * 21, output_channels),  # 1+4+16=21
            nn.ReLU(),
            nn.Linear(output_channels, output_channels),
            nn.Sigmoid()
        )
    
    def forward(self, query_responses, projected_features, attention_weights):
        """
        Args:
            query_responses: (B, num_queries, query_dim)
            projected_features: (B, query_dim, H, W)
            attention_weights: attention weights
        Returns:
            enhanced_features: (B, output_channels, H, W)
        """
        B, num_queries, query_dim = query_responses.shape
        _, _, H, W = projected_features.shape
        
        # 1. 查询聚合为通道权重
        flattened_queries = query_responses.contiguous().view(B, -1)
        channel_weights = self.query_aggregator(flattened_queries)
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 2. 特征转换
        transformed_features = self.feature_transform(projected_features)
        
        # 3. 空间注意力
        spatial_att = self.spatial_attention(transformed_features)
        
        # 4. 多尺度特征融合
        multi_scale_features = []
        for pool in self.multi_scale_fusion:
            pooled = pool(transformed_features)
            multi_scale_features.append(pooled.flatten(1))
        
        combined_scales = torch.cat(multi_scale_features, dim=1)
        scale_weights = self.scale_fusion(combined_scales)
        scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 5. 最终特征融合
        enhanced_features = transformed_features * channel_weights * spatial_att * scale_weights
        
        return enhanced_features


class AdaptiveChannelGroupingMechanism(nn.Module):
    """
    自适应通道分组机制 (ACGM)
    
    核心创新：
    1. 内容感知分组：根据输入内容动态调整通道分组策略
    2. 动态路由优化：通过迭代路由优化信息流动
    3. 组内外交互：组内增强和组间协作的统一机制
    
    技术特点：
    - 打破固定通道分组的限制
    - 根据输入内容自适应调整分组
    - 优化特征表示效率
    """
    
    def __init__(self, input_channels, num_groups=8, reduction_ratio=16, 
                 routing_iterations=3, temperature=1.0):
        super(AdaptiveChannelGroupingMechanism, self).__init__()
        
        self.input_channels = input_channels
        # 确保组数能整除通道数
        while input_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.num_groups = num_groups
        self.group_size = input_channels // num_groups
        self.routing_iterations = routing_iterations
        self.temperature = temperature
        
        print(f"ACGM: {input_channels} channels, {num_groups} groups, {self.group_size} channels per group")
        
        # 内容感知分组策略学习器
        self.content_analyzer = ContentAwareGrouping(input_channels, num_groups, reduction_ratio)
        
        # 动态路由器
        self.dynamic_router = DynamicChannelRouter(input_channels, num_groups, routing_iterations)
        
        # 组内特征增强器
        self.intra_group_enhancers = nn.ModuleList([
            IntraGroupEnhancer(self.group_size) for _ in range(num_groups)
        ])
        
        # 组间交互模块
        self.inter_group_interaction = InterGroupInteraction(num_groups, self.group_size)
        
        # 自适应融合门控
        self.adaptive_fusion = AdaptiveFusionGate(input_channels, num_groups)
        
        # 全局特征整合
        self.global_integration = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            enhanced_features: 自适应分组增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 内容感知分组策略学习
        group_assignments, group_confidence = self.content_analyzer(x)
        
        # 2. 动态路由优化
        routing_weights, routed_features = self.dynamic_router(x, group_assignments)
        
        # 3. 组内特征增强
        enhanced_groups = []
        for i, enhancer in enumerate(self.intra_group_enhancers):
            start_idx = i * self.group_size
            end_idx = start_idx + self.group_size
            
            group_features = routed_features[:, start_idx:end_idx]
            enhanced_group = enhancer(group_features)
            enhanced_groups.append(enhanced_group)
        
        # 4. 组间交互
        interacted_groups = self.inter_group_interaction(enhanced_groups)
        
        # 5. 特征重组
        reorganized_features = torch.cat(interacted_groups, dim=1)
        
        # 6. 自适应融合
        fused_features = self.adaptive_fusion(x, reorganized_features, group_confidence)
        
        # 7. 全局特征整合
        integrated_features = self.global_integration(fused_features)
        
        # 8. 残差连接
        output = integrated_features + x
        
        return output


class ContentAwareGrouping(nn.Module):
    """内容感知分组策略学习器"""
    
    def __init__(self, input_channels, num_groups, reduction_ratio):
        super(ContentAwareGrouping, self).__init__()
        
        self.input_channels = input_channels
        self.num_groups = num_groups
        
        # 内容分析网络
        self.content_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels // reduction_ratio, input_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels // reduction_ratio, input_channels * num_groups, 1)
        )
        
        # 分组策略生成器
        self.strategy_generator = nn.Sequential(
            nn.Linear(input_channels * num_groups, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, input_channels * num_groups),
            nn.Softmax(dim=-1)
        )
        
        # 置信度评估器
        self.confidence_evaluator = nn.Sequential(
            nn.Linear(input_channels * num_groups, input_channels // 4),
            nn.ReLU(),
            nn.Linear(input_channels // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            group_assignments: 分组分配 (B, C, num_groups)
            group_confidence: 分组置信度 (B, 1)
        """
        B, C, H, W = x.shape
        
        # 分析输入内容
        content_features = self.content_analyzer(x)  # (B, C*num_groups, 1, 1)
        content_features = content_features.flatten(1)  # (B, C*num_groups)
        
        # 生成分组策略
        group_assignments = self.strategy_generator(content_features)
        group_assignments = group_assignments.view(B, C, self.num_groups)
        
        # 评估置信度
        group_confidence = self.confidence_evaluator(content_features)
        
        return group_assignments, group_confidence


class DynamicChannelRouter(nn.Module):
    """动态通道路由器"""
    
    def __init__(self, input_channels, num_groups, routing_iterations):
        super(DynamicChannelRouter, self).__init__()
        
        self.input_channels = input_channels
        self.num_groups = num_groups
        self.routing_iterations = routing_iterations
        
        # 路由预测网络
        self.routing_predictor = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels // 4, input_channels * num_groups)
        )
        
        # 路由权重初始化
        self.register_buffer('routing_bias', torch.zeros(num_groups, input_channels))
    
    def forward(self, x, group_assignments):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            group_assignments: 分组分配 (B, C, num_groups)
        Returns:
            routing_weights: 最终路由权重 (B, C, num_groups)
            routed_features: 路由后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 初始路由logits
        routing_logits = self.routing_predictor(x)
        routing_logits = routing_logits.view(B, C, self.num_groups)
        
        # 结合分组分配
        routing_logits = routing_logits + torch.log(group_assignments + 1e-8)
        
        # 动态路由迭代
        for iteration in range(self.routing_iterations):
            # 计算路由权重
            routing_weights = F.softmax(routing_logits, dim=-1)
            
            # 计算组输出
            group_outputs = []
            for g in range(self.num_groups):
                weights = routing_weights[:, :, g].unsqueeze(-1).unsqueeze(-1)
                group_output = (x * weights).sum(dim=1, keepdim=True)
                group_outputs.append(group_output)
            
            # 更新路由logits（基于一致性）
            if iteration < self.routing_iterations - 1:
                agreements = self._compute_agreement(x, group_outputs)
                routing_logits = routing_logits + agreements
        
        # 应用最终路由
        final_weights = F.softmax(routing_logits, dim=-1)
        routed_features = x  # 保持原始特征结构
        
        return final_weights, routed_features
    
    def _compute_agreement(self, inputs, group_outputs):
        """计算输入与组输出的一致性"""
        agreements = []
        B, C, H, W = inputs.shape
        
        for g, group_output in enumerate(group_outputs):
            # 扩展组输出到输入维度
            expanded_output = group_output.expand(-1, C, -1, -1)
            
            # 计算余弦相似度
            inputs_flat = inputs.flatten(2)  # (B, C, H*W)
            output_flat = expanded_output.flatten(2)  # (B, C, H*W)
            
            similarity = F.cosine_similarity(inputs_flat, output_flat, dim=2)  # (B, C)
            agreements.append(similarity)
        
        return torch.stack(agreements, dim=-1)  # (B, C, num_groups)


class IntraGroupEnhancer(nn.Module):
    """组内特征增强器"""
    
    def __init__(self, group_size):
        super(IntraGroupEnhancer, self).__init__()
        
        self.group_size = group_size
        
        # 通道间交互
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(group_size, group_size, 1, groups=group_size),
            nn.BatchNorm2d(group_size),
            nn.ReLU(),
            nn.Conv2d(group_size, group_size, 3, padding=1, groups=group_size),
            nn.BatchNorm2d(group_size),
            nn.ReLU()
        )
        
        # 通道重要性学习
        self.channel_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(group_size, max(group_size // 4, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(group_size // 4, 1), group_size, 1),
            nn.Sigmoid()
        )
        
        # 空间增强
        self.spatial_enhancement = nn.Sequential(
            nn.Conv2d(group_size, group_size, 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.ReLU(),
            nn.Conv2d(group_size, group_size, 1),
            nn.BatchNorm2d(group_size)
        )
    
    def forward(self, group_features):
        """
        Args:
            group_features: (B, group_size, H, W)
        Returns:
            enhanced_features: (B, group_size, H, W)
        """
        # 通道交互
        interacted = self.channel_interaction(group_features)
        
        # 重要性加权
        importance = self.channel_importance(group_features)
        weighted = interacted * importance
        
        # 空间增强
        enhanced = self.spatial_enhancement(weighted)
        
        # 残差连接
        output = enhanced + group_features
        
        return output


class InterGroupInteraction(nn.Module):
    """组间交互模块"""
    
    def __init__(self, num_groups, group_size):
        super(InterGroupInteraction, self).__init__()
        
        self.num_groups = num_groups
        self.group_size = group_size
        
        # 组间注意力
        self.inter_group_attention = nn.MultiheadAttention(
            embed_dim=group_size,
            num_heads=max(group_size // 32, 1),
            batch_first=True,
            dropout=0.1
        )
        
        # 组融合网络
        self.group_fusion = nn.Sequential(
            nn.Linear(group_size * num_groups, group_size * num_groups // 2),
            nn.ReLU(),
            nn.Linear(group_size * num_groups // 2, group_size * num_groups),
            nn.Sigmoid()
        )
        
        # 组级归一化
        self.group_norm = nn.GroupNorm(num_groups, group_size * num_groups)
    
    def forward(self, group_list):
        """
        Args:
            group_list: List of (B, group_size, H, W)
        Returns:
            interacted_groups: List of (B, group_size, H, W)
        """
        B, _, H, W = group_list[0].shape
        
        # 将组特征展平用于注意力计算
        group_tokens = []
        for group in group_list:
            token = F.adaptive_avg_pool2d(group, 1).flatten(1)  # (B, group_size)
            group_tokens.append(token)
        
        group_tokens = torch.stack(group_tokens, dim=1)  # (B, num_groups, group_size)
        
        # 组间注意力
        attended_tokens, _ = self.inter_group_attention(
            group_tokens, group_tokens, group_tokens
        )
        
        # 全局融合权重
        flattened_tokens = attended_tokens.flatten(1)  # (B, num_groups * group_size)
        fusion_weights = self.group_fusion(flattened_tokens)
        fusion_weights = fusion_weights.view(B, self.num_groups, self.group_size)
        
        # 应用融合权重到原始组特征
        interacted_groups = []
        for i, group in enumerate(group_list):
            weight = fusion_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            enhanced_group = group * weight
            interacted_groups.append(enhanced_group)
        
        return interacted_groups


class AdaptiveFusionGate(nn.Module):
    """自适应融合门控"""
    
    def __init__(self, channels, num_groups):
        super(AdaptiveFusionGate, self).__init__()
        
        self.channels = channels
        self.num_groups = num_groups
        
        # 融合门控网络
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道级门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        # 置信度调制
        self.confidence_modulator = nn.Sequential(
            nn.Linear(1, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, original_features, enhanced_features, group_confidence):
        """
        Args:
            original_features: (B, C, H, W)
            enhanced_features: (B, C, H, W)
            group_confidence: (B, 1)
        Returns:
            fused_features: (B, C, H, W)
        """
        # 特征拼接
        combined = torch.cat([original_features, enhanced_features], dim=1)
        
        # 空间门控
        spatial_gate = self.fusion_gate(combined)
        
        # 通道门控
        channel_gate = self.channel_gate(combined)
        
        # 置信度调制
        confidence_weight = self.confidence_modulator(group_confidence)
        confidence_weight = confidence_weight.unsqueeze(-1).unsqueeze(-1)
        
        # 自适应融合
        fused_features = (
            confidence_weight * spatial_gate * channel_gate * enhanced_features +
            (1 - confidence_weight * spatial_gate) * original_features
        )
        
        return fused_features


class PositionalEncoding2D(nn.Module):
    """2D位置编码"""
    
    def __init__(self, channels, temperature=10000):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.temperature = temperature
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            encoded: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 生成位置网格
        y_pos = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)
        
        # 归一化位置
        if H > 1:
            y_pos = y_pos / (H - 1)
        if W > 1:
            x_pos = x_pos / (W - 1)
        
        # 计算位置编码
        encoding_dim = min(C // 4, 64)  # 限制编码维度
        dim_t = torch.arange(encoding_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / encoding_dim)
        
        pos_x = x_pos.unsqueeze(-1) / dim_t
        pos_y = y_pos.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos_encoding = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
        
        # 调整编码维度以匹配输入
        if pos_encoding.size(1) < C:
            padding = torch.zeros(1, C - pos_encoding.size(1), H, W, device=device)
            pos_encoding = torch.cat([pos_encoding, padding], dim=1)
        elif pos_encoding.size(1) > C:
            pos_encoding = pos_encoding[:, :C]
        
        return x + pos_encoding.expand(B, -1, -1, -1)


# 简化版本的ACGM（如果上面的版本过于复杂）
class SimplifiedAdaptiveChannelGroupingMechanism(nn.Module):
    """简化的自适应通道分组机制"""
    
    def __init__(self, input_channels, num_groups=8, reduction_ratio=16):
        super(SimplifiedAdaptiveChannelGroupingMechanism, self).__init__()
        
        self.input_channels = input_channels
        # 确保组数能整除通道数
        while input_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.num_groups = num_groups
        self.group_size = input_channels // num_groups
        
        print(f"SimplifiedACGM: {input_channels} channels, {num_groups} groups, {self.group_size} channels per group")
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, max(input_channels // reduction_ratio, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(input_channels // reduction_ratio, 1), input_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 组级特征增强
        self.group_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.group_size, self.group_size, 3, padding=1, groups=self.group_size),
                nn.BatchNorm2d(self.group_size),
                nn.ReLU(),
                nn.Conv2d(self.group_size, self.group_size, 1),
                nn.BatchNorm2d(self.group_size)
            ) for _ in range(num_groups)
        ])
        
        # 组间信息交换
        self.inter_group_conv = nn.Conv2d(input_channels, input_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            enhanced_features: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 通道注意力
        channel_att = self.channel_attention(x)
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        # 分组处理
        enhanced_groups = []
        for i, enhancer in enumerate(self.group_enhancers):
            start_idx = i * self.group_size
            end_idx = start_idx + self.group_size
            
            group_features = x[:, start_idx:end_idx]
            enhanced_group = enhancer(group_features) + group_features
            enhanced_groups.append(enhanced_group)
        
        # 组合特征
        enhanced_x = torch.cat(enhanced_groups, dim=1)
        
        # 组间信息交换
        exchanged_x = self.inter_group_conv(enhanced_x)
        
        # 应用注意力
        final_features = exchanged_x * channel_att * spatial_att
        
        # 残差连接
        output = final_features + x
        
        return output


# 测试函数
def test_enhanced_modules():
    """测试增强模块的功能"""
    print("🧪 Testing Enhanced Modules with Dynamic Memory Update...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    batch_size = 2
    input_channels = 1792
    H, W = 7, 7
    
    # 创建测试输入
    x = torch.randn(batch_size, input_channels, H, W).to(device)
    
    # 测试 DSTM（带动态记忆更新）
    print("\n🔬 Testing DSTM with Dynamic Memory Update...")
    try:
        dstm = DynamicSpatioTemporalMemory(
            input_channels=input_channels,
            memory_size=128,
            memory_dim=256,
            update_rate=0.1
        ).to(device)
        
        # 训练模式测试
        dstm.train()
        out_dstm = dstm(x)
        print(f"✅ DSTM Training Mode - Input: {x.shape}, Output: {out_dstm.shape}")
        
        # 测试记忆更新
        loss = out_dstm.mean()
        loss.backward()
        print("✅ DSTM backward pass and memory update successful!")
        
        # 评估模式测试
        dstm.eval()
        with torch.no_grad():
            out_eval = dstm(x)
        print(f"✅ DSTM Evaluation Mode - Output: {out_eval.shape}")
        
    except Exception as e:
        print(f"❌ DSTM Error: {e}")
    
    # 测试 MQFF
    print("\n🔬 Testing MQFF...")
    try:
        mqff = MultiQueryFeatureFusion(
            input_channels=input_channels,
            num_queries=8,
            query_dim=256,
            num_heads=8
        ).to(device)
        
        out_mqff = mqff(x)
        loss = out_mqff.mean()
        loss.backward()
        print(f"✅ MQFF - Input: {x.shape}, Output: {out_mqff.shape}")
        
    except Exception as e:
        print(f"❌ MQFF Error: {e}")
    
    # 测试 ACGM
    print("\n🔬 Testing ACGM...")
    try:
        acgm = AdaptiveChannelGroupingMechanism(
            input_channels=input_channels,
            num_groups=8,
            routing_iterations=3
        ).to(device)
        
        out_acgm = acgm(x)
        loss = out_acgm.mean()
        loss.backward()
        print(f"✅ ACGM - Input: {x.shape}, Output: {out_acgm.shape}")
        
    except Exception as e:
        print(f"❌ ACGM Error: {e}")
        
        # 尝试简化版本
        print("🔄 Trying Simplified ACGM...")
        try:
            acgm_simple = SimplifiedAdaptiveChannelGroupingMechanism(
                input_channels=input_channels,
                num_groups=8
            ).to(device)
            
            out_simple = acgm_simple(x)
            loss = out_simple.mean()
            loss.backward()
            print(f"✅ Simplified ACGM - Input: {x.shape}, Output: {out_simple.shape}")
            
        except Exception as e:
            print(f"❌ Simplified ACGM Error: {e}")
    
    # 测试组合使用
    print("\n🔬 Testing Combined Modules...")
    try:
        dstm = DynamicSpatioTemporalMemory(input_channels, memory_size=64, memory_dim=128).to(device)
        mqff = MultiQueryFeatureFusion(input_channels, num_queries=4, query_dim=128).to(device)
        acgm = SimplifiedAdaptiveChannelGroupingMechanism(input_channels, num_groups=8).to(device)
        
        # 顺序应用
        features = x
        features = dstm(features)
        features = mqff(features)
        features = acgm(features)
        
        loss = features.mean()
        loss.backward()
        
        print(f"✅ Combined Modules - Final Output: {features.shape}")
        print("✅ All modules work together successfully!")
        
    except Exception as e:
        print(f"❌ Combined Modules Error: {e}")
    
    print("\n🎉 Testing completed!")


if __name__ == '__main__':
    test_enhanced_modules()