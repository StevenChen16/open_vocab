import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenVocabDetect(nn.Module):
    """
    开放词汇目标检测头，替代YOLO原始Detect层
    兼容ultralytics框架，处理多尺度特征图
    
    该模块分为两个主要分支：
    1. 边界框回归分支：与传统YOLO相同，预测坐标和置信度
    2. 语义特征分支：输出语义特征向量，用于开放词汇类别匹配
    """
    def __init__(self, nc=80, anchors=(), ch=(), semantic_dim=512, inplace=True):
        super().__init__()
        self.nc = nc                 # 类别数量 (用于兼容YOLO，实际使用语义向量)
        self.no = 4 + 1              # 输出通道数 (x,y,w,h,conf)
        self.nl = len(anchors)       # 特征图数量
        self.na = len(anchors[0]) // 2  # 每个位置的锚点数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格
        self.semantic_dim = semantic_dim  # 语义向量维度
        self.inplace = inplace
        
        # 添加ultralytics必要属性
        self.f = -1       # 输入来源层索引，-1表示来自上一层
        self.i = 0        # 模块在网络中的索引位置
        self.type = 'OpenVocabDetect'  # 模块类型名称
        
        # 1. 边界框回归分支：负责预测位置和置信度
        self.reg_conv = nn.ModuleList(
            nn.Conv2d(x, self.na * self.no, 1) for x in ch
        )
        
        # 2. 语义特征提取分支：提取视觉特征
        # 使用两个3x3卷积后接1x1卷积，提取更丰富的特征
        self.semantic_extract = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, x, 3, padding=1),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, x, 3, padding=1),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, semantic_dim, 1)
            ) for x in ch
        )
        
        # 3. 语义投影MLP：将视觉特征映射到与CLIP兼容的语义空间
        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim * 2),
            nn.LayerNorm(semantic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )
        
        # 初始化
        self._initialize_biases()
        
        # 设置锚点 - 确保先转为numpy然后再转为tensor，避免引用原始对象
        anchors_np = np.array(anchors, dtype=np.float32)
        a = torch.tensor(anchors_np).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.stride = None

    def _initialize_biases(self):
        """初始化检测头偏置，提高训练稳定性"""
        # 初始化边界框回归分支
        for conv in self.reg_conv:
            b = conv.bias.view(self.na, -1)
            # 将置信度初始化为更小的值
            b.data[:, 4] = b.data[:, 4] - 4.5
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            
        # 初始化语义特征提取分支
        for m in self.semantic_extract.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 初始化语义投影MLP - 使用正交初始化
        for m in self.semantic_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.9)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播：处理特征图列表
        
        Args:
            x: 从backbone+neck获取的特征图列表
               [P3/8, P4/16, P5/32] 对应dam.yaml
        
        Returns:
            训练模式下：[原始检测输出, 语义特征]
            推理模式下：[解码后的检测结果, 语义特征]
        """
        # 检查输入x的类型
        if isinstance(x, torch.Tensor):
            x = [x]  # 将单个张量包装成列表
            
        # 完善的特征图数量检查与调整
        if isinstance(x, list) and (len(x) != self.nl):
            x = self._adjust_feature_maps(x)
        
        # 确保输入特征通道数与卷积层匹配
        x = self._adjust_feature_channels(x)
        
        z = []  # 回归输出（边界框+置信度）
        semantic_feats = []  # 语义特征
        
        for i in range(self.nl):
            # 1. 边界框回归分支
            reg_output = self.reg_conv[i](x[i])
            bs, _, ny, nx = reg_output.shape
            reg_output = reg_output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 2. 语义特征提取
            sem_feat = self.semantic_extract[i](x[i])
            
            # 处理语义特征: [B,C,H,W] -> [B,H*W,C]
            B, C, H, W = sem_feat.shape
            sem_feat = sem_feat.view(B, C, H*W).permute(0, 2, 1)
            semantic_feats.append(sem_feat)
            
            # 训练阶段直接返回特征，测试阶段解码为实际边界框
            if not self.training:
                # 不存在预定义的网格时创建
                if self.grid[i].shape[2:4] != reg_output.shape[2:4] or self.stride is None:
                    self._make_grid(nx, ny, i)
                    
                # 应用sigmoid和网格解码
                y = reg_output.sigmoid()
                
                # 使用inplace操作解码边界框
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                
                z.append(y.view(bs, -1, self.no))
            else:
                z.append(reg_output)
        
        # 合并所有尺度的语义特征
        combined_semantic = torch.cat(semantic_feats, dim=1)  # [B, sum(H*W), C]
        
        # 应用语义投影MLP，映射到与CLIP兼容的语义空间
        projected_semantic = self.semantic_projection(combined_semantic)
        
        # 标准化特征向量，便于与文本特征进行余弦相似度计算
        normalized_semantic = F.normalize(projected_semantic, p=2, dim=-1)
        
        if self.training:
            # 训练模式下返回原始特征和语义特征
            return [torch.cat(z, 1), normalized_semantic]
        else:
            # 测试模式下返回解码后的检测结果和语义特征
            return torch.cat(z, 1), normalized_semantic

    def _adjust_feature_maps(self, x):
        """调整特征图数量，使其与模型期望的数量匹配"""
        print(f"特征图列表长度 = {len(x)}, 需要 = {self.nl}")
        if len(x) > 0:
            for i, feat in enumerate(x):
                print(f"  特征图 {i}: shape = {feat.shape}")
                
        # 调整特征图数量
        new_x = []
        if len(x) == 1 and self.nl > 1:
            # 单特征图 -> 多特征图
            orig_x = x[0]
            for i in range(self.nl):
                if i == 0:
                    new_x.append(orig_x)
                else:
                    # 通过最大池化进行下采样
                    new_x.append(torch.nn.functional.max_pool2d(new_x[-1], kernel_size=2, stride=2))
        elif len(x) > self.nl:
            # 特征图太多，只取前几个
            new_x = x[:self.nl]
        elif len(x) < self.nl:
            # 特征图太少，但不是只有一个
            # 复制最后一个特征图，并重新调整大小
            for i in range(len(x)):
                new_x.append(x[i])
            last_feat = x[-1] if len(x) > 0 else torch.zeros(1, 64, 80, 80, device=x[0].device)
            for i in range(len(x), self.nl):
                # 下采样并添加新特征图
                last_feat = torch.nn.functional.max_pool2d(last_feat, kernel_size=2, stride=2)
                new_x.append(last_feat)
        else:
            return x  # 数量已经正确
            
        print(f"特征图列表长度已调整为 {len(new_x)}")
        for i, feat in enumerate(new_x):
            print(f"  调整后特征图 {i}: shape = {feat.shape}")
            
        return new_x
    
    def _adjust_feature_channels(self, x):
        """确保输入特征通道数与卷积层匹配"""
        for i, feat in enumerate(x):
            if i < len(self.reg_conv):
                expected_channels = self.reg_conv[i].in_channels
                if feat.shape[1] != expected_channels:
                    print(f"特征图 {i} 通道数不匹配: 实际 = {feat.shape[1]}, 期望 = {expected_channels}")
                    # 动态调整通道数
                    if feat.shape[1] < expected_channels:
                        # 通道数太少，需要扩展
                        padding = torch.zeros(
                            feat.shape[0], expected_channels - feat.shape[1], 
                            feat.shape[2], feat.shape[3], device=feat.device
                        )
                        x[i] = torch.cat([feat, padding], dim=1)
                    else:
                        # 通道数太多，需要截断
                        x[i] = feat[:, :expected_channels, ...]
                    print(f"  已调整为: {x[i].shape}")
        return x

    def _make_grid(self, nx=20, ny=20, i=0):
        """创建网格和锚点网格"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        
        # 计算特征图尺度
        if self.stride is None:
            self.stride = torch.tensor([8, 16, 32], device=d)
                
        # 建立网格坐标
        yv, xv = torch.meshgrid([torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)], indexing='ij')
        self.grid[i] = torch.stack((xv, yv), 2).expand(shape) - 0.5


class SemanticHead(nn.Module):
    """
    独立的语义投影头，用于优化语义映射
    这个模块可以分离出来进行单独训练
    """
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化
        self._initialize_params()
        
    def _initialize_params(self):
        """正交初始化参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.9)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        投影视觉特征到语义空间
        
        Args:
            x: 视觉特征，形状为 [B, N, D]
               B是批次大小，N是特征数量，D是特征维度
               
        Returns:
            投影后的特征，形状为 [B, N, output_dim]
        """
        projected = self.projection(x)
        # 标准化特征向量以便于余弦相似度计算
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized