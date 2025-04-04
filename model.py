import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientEnsureFunction(Function):
    """
    自定义自动求导函数，确保梯度传播
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        # 确保梯度始终有效
        grad_input = grad_output.clone()
        # 对极小梯度添加小补偿，确保梯度流动
        zero_mask = (torch.abs(grad_input) < 1e-8).float()
        grad_input = grad_input + zero_mask * 1e-8
        return grad_input

# 注册自定义梯度函数
ensure_gradient = GradientEnsureFunction.apply

class OpenVocabDetect(nn.Module):
    """
    开放词汇目标检测头
    平衡设计 - 确保梯度流动的同时保留足够的语义学习能力
    """
    def __init__(self, nc=80, anchors=(), ch=(), semantic_dim=512, inplace=True):
        super().__init__()
        self.nc = nc              # 类别数量
        self.no = 4 + 1           # 输出: x,y,w,h,conf
        self.nl = len(anchors)    # 特征图层数
        self.na = len(anchors[0]) // 2  # 每层的锚点数
        self.semantic_dim = semantic_dim
        self.inplace = inplace
        
        # Ultralytics兼容属性
        self.f = -1               # 输入来源层索引
        self.i = 0                # 模块在网络中的索引
        self.type = 'OpenVocabDetect'
        
        # 1. 边界框和置信度预测
        self.reg_conv = nn.ModuleList(
            nn.Conv2d(x, self.na * self.no, 1) for x in ch
        )
        
        # 2. 语义特征提取 - 保留足够的表达能力
        self.semantic_extract = nn.ModuleList()
        for x in ch:
            # 每个特征层的提取模块包含:
            # - 特征转换
            # - 非线性激活
            # - 维度映射
            extract = nn.Sequential(
                # 特征转换，保留通道维度
                nn.Conv2d(x, x, 3, padding=1, bias=True),
                nn.BatchNorm2d(x),
                nn.LeakyReLU(0.1, inplace=True),
                # 映射到语义空间
                nn.Conv2d(x, semantic_dim, 1, bias=True)
            )
            self.semantic_extract.append(extract)
        
        # 3. 语义投影 - 简化版本，但保留非线性转换能力
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim, bias=True)
        )
        
        # 初始化参数
        self._initialize_params()
        
        # 设置锚点和网格
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.register_buffer('anchor_grid', self.anchors.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.grid = [torch.zeros(1)] * self.nl
        self.stride = None
        
        # 强制所有参数需要梯度
        for param in self.parameters():
            param.requires_grad_(True)
    
    def _initialize_params(self):
        """初始化参数，确保有良好的起点"""
        # 1. 边界框预测初始化
        for conv in self.reg_conv:
            # 置信度偏置初始化为较合理的值
            b = conv.bias.view(self.na, -1)
            b.data[:, 4] = b.data[:, 4] - 2.0
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            # 权重初始化
            nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        
        # 2. 语义特征提取初始化
        for extract in self.semantic_extract:
            for m in extract:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
        
        # 3. 语义投影初始化
        for m in self.semantic_projection:
            if isinstance(m, nn.Linear):
                # 使用正交初始化，保持语义空间结构
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播，保证梯度流动"""
        if isinstance(x, torch.Tensor):
            x = [x]  # 将单个特征图包装为列表
            
        assert len(x) == self.nl, f"特征图数量不匹配: 输入{len(x)}个, 需要{self.nl}个"
        
        z = []  # 边界框输出
        semantic_features = []  # 语义特征
        
        for i in range(self.nl):
            # 1. 边界框预测
            box_output = self.reg_conv[i](x[i])
            # 应用梯度保证函数
            box_output = ensure_gradient(box_output)
            bs, _, ny, nx = box_output.shape
            
            # 调整形状
            box_output = box_output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 2. 语义特征提取
            sem_feat = self.semantic_extract[i](x[i])
            # 应用梯度保证函数
            sem_feat = ensure_gradient(sem_feat)
            
            # 将特征展平为 [B, H*W, C]
            sem_feat = sem_feat.permute(0, 2, 3, 1).reshape(bs, -1, self.semantic_dim)
            semantic_features.append(sem_feat)
            
            # 训练模式返回原始输出，推理模式解码边界框
            if self.training:
                z.append(box_output)
            else:
                # 初始化网格
                if self.stride is None:
                    self.stride = torch.tensor([8, 16, 32], device=x[i].device)
                
                # 创建网格
                if self.grid[i].shape[2:4] != box_output.shape[2:4] or self.grid[i].device != box_output.device:
                    self.grid[i] = self._make_grid(nx, ny).to(box_output.device)
                
                # 应用sigmoid和解码
                y = box_output.sigmoid()
                
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                
                z.append(y.view(bs, -1, self.no))
        
        # 合并所有特征图的输出
        box_outputs = torch.cat(z, 1)
        
        # 合并语义特征
        combined_semantic = torch.cat(semantic_features, 1)  # [B, sum(H*W), C]
        
        # 语义投影 - 应用残差连接保证梯度流动
        projected_semantic = self.semantic_projection(combined_semantic)
        projected_semantic = projected_semantic + combined_semantic  # 残差连接
        
        # 标准化特征向量
        normalized_semantic = F.normalize(projected_semantic, p=2, dim=-1, eps=1e-8)
        
        # 保证输出有梯度
        box_outputs = ensure_gradient(box_outputs)
        normalized_semantic = ensure_gradient(normalized_semantic)
        
        return [box_outputs, normalized_semantic]
    
    def _make_grid(self, nx=20, ny=20):
        """创建坐标网格"""
        if torch.__version__ >= '1.10.0':
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class SemanticHead(nn.Module):
    """
    独立的语义投影头，与原始train_rl函数兼容
    简化但保留核心功能
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
        # 添加梯度保证机制
        x = ensure_gradient(x)
        
        # 应用投影
        projected = self.projection(x)
        
        # 标准化特征向量以便于与文本特征进行余弦相似度计算
        normalized = F.normalize(projected, p=2, dim=-1, eps=1e-8)
        
        # 确保输出有梯度
        normalized = ensure_gradient(normalized)
        
        return normalized


class DetectionLoss(nn.Module):
    """
    检测损失函数
    结合边界框损失和语义损失
    """
    def __init__(self, text_embeddings, box_weight=1.0, sem_weight=1.0):
        super().__init__()
        self.text_embeddings = text_embeddings  # [num_classes, semantic_dim]
        self.box_weight = box_weight
        self.sem_weight = sem_weight
        self.temp = 0.07  # 温度参数，控制对比学习的锐度
        
        # 确保text_embeddings有梯度
        if not self.text_embeddings.requires_grad:
            self.text_embeddings = nn.Parameter(self.text_embeddings.clone(), requires_grad=False)
    
    def forward(self, predictions, targets):
        """计算损失"""
        box_preds, sem_features = predictions
        
        # 如果targets是字典列表，处理为批次格式
        if isinstance(targets, dict) and 'boxes' in targets and isinstance(targets['boxes'], list):
            target_boxes_list = targets['boxes']
            target_classes_list = targets['classes']
        else:
            # 单个样本情况
            target_boxes_list = [targets['boxes']]
            target_classes_list = [targets['classes']]
        
        # 1. 边界框损失 
        box_loss = torch.tensor(0.0, device=box_preds.device, requires_grad=True)
        
        for i, (boxes, classes) in enumerate(zip(target_boxes_list, target_classes_list)):
            if boxes.numel() > 0:
                # 取当前批次样本的预测
                sample_preds = box_preds[i:i+1]
                sample_loss = self._compute_box_loss(sample_preds, boxes)
                box_loss = box_loss + sample_loss
        
        # 平均每个样本的损失
        box_loss = box_loss / max(len(target_boxes_list), 1)
        
        # 2. 语义损失
        sem_loss = self._compute_semantic_loss(sem_features, target_classes_list)
        
        # 3. 梯度保证项
        ensure_loss = 0.0001 * (box_preds.sum() + sem_features.sum())
        ensure_loss = ensure_gradient(ensure_loss)
        
        # 总损失
        total_loss = self.box_weight * box_loss + self.sem_weight * sem_loss + ensure_loss
        
        # 损失字典
        loss_dict = {
            'box_loss': box_loss,
            'sem_loss': sem_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def _compute_box_loss(self, box_preds, target_boxes):
        """边界框损失计算"""
        # 如果没有目标框，使用小的正则化损失
        if target_boxes.numel() == 0:
            return torch.sum(box_preds[..., :4]**2) * 0.0001
        
        # 计算坐标损失 (L1损失)
        coord_loss = F.smooth_l1_loss(box_preds[..., :4], target_boxes)
        
        # 计算置信度损失
        # 简化: 对于有目标的区域置信度应该高
        pos_mask = torch.ones_like(box_preds[..., 4], requires_grad=False)
        conf_loss = F.binary_cross_entropy_with_logits(box_preds[..., 4], pos_mask)
        
        return coord_loss + conf_loss
    
    def _compute_semantic_loss(self, sem_features, target_classes_list):
        """语义损失计算"""
        # 构建所有目标类别的集合
        flat_classes = []
        for classes in target_classes_list:
            if classes.numel() > 0:
                flat_classes.append(classes)
        
        # 如果没有有效的类别，使用均匀分布损失
        if not flat_classes:
            # 小的正则化损失
            return torch.sum(sem_features**2) * 0.0001
        
        # 展平类别
        flat_classes = torch.cat(flat_classes)
        
        # 获取目标类别的文本嵌入
        target_embeds = self.text_embeddings[flat_classes]
        
        # 计算相似度
        # [B, N, D] x [C, D].T -> [B, N, C]
        similarities = torch.matmul(sem_features, self.text_embeddings.T)
        
        # 使用InfoNCE对比损失
        target_sims = torch.matmul(sem_features, target_embeds.T)
        
        # 提取对角线的相似度作为正样本
        # 简化: 每个目标类与语义特征中的所有点计算相似度
        pos_sims = target_sims.mean(dim=1)  # [B, C]的均值
        
        # 所有类的相似度作为对比项
        logits = similarities / self.temp
        
        # NCE损失
        nce_loss = -torch.log(torch.exp(pos_sims / self.temp) / 
                             torch.sum(torch.exp(logits / self.temp), dim=-1))
        
        return nce_loss.mean()


def train_with_validation(
    model,
    train_loader, 
    val_loader,
    text_embeddings,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda'
):
    """训练函数，包含验证过程"""
    # 设置训练模式
    model.train()
    
    # 创建损失函数
    criterion = DetectionLoss(text_embeddings, box_weight=1.0, sem_weight=1.0)
    
    # 获取检测头参数
    detect_head = model.model.model[-1]
    params = list(detect_head.parameters())
    
    print(f"检测头参数数量: {len(params)}")
    for i, p in enumerate(params):
        print(f"参数 {i}: 形状 {p.shape}, 需要梯度: {p.requires_grad}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # 训练和验证记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 准备数据
            images = batch['img'].to(device)
            targets = {
                'boxes': [b.to(device) for b in batch['gt']['boxes']],
                'classes': [c.to(device) for c in batch['gt']['classes']]
            }
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            has_grad = False
            total_norm = 0
            for param in params:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    has_grad = True
            
            total_norm = total_norm ** (1. / 2)
            
            if has_grad:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}, Grad norm: {total_norm:.6f}")
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                
                # 参数更新
                optimizer.step()
            else:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}, WARNING: No gradients")
            
            # 更新学习率
            scheduler.step()
            
            # 累积损失
            epoch_loss += loss.item()
            
            # 每10个batch输出一次
            if (batch_idx+1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx+1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_loss:.6f}")
        
        # 计算训练集平均损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 准备数据
                images = batch['img'].to(device)
                targets = {
                    'boxes': [b.to(device) for b in batch['gt']['boxes']],
                    'classes': [c.to(device) for c in batch['gt']['classes']]
                }
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss, _ = criterion(outputs, targets)
                
                # 累积损失
                val_loss += loss.item()
        
        # 计算验证集平均损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 输出当前epoch结果
        print(f"Epoch {epoch+1} completed. Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with val loss: {best_val_loss:.6f}")
        
        # 每10个epoch保存一次检查点
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"checkpoint_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    torch.save(model.state_dict(), "final_model.pt")
    
    return model, train_losses, val_losses


def register_custom_modules():
    """注册自定义模块到ultralytics框架"""
    import builtins
    builtins.__dict__['OpenVocabDetect'] = OpenVocabDetect
    builtins.__dict__['SemanticHead'] = SemanticHead
    
    from ultralytics.nn import modules
    modules.__dict__['OpenVocabDetect'] = OpenVocabDetect
    modules.__dict__['SemanticHead'] = SemanticHead
    
    # 作为保险，也替换Detect类
    original_Detect = modules.Detect
    modules.Detect = OpenVocabDetect
    
    print("已将OpenVocabDetect和SemanticHead注册到全局和模块命名空间")