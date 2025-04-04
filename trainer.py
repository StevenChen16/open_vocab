import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import logging
import clip
from tqdm import tqdm
from ultralytics.engine.model import Model
from model import OpenVocabDetect, SemanticHead  # 引入自定义检测头
from torchvision.ops import box_iou, nms  # 导入边界框操作函数

# 配置日志系统
def setup_logger(level=logging.INFO):
    """配置日志系统
    
    Args:
        level: 日志级别，默认为INFO，可选DEBUG/INFO/WARNING/ERROR/CRITICAL
    
    Returns:
        logger: 日志对象
    """
    logger = logging.getLogger("OpenVocabDetection")
    logger.setLevel(level)
    
    # 清除现有的handler，避免重复日志
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 定义格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志对象
logger = setup_logger()

# 注册自定义模块到ultralytics
def register_custom_modules():
    """注册自定义模块到ultralytics框架"""
    # 将自定义类直接注册到globals()中
    import builtins
    builtins.__dict__['OpenVocabDetect'] = OpenVocabDetect
    
    # 同时也修改ultralytics.nn.modules模块的全局命名空间
    from ultralytics.nn import modules
    modules.__dict__['OpenVocabDetect'] = OpenVocabDetect
    
    # 作为保险，也替换Detect类
    original_Detect = modules.Detect
    modules.Detect = OpenVocabDetect
    
    logger.info(f"已将OpenVocabDetect注册到全局命名空间和模块命名空间")


# 修改YAML配置文件以使用自定义检测头
def modify_yaml_config(yaml_path, new_yaml_path, semantic_dim=512):
    """
    修改YAML配置，将最后的Detect层替换为OpenVocabDetect
    
    Args:
        yaml_path: 原始YAML配置文件路径
        new_yaml_path: 新的YAML配置文件路径
        semantic_dim: 语义向量维度
    """
    # 直接读取YAML文件为文本
    with open(yaml_path, 'r') as f:
        yaml_text = f.read()
    
    # 使用文本替换，这样更可靠，不会改变YAML结构
    yaml_text = yaml_text.replace("Detect, [nc]", f"OpenVocabDetect, [{{nc: 80, semantic_dim: {semantic_dim}}}]")
    
    # 保存修改后的文本
    with open(new_yaml_path, 'w') as f:
        f.write(yaml_text)
    
    logger.info(f"已创建修改后的配置文件: {new_yaml_path}")
    return new_yaml_path


# 创建演示数据集类
class DemoDataset:
    """简单的演示数据集"""
    def __init__(self, size=10, device='cpu'):
        self.size = size
        self.device = device
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 创建随机图像和标注 - 注意我们在CPU上创建，让环境自己决定转移到哪个设备
        image = torch.rand(3, 640, 640)
        boxes = torch.tensor([[100, 100, 200, 200]])
        classes = torch.tensor([0])
        
        return {
            'img': image,
            'gt': {
                'boxes': boxes,
                'classes': classes
            },
            'pred_boxes': boxes  # 添加预测框，供环境使用
        }


# 获取类别文本特征
def get_text_embeddings(class_names, semantic_dim=512, device='cuda'):
    """
    获取类别的文本特征向量
    
    Args:
        class_names: 类别名称列表或字典
        semantic_dim: 语义向量维度
        device: 使用的设备
        
    Returns:
        text_embeddings: 文本特征向量，形状为[num_classes, semantic_dim]
    """
    logger.info("准备类别文本特征向量...")
    
    # 尝试使用CLIP模型获取文本特征
    try:
        # 加载CLIP模型
        clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # 处理类别名称
        if isinstance(class_names, dict):
            texts = [f"a photo of a {name}" for name in class_names.values()]
        else:
            texts = [f"a photo of a {name}" for name in class_names]
            
        # 编码文本
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)
            text_features = clip_model.encode_text(text_tokens)
            # 确保转换为float32并归一化
            text_features = text_features.float()  # 明确转换为float32
            text_embeddings = F.normalize(text_features, p=2, dim=1)
            
        logger.info(f"成功使用CLIP生成文本特征，形状为{text_embeddings.shape}，数据类型为{text_embeddings.dtype}")
        return text_embeddings
        
    except Exception as e:
        logger.warning(f"无法使用CLIP：{e}，将使用随机初始化的特征")
        
        # 如果无法使用CLIP，则生成随机特征
        if isinstance(class_names, dict):
            num_classes = len(class_names)
        else:
            num_classes = len(class_names)
            
        text_embeddings = F.normalize(torch.randn(num_classes, semantic_dim, device=device), p=2, dim=1)
        logger.info(f"生成随机文本特征，形状为{text_embeddings.shape}")
        return text_embeddings


# 强化学习环境
class DetectionRLEnv:
    """
    目标检测的强化学习环境，支持完整的三部分奖励机制
    """
    def __init__(
        self,
        model,
        text_embeddings,  # CLIP文本嵌入，形状 [num_classes, semantic_dim]
        dataset,          # 训练数据集
        conf_threshold=0.25,
        nms_threshold=0.45,
        semantic_dim=512,
        reward_weights={'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1},
        reward_scale=10.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        ch_list=None      # 检测头输入通道数列表
    ):
        self.device = device
        # 确保模型在正确的设备上
        self.model = model.to(device)
        # 确保文本嵌入在正确的设备上
        self.text_embeddings = text_embeddings.to(device)
        self.dataset = dataset
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.semantic_dim = semantic_dim
        self.current_image_idx = 0
        self.current_batch = None
        self.current_preds = None
        self.current_gt = None
        
        # 奖励函数权重
        self.reward_weights = reward_weights
        self.reward_scale = reward_scale
        
        # 存储通道数信息
        self.ch_list = ch_list if ch_list is not None else [64, 128, 256]
        logger.info(f"环境初始化使用通道数: {self.ch_list}")
        
        # 存储语义特征历史，用于探索奖励计算
        self.semantic_history = []
        self.max_history_size = 1000
        
        logger.info(f"RL环境初始化完成，设备: {device}, 文本嵌入形状: {self.text_embeddings.shape}")
    
    def reset(self):
        """重置环境，返回新图像的状态"""
        # 获取下一张图像和标注
        self.current_image_idx = np.random.randint(0, len(self.dataset))
        batch = self.dataset[self.current_image_idx]
        
        # 提取图像和标注，确保所有数据都在正确的设备上
        if isinstance(batch, dict) and 'img' in batch:
            images = batch['img'].to(self.device).float() / 255.0  # 标准化
        else:
            # 尝试从batch中提取数据，格式可能不同
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0].to(self.device).float() / 255.0
            else:
                # 创建虚拟数据
                logger.debug(f"无法解析数据集条目，创建虚拟图像，批次类型: {type(batch)}")
                images = torch.randn(1, 3, 640, 640, device=self.device)
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        # 将batch内容都移动到正确设备
        if isinstance(batch, dict):
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    processed_batch[k] = {k2: v2.to(self.device) if isinstance(v2, torch.Tensor) else v2 
                                         for k2, v2 in v.items()}
                else:
                    processed_batch[k] = v
            self.current_batch = processed_batch
        else:
            self.current_batch = batch
        
        # 提取GT标注，确保在正确设备上
        if isinstance(batch, dict) and 'gt' in batch:
            gt = batch['gt']
            self.current_gt = {
                'boxes': gt['boxes'].to(self.device) if isinstance(gt['boxes'], torch.Tensor) else gt['boxes'],
                'classes': gt['classes'].to(self.device) if isinstance(gt['classes'], torch.Tensor) else gt['classes'],
            }
        else:
            # 创建虚拟GT标注
            self.current_gt = {
                'boxes': torch.tensor([[100, 100, 200, 200]], device=self.device),
                'classes': torch.tensor([0], device=self.device),
            }
            
        # 直接创建虚拟特征而不尝试通过模型前向传播
        logger.debug("创建虚拟特征")
            
        # 使用初始化时确定的通道数，确保与检测头的卷积层匹配
        features = [
            torch.randn(1, self.ch_list[0], 80, 80, device=self.device),
            torch.randn(1, self.ch_list[1], 40, 40, device=self.device),
            torch.randn(1, self.ch_list[2], 20, 20, device=self.device)
        ]
            
        self.current_preds = features
        
        return {
            'features': features,
            'image': images
        }
    
    def step(self, action, eval_mode=False):
        """
        执行动作并返回新状态、奖励和完成标志
        
        Args:
            action: 语义投影参数，用于将视觉特征映射到语义空间
                   可以是单个矩阵或多个矩阵的列表
            eval_mode: 如果为True，表示这是计算损失函数时的计算图构建，
                       而不是真实环境中的步骤执行
        
        Returns:
            next_state: 下一个状态
            reward: 本步骤的奖励
            done: 是否完成
            info: 附加信息
        """
        # 解构当前状态
        features = self.current_preds
        batch = self.current_batch
        gt_boxes = self.current_gt['boxes']
        gt_classes = self.current_gt['classes']
        
        # 应用动作（语义投影）
        # 检查action是否是单个矩阵，如果是，应该处理所有特征级别
        if isinstance(action, torch.Tensor) and action.dim() == 2:
            if action.shape[0] != self.ch_list[0] or action.shape[1] != self.semantic_dim:
                logger.debug(f"动作形状与第一个特征级别不匹配，需要独立处理每个特征级别")
                # 分别为每个特征级别创建适合的投影矩阵
                projection_matrices = []
                for ch in self.ch_list:
                    # 使用随机初始化或从原始action切片获取
                    if action.shape[0] >= ch:
                        # 从大矩阵切出所需大小
                        proj_matrix = action[:ch, :self.semantic_dim]
                    else:
                        # 创建新的随机矩阵
                        proj_matrix = torch.randn(ch, self.semantic_dim, device=self.device)
                    projection_matrices.append(proj_matrix)
                action = projection_matrices
            
        semantic_features = self._apply_semantic_projection(features, action)
        
        # 确保数据类型匹配
        semantic_features = semantic_features.float()  # 转为float32
        self.text_embeddings = self.text_embeddings.float()  # 确保也是float32
        
        # 计算预测框与文本嵌入的相似度
        # 假设semantic_features的形状是 [batch_size, num_boxes, semantic_dim]
        similarities = torch.matmul(semantic_features, self.text_embeddings.T)
        
        # 获取最大相似度和对应的类别
        max_sim, pred_classes = similarities.max(dim=-1)
        
        # 打印形状信息，帮助调试
        logger.debug(f"语义特征形状: {semantic_features.shape}")
        logger.debug(f"相似度矩阵形状: {similarities.shape}")
        logger.debug(f"最大相似度形状: {max_sim.shape}")
        logger.debug(f"预测类别形状: {pred_classes.shape}")
        
        # 解决维度不匹配问题
        # 方法：生成与特征点数量匹配的预测框
        num_features = max_sim.shape[1]  # 特征点数量
        
        # 检查batch是否已包含足够的预测框
        if 'pred_boxes' not in batch or batch['pred_boxes'].shape[1] != 4 or batch['pred_boxes'].shape[0] != num_features:
            # 生成预测框
            pred_boxes = self._generate_prediction_boxes(num_features)
        else:
            # 使用batch中已有的预测框
            pred_boxes = batch['pred_boxes']
        
        # 应用置信度过滤
        conf_mask = max_sim > self.conf_threshold
        # 确保掩码和预测框形状匹配
        if conf_mask.dim() > 1 and conf_mask.shape[0] == 1:
            conf_mask = conf_mask.squeeze(0)  # 移除批次维度
        
        # 应用掩码过滤
        filtered_boxes = pred_boxes[conf_mask]
        filtered_classes = pred_classes.reshape(-1)[conf_mask]
        filtered_scores = max_sim.reshape(-1)[conf_mask]
        
        # 应用NMS
        keep_indices = self._apply_nms(filtered_boxes, filtered_scores)
        final_boxes = filtered_boxes[keep_indices]
        final_classes = filtered_classes[keep_indices]
        final_scores = filtered_scores[keep_indices]
        
        # 对于训练模式，将当前语义特征添加到历史记录中，用于计算探索奖励
        if not eval_mode:
            self._update_semantic_history(semantic_features)
        
        # 计算奖励
        reward, reward_info = self._calculate_reward(
            final_boxes, 
            final_classes, 
            final_scores,
            self.current_gt['boxes'], 
            self.current_gt['classes'], 
            semantic_features
        )
        
        # 更新状态
        info = {
            'pred_boxes': final_boxes,
            'pred_classes': final_classes,
            'pred_scores': final_scores,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'reward_info': reward_info  # 添加详细奖励信息以便分析
        }
        
        # 如果是评估模式，仅返回奖励而不重置环境
        if eval_mode:
            # 仅返回奖励，不重置环境，因为这只是为了计算梯度
            # 需要确保奖励与参数有计算图连接关系
            return None, reward, False, info
        else:
            # 正常模式下重置环境
            next_state = self.reset()  # 简单起见，每步后重置环境
            done = True  # 每张图片只处理一次
            
            return next_state, reward, done, info
    
    def _generate_prediction_boxes(self, num_features):
        """生成与特征点数量匹配的预测框"""
        logger.debug(f"生成与特征点匹配的预测框，数量: {num_features}")
        # 创建虚拟预测框，每个特征点对应一个框
        # 简单方法：为每个特征点生成一个与图像尺寸相关的框
        # 假设原始图像尺寸是640x640
        img_size = 640
        
        # 计算网格坐标，每个特征点对应原始图像中的一个区域
        # 大特征图(80x80)对应小框，小特征图(20x20)对应大框
        boxes = []
        
        # 处理第一个特征图 80x80
        grid_size = img_size / 80
        for h in range(80):
            for w in range(80):
                cx, cy = w * grid_size + grid_size/2, h * grid_size + grid_size/2
                box_size = grid_size * 0.8  # 框稍小于网格
                x1, y1 = cx - box_size/2, cy - box_size/2
                x2, y2 = cx + box_size/2, cy + box_size/2
                boxes.append([x1, y1, x2, y2])
                
        # 处理第二个特征图 40x40
        grid_size = img_size / 40
        for h in range(40):
            for w in range(40):
                cx, cy = w * grid_size + grid_size/2, h * grid_size + grid_size/2
                box_size = grid_size * 0.8
                x1, y1 = cx - box_size/2, cy - box_size/2
                x2, y2 = cx + box_size/2, cy + box_size/2
                boxes.append([x1, y1, x2, y2])
                
        # 处理第三个特征图 20x20
        grid_size = img_size / 20
        for h in range(20):
            for w in range(20):
                cx, cy = w * grid_size + grid_size/2, h * grid_size + grid_size/2
                box_size = grid_size * 0.8
                x1, y1 = cx - box_size/2, cy - box_size/2
                x2, y2 = cx + box_size/2, cy + box_size/2
                boxes.append([x1, y1, x2, y2])
        
        # 转换为tensor
        return torch.tensor(boxes, device=self.device)
    
    def _apply_semantic_projection(self, features, action):
        """应用语义投影，将视觉特征映射到语义空间"""
        projected_features = []
        
        # 检测是否使用模型中预定义的投影层
        if hasattr(self.model, 'projection_layers') and len(self.model.projection_layers) >= len(features):
            logger.debug(f"使用预定义的投影层进行语义投影")
            # 使用预定义的投影层
            for i, feat in enumerate(features):
                proj_layer = self.model.projection_layers[i]
                
                # 展平特征图
                # [B, C, H, W] -> [B, H*W, C]
                flattened = feat.flatten(2).transpose(1, 2)  
                # [B, H*W, C] -> [B*H*W, C]
                flattened_batch = flattened.reshape(-1, flattened.shape[-1])
                
                # 直接应用线性层
                # [B*H*W, C] -> [B*H*W, semantic_dim]
                projected_batch = proj_layer(flattened_batch)
                
                # 恢复形状
                # [B*H*W, semantic_dim] -> [B, H*W, semantic_dim]
                projected = projected_batch.reshape(flattened.shape[0], flattened.shape[1], -1)
                
                # 标准化
                projected = F.normalize(projected, p=2, dim=-1)
                projected_features.append(projected)
            
        else:  
            # 使用传入的动作来确定投影矩阵
            # 检查action的类型
            if isinstance(action, torch.Tensor) and action.dim() == 2:
                # 如果action是单个矩阵，需要分解为多个矩阵
                logger.debug(f"使用传入的动作投影矩阵，并分配给每个特征级别")
                # 确保目标维度是semantic_dim
                projection_matrices = []
                for i, feat in enumerate(features):
                    ch = feat.shape[1]  # 获取当前特征的通道数
                    # 为每个特征级别创建一个独立的投影矩阵
                    if action.shape[0] >= ch:
                        proj_matrix = action[:ch, :]
                    else:
                        logger.debug(f"  动作维度不足，为特征 {i} (形状: {feat.shape}) 创建随机投影矩阵")
                        proj_matrix = torch.randn(ch, self.semantic_dim, device=self.device)
                    projection_matrices.append(proj_matrix)
            else:
                # 如果action已经是一个列表或tensor序列，直接使用
                logger.debug(f"使用提供的投影矩阵列表，数量: {len(action) if isinstance(action, list) else 1}")
                projection_matrices = action if isinstance(action, list) else [action] * len(features)
            
            # 确保投影矩阵数量足够
            if len(projection_matrices) < len(features):
                logger.warning(f"投影矩阵数量不足({len(projection_matrices)})，自动创建缺失的矩阵以匹配特征数量({len(features)})")
                for i in range(len(projection_matrices), len(features)):
                    ch = features[i].shape[1]
                    projection_matrices.append(torch.randn(ch, self.semantic_dim, device=self.device))
            
            # 逐个特征级别应用投影
            for i, feat in enumerate(features):
                
                # 获取当前特征的投影矩阵
                proj_matrix = projection_matrices[i]
                
                # 确保投影矩阵形状与特征匹配
                ch = feat.shape[1]
                if proj_matrix.shape[0] != ch or proj_matrix.shape[1] != self.semantic_dim:
                    logger.debug(f"  调整投影矩阵形状: {proj_matrix.shape} -> [{ch}, {self.semantic_dim}]")
                    proj_matrix = torch.randn(ch, self.semantic_dim, device=self.device)
                    projection_matrices[i] = proj_matrix
                
                # 应用投影 - 首先展平空间维度，然后转置使通道在最后
                # [B, C, H, W] -> [B, H*W, C]
                flattened = feat.flatten(2).transpose(1, 2)
                
                # 使用线性变换进行投影
                # F.linear需要权重形状为[out_features, in_features]
                # 而我们的proj_matrix形状是[C, semantic_dim]，需要转置
                # [B, H*W, C] x [semantic_dim, C] -> [B, H*W, semantic_dim]
                projected = F.linear(flattened, proj_matrix.t())
                
                # 标准化特征向量
                projected = F.normalize(projected, p=2, dim=-1)
                projected_features.append(projected)
        
        # 检查有多少特征被成功投影
        if len(projected_features) == 0:
            logger.warning("警告: 没有特征被成功投影，创建默认随机特征")
            # 创建一个随机特征作为备选
            default_feature = torch.randn(1, 100, self.semantic_dim, device=self.device)
            default_feature = F.normalize(default_feature, p=2, dim=-1)
            projected_features.append(default_feature)
        
        # 合并不同尺度的特征
        semantic_features = torch.cat(projected_features, dim=1)
        logger.debug(f"投影后的语义特征形状: {semantic_features.shape}")
        return semantic_features
    
    def _update_semantic_history(self, semantic_features):
        """更新语义特征历史，用于计算探索奖励"""
        # 将批次维度分离
        if semantic_features.dim() > 2:
            # [B, N, D] -> 多个 [N, D]
            for b in range(semantic_features.shape[0]):
                feat = semantic_features[b].detach()  # 分离梯度
                # 随机选择一部分特征点以节省内存
                if feat.shape[0] > 10:
                    indices = torch.randperm(feat.shape[0])[:10]
                    feat = feat[indices]
                self.semantic_history.append(feat)
        else:
            # 已经是 [N, D]
            feat = semantic_features.detach()  # 分离梯度
            if feat.shape[0] > 10:
                indices = torch.randperm(feat.shape[0])[:10]
                feat = feat[indices]
            self.semantic_history.append(feat)
            
        # 限制历史大小
        if len(self.semantic_history) > self.max_history_size:
            self.semantic_history = self.semantic_history[-self.max_history_size:]
    
    def _apply_nms(self, boxes, scores):
        """应用非极大值抑制"""
        # 使用torchvision的NMS
        keep = nms(boxes, scores, self.nms_threshold)
        return keep
    
    def _calculate_reward(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, semantic_features):
        """
        计算奖励信号，确保梯度可以正确传递
        
        这个函数计算三部分奖励：
        1. 准确性奖励：基于IoU和类别匹配
        2. 语义相似度奖励：与目标类别的语义相似度
        3. 探索奖励：特征与历史特征的不同程度
        """
        # 初始化奖励组件字典
        reward_info = {
            'accuracy': torch.tensor(0.0, device=self.device, dtype=torch.float32),
            'semantic': torch.tensor(0.0, device=self.device, dtype=torch.float32),
            'exploration': torch.tensor(0.0, device=self.device, dtype=torch.float32)
        }
        
        # 如果没有预测，返回负面奖励
        if len(pred_classes) == 0:
            total_reward = torch.tensor(-1.0, device=self.device, dtype=torch.float32, requires_grad=True)
            # 更新奖励信息
            for k in reward_info:
                reward_info[k] = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
            return total_reward * self.reward_scale, reward_info
        
        # 确保所有输入都是张量
        if not isinstance(pred_scores, torch.Tensor):
            pred_scores = torch.tensor(pred_scores, device=self.device)
            
        # 1. 准确性奖励计算
        accuracy_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # 匹配预测框与GT框
        correct_detections = 0
        total_gt = len(gt_classes)
        
        for i, pred_class in enumerate(pred_classes):
            if pred_class in gt_classes:
                # 找到相同类别的GT框
                gt_indices = (gt_classes == pred_class).nonzero(as_tuple=True)[0]
                
                for gt_idx in gt_indices:
                    # 计算IoU
                    iou = box_iou(pred_boxes[i:i+1], gt_boxes[gt_idx:gt_idx+1])[0][0]
                    
                    if iou > 0.5:  # 正确检测
                        # 增加正确检测计数
                        correct_detections += 1
                        # 基础奖励 + IoU奖励 + 置信度奖励
                        accuracy_reward = accuracy_reward + 1.0 + iou * 0.5 + pred_scores[i] * 0.3
                        break
            else:
                # 惩罚错误类别
                accuracy_reward = accuracy_reward - 0.5 * pred_scores[i]
        
        # 惩罚过多错误检测
        false_positives = max(0, len(pred_classes) - correct_detections)
        if false_positives > 0:
            accuracy_reward = accuracy_reward - false_positives * 0.2
            
        # 惩罚漏检
        false_negatives = max(0, total_gt - correct_detections)
        if false_negatives > 0:
            accuracy_reward = accuracy_reward - false_negatives * 0.3
            
        # 将准确性奖励归一化到 [-1, 1] 范围
        accuracy_reward = torch.clamp(accuracy_reward / max(1, total_gt), min=-1.0, max=1.0)
        reward_info['accuracy'] = accuracy_reward
        
        # 2. 语义相似度奖励 - 计算检测框与对应GT类别的语义相似度
        semantic_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # 确保语义特征的形状
        if isinstance(semantic_features, list):
            # 如果是语义特征列表（多个特征图）
            logger.debug(f"处理多特征图语义相似度，数量: {len(semantic_features)}")
            for feat_idx, feat in enumerate(semantic_features):
                # 处理每个特征图
                if feat_idx < len(pred_classes):
                    pred_class = pred_classes[feat_idx]
                    if pred_class < len(self.text_embeddings):
                        gt_embedding = self.text_embeddings[pred_class]
                        # 使用张量运算替代函数调用，保证梯度流动
                        feat_norm = F.normalize(feat, p=2, dim=-1)
                        gt_norm = F.normalize(gt_embedding, p=2, dim=-1)
                        sim = torch.sum(feat_norm * gt_norm)
                        semantic_reward = semantic_reward + sim
        else:
            # 假设是单个批次的语义特征 [batch, num_features, dim]
            logger.debug(f"处理批次语义特征，形状: {semantic_features.shape}")
            
            batch_size = semantic_features.shape[0]
            num_features = min(semantic_features.shape[1], len(pred_classes))
            
            for i in range(batch_size):
                for j in range(num_features):
                    if j < len(pred_classes):
                        pred_class = pred_classes[j]
                        if pred_class < len(self.text_embeddings):
                            feat = semantic_features[i, j]
                            gt_embedding = self.text_embeddings[pred_class]
                            # 使用余弦相似度
                            sim = F.cosine_similarity(feat.unsqueeze(0), gt_embedding.unsqueeze(0))
                            semantic_reward = semantic_reward + sim
                            
        # 将语义奖励归一化到 [-1, 1] 范围
        semantic_reward = torch.clamp(semantic_reward / max(1, len(pred_classes)), min=-1.0, max=1.0)
        reward_info['semantic'] = semantic_reward
        
        # 3. 探索奖励 - 计算当前特征与历史特征的平均距离
        exploration_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        if len(self.semantic_history) > 0 and semantic_features is not None:
            # 平均历史特征
            avg_history = torch.mean(torch.cat(self.semantic_history, dim=0), dim=0)
            
            # 计算当前特征与历史平均的距离
            if semantic_features.dim() > 2:
                # [B, N, D]
                for b in range(semantic_features.shape[0]):
                    # 计算每个特征向量与平均历史的余弦距离
                    distances = 1.0 - F.cosine_similarity(
                        semantic_features[b], 
                        avg_history.unsqueeze(0).expand(semantic_features[b].shape[0], -1),
                        dim=1
                    )
                    # 鼓励探索不同区域
                    exploration_reward = exploration_reward + distances.mean()
            else:
                # 单特征 [N, D]
                distances = 1.0 - F.cosine_similarity(
                    semantic_features, 
                    avg_history.unsqueeze(0).expand(semantic_features.shape[0], -1),
                    dim=1
                )
                exploration_reward = exploration_reward + distances.mean()
                
            # 归一化到 [0, 1] 范围，然后转换到 [-1, 1]
            exploration_reward = torch.clamp(exploration_reward, min=0.0, max=1.0) * 2 - 1
        else:
            # 如果没有历史，给予适中的探索奖励
            exploration_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
        reward_info['exploration'] = exploration_reward
            
        # 组合奖励
        total_reward = (
            self.reward_weights['accuracy'] * accuracy_reward +
            self.reward_weights['semantic'] * semantic_reward +
            self.reward_weights['exploration'] * exploration_reward
        )
        
        logger.debug(f"奖励计算完成 - 准确性: {accuracy_reward.item():.3f}, 语义: {semantic_reward.item():.3f}, 探索: {exploration_reward.item():.3f}, 总计: {total_reward.item():.3f}")
        
        return total_reward * self.reward_scale, reward_info


# 经验回放缓冲区
class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样经验"""
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)


# 基于奖励的参数更新策略
def update_projection_params(projection_params, reward, lr_scale=0.01):
    """
    基于奖励信号更新投影参数
    
    Args:
        projection_params: 投影参数列表
        reward: 奖励值
        lr_scale: 学习率缩放因子
    """
    # 将奖励映射到[0, 1]范围作为学习率缩放
    reward_scale = torch.sigmoid(torch.tensor(reward / 5.0)).item()
    
    # 根据奖励调整参数
    with torch.no_grad():
        for param in projection_params:
            # 添加噪声进行探索
            if reward > 0:
                # 正奖励：小幅度优化参数
                noise_scale = 0.001 * lr_scale * reward_scale
            else:
                # 负奖励：增加探索
                noise_scale = 0.01 * lr_scale * (1 - reward_scale)
            
            # 计算参数标准差以缩放噪声
            std = param.std().item() if param.numel() > 1 else 0.1
            
            # 加噪声
            noise = torch.randn_like(param) * noise_scale * std
            param.add_(noise)


# 强化学习训练主循环
def train_rl(
    model_path,          # 预训练模型路径
    yaml_path,           # 修改后的yaml配置路径
    dataset,             # 训练数据集
    class_names,         # 类别名称
    num_episodes=1000,   # 训练回合数
    learning_rate=1e-4,  # 学习率
    gamma=0.99,          # 折扣因子
    epsilon=0.3,         # 探索率
    epsilon_decay=0.995, # 探索率衰减
    min_epsilon=0.05,    # 最小探索率
    batch_size=32,       # 批大小
    semantic_dim=512,    # 语义向量维度
    reward_weights={'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1},  # 奖励权重
    reward_scale=10.0,   # 奖励缩放因子
    save_dir='runs/rl_train', # 保存目录
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 设备
    use_reward_update=True,  # 是否使用基于奖励的参数更新
):
    """
    使用强化学习训练开放词汇检测模型
    
    Args:
        model_path: 预训练YOLO模型路径
        yaml_path: 修改后的yaml配置路径
        dataset: 训练数据集
        class_names: 类别名称列表
        num_episodes: 训练回合数
        learning_rate: 学习率
        gamma: 折扣因子
        epsilon: 探索率初始值
        epsilon_decay: 探索率衰减系数
        min_epsilon: 最小探索率
        batch_size: 经验回放批大小
        semantic_dim: 语义向量维度
        reward_weights: 奖励权重字典
        reward_scale: 奖励缩放因子
        save_dir: 保存目录
        device: 设备
        use_reward_update: 是否使用基于奖励的参数更新
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 首先注册自定义模块
    register_custom_modules()
    logger.info("已注册自定义模块OpenVocabDetect到Ultralytics框架")
    
    # 获取类别文本特征
    text_embeddings = get_text_embeddings(class_names, semantic_dim, device)
    
    # 加载预训练模型
    model = Model(model_path)
    
    # 获取原始检测头
    original_detect = model.model.model[-1]
    original_detect_type = type(original_detect).__name__
    logger.info(f"原始检测头类型: {original_detect_type}")
    
    # 如果原始检测头不是OpenVocabDetect，替换它
    if original_detect_type != 'OpenVocabDetect':
        logger.info("替换原始检测头为自定义的OpenVocabDetect...")
        
        # 提取原始检测头的属性
        original_anchors = original_detect.anchors.detach().cpu().numpy() if hasattr(original_detect, 'anchors') else np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
        original_stride = original_detect.stride if hasattr(original_detect, 'stride') else None
        
        # 检测通道数
        ch_list = []
        # 使用确定的通道映射表
        for module in model.model.model[-1].modules():
            if isinstance(module, nn.Conv2d) and module.in_channels in [64, 128, 256, 512, 1024]:
                ch_list.append(module.in_channels)
                
        # 如果没有检测到正确的通道数，使用默认值
        if not ch_list or len(ch_list) != 3:  # 需要精确的3个通道
            logger.info("使用默认通道数: [64, 128, 256]")
            ch_list = [64, 128, 256]
        else:
            # 去除重复元素，保留排序
            ch_list = sorted(list(set(ch_list)))
            # 只保留3个通道（如果超过）
            if len(ch_list) > 3:
                ch_list = ch_list[-3:]
        
        logger.info(f"检测到的通道数: {ch_list}")
        
        # 创建自定义检测头
        detection_head = OpenVocabDetect(
            nc=80,
            anchors=original_anchors,
            ch=ch_list,
            semantic_dim=semantic_dim
        )
        
        # 复制必要属性
        for attr in ['f', 'i', 'type', 'inplace', 'stride']:
            if hasattr(original_detect, attr):
                setattr(detection_head, attr, getattr(original_detect, attr))
        
        # 替换检测头
        model.model.model[-1] = detection_head
        logger.info("成功替换为OpenVocabDetect")
    
    # 冻结backbone和neck
    for m in model.model.model[:-1]:  # 除了最后一层外的所有层
        for param in m.parameters():
            param.requires_grad = False
    
    # 确保检测头是可训练的
    for param in model.model.model[-1].parameters():
        param.requires_grad = True
    
    # 确保模型在正确的设备上
    model.model = model.model.to(device)
    
    # 检测头的通道数
    ch_list = []
    last_layer = model.model.model[-1]
    if hasattr(last_layer, 'reg_conv') and isinstance(last_layer.reg_conv, nn.ModuleList):
        for conv in last_layer.reg_conv:
            if hasattr(conv, 'in_channels'):
                ch_list.append(conv.in_channels)
    
    if not ch_list:
        logger.warning("无法从检测头确定通道数，使用默认值[64, 128, 256]")
        ch_list = [64, 128, 256]
        
    logger.info(f"检测到的通道数: {ch_list}")
    
    # 创建RL环境，将通道数传入环境
    env = DetectionRLEnv(
        model=model.model,
        text_embeddings=text_embeddings,
        dataset=dataset,
        semantic_dim=semantic_dim,
        reward_weights=reward_weights,
        reward_scale=reward_scale,
        device=device,
        ch_list=ch_list  # 传入检测到的通道数
    )
    
    # 定义投影层优化器
    projection_params = []
    
    # 获取模型的最后一层（检测头）
    detection_head = model.model.model[-1]
    
    # 打印模块类型信息，帮助分析
    logger.info(f"检测头类型: {type(detection_head).__name__}")
    
    # 收集语义投影层参数
    if hasattr(detection_head, 'semantic_projection'):
        logger.info(f"发现semantic_projection层，类型: {type(detection_head.semantic_projection).__name__}")
        # 收集语义投影MLP的所有参数
        projection_params.extend(list(detection_head.semantic_projection.parameters()))
        logger.info(f"  投影层: 找到 {len(list(detection_head.semantic_projection.parameters()))} 个参数")
    else:
        logger.warning("未找到semantic_projection属性，尝试查找其他语义相关层")
        
    if hasattr(detection_head, 'semantic_extract'):
        logger.info(f"发现semantic_extract层，包含 {len(detection_head.semantic_extract)} 个提取模块")
        # 收集语义特征提取网络的所有参数
        for i, extract_module in enumerate(detection_head.semantic_extract):
            params = list(extract_module.parameters())
            projection_params.extend(params)
            logger.info(f"  特征提取模块 {i}: 找到 {len(params)} 个参数")
    
    # 如果仍然没有找到参数，搜索可能的投影参数
    if not projection_params:
        logger.warning("未能找到现有的语义投影层参数，递归搜索所有可能的语义相关参数")
        for name, module in detection_head.named_modules():
            if isinstance(module, nn.Linear) or \
               (isinstance(module, nn.Conv2d) and module.out_channels == semantic_dim):
                params = list(module.parameters())
                projection_params.extend(params)
                logger.info(f"  找到可能的语义层 {name}: 参数数量 {len(params)}")
    
    # 如果还是没有找到，创建一个独立的语义投影头
    if not projection_params:
        logger.warning("警告: 未能找到任何语义投影参数，创建独立的语义投影头")
        
        semantic_head = SemanticHead(
            input_dim=semantic_dim,
            hidden_dim=semantic_dim * 2,
            output_dim=semantic_dim
        ).to(device)
        
        # 存储投影头，以便在RL环境中使用
        model.semantic_head = semantic_head
        projection_params.extend(list(semantic_head.parameters()))
        logger.info(f"  创建独立语义投影头: 参数数量: {len(list(semantic_head.parameters()))}")
    
    logger.info(f"总共找到 {len(projection_params)} 个语义投影相关参数")
    
    # 创建优化器
    optimizer = torch.optim.Adam(projection_params, lr=learning_rate)
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer()
    
    # RL训练循环
    total_steps = 0
    episode_rewards = []
    
    # 使用tqdm进度条
    for episode in tqdm(range(num_episodes), desc="Training RL", unit="episode"):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 探索或利用
            if np.random.random() < epsilon:
                # 探索: 为每个特征级别创建独立的随机投影矩阵
                action = []
                for ch in env.ch_list:
                    action.append(torch.randn(ch, semantic_dim, device=env.device))
                
                # 确保创建足够的矩阵，每个特征级别对应一个
                if len(action) < len(env.ch_list):
                    logger.warning(f"自动补充缺失的投影矩阵: 当前{len(action)}，需要{len(env.ch_list)}")
                    while len(action) < len(env.ch_list):
                        ch = env.ch_list[len(action)]
                        action.append(torch.randn(ch, semantic_dim, device=env.device))
                        
                logger.debug(f"创建探索动作: {len(action)}个投影矩阵，形状为 {[a.shape for a in action]}")
            else:
                # 利用: 使用当前模型的投影参数
                with torch.no_grad():
                    # 获取语义投影层的权重
                    action = []
                    
                    # 检查模型最后一层的类型
                    detection_head = model.model.model[-1]
                    head_type = type(detection_head).__name__
                    logger.debug(f"检测头类型: {head_type}")
                    
                    # 尝试提取投影矩阵
                    if head_type == 'OpenVocabDetect':
                        if hasattr(detection_head, 'semantic_projection'):
                            # 提取语义投影MLP的第一层权重
                            logger.debug("从semantic_projection提取投影矩阵")
                            for name, param in detection_head.semantic_projection.named_parameters():
                                if 'weight' in name and '0.weight' in name:  # 第一个线性层
                                    # 将MLP第一层权重转置为投影矩阵
                                    action.append(param.t())  # [out_features, in_features] -> [in_features, out_features]
                                    break
                        
                        if not action and hasattr(detection_head, 'semantic_extract'):
                            logger.debug("从semantic_extract提取投影矩阵")
                            for i, extract in enumerate(detection_head.semantic_extract):
                                # 获取最后一个卷积层作为投影矩阵
                                last_conv = None
                                for m in extract.modules():
                                    if isinstance(m, nn.Conv2d) and m.out_channels == semantic_dim:
                                        last_conv = m
                                
                                if last_conv is not None:
                                    # 将卷积权重重塑为投影矩阵
                                    weight = last_conv.weight.data
                                    # [out_channels, in_channels, k, k] -> [in_channels, out_channels]
                                    proj_matrix = weight.view(weight.shape[0], weight.shape[1], -1).mean(dim=2).t()
                                    action.append(proj_matrix)
                    
                    # 如果模型有自定义的语义投影头
                    if not action and hasattr(model, 'semantic_head'):
                        logger.debug("从semantic_head提取投影矩阵")
                        for name, param in model.semantic_head.named_parameters():
                            if 'weight' in name and '0.weight' in name:  # 第一个线性层
                                action.append(param.t())
                                break
                    
                    # 如果以上方法都失败，使用随机初始化
                    if not action:
                        logger.warning("无法从模型提取投影矩阵，使用随机初始化")
                        for ch in env.ch_list:
                            action.append(torch.randn(ch, semantic_dim, device=env.device))
                    
                    # 确保创建足够的矩阵，每个特征级别对应一个
                    if len(action) < len(env.ch_list):
                        logger.warning(f"自动补充缺失的投影矩阵: 当前{len(action)}，需要{len(env.ch_list)}")
                        while len(action) < len(env.ch_list):
                            ch = env.ch_list[len(action)]
                            action.append(torch.randn(ch, semantic_dim, device=env.device))
                    
                    logger.debug(f"创建利用动作: {len(action)}个投影矩阵，形状为 {[a.shape for a in action if isinstance(a, torch.Tensor)]}")
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 经验回放学习
            if len(replay_buffer) > batch_size:
                # 从回放缓冲区采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 为计算梯度，我们需要重新执行动作并获取奖励
                optimizer.zero_grad()
                
                # 为了建立计算图，重新训练每个动作获取与模型参数相关的奖励
                calculated_rewards = []
                for i in range(batch_size):
                    # 获取训练样本
                    sample_state = states[i]
                    sample_action = actions[i]
                    original_reward = rewards[i]
                    
                    # 将动作应用到当前环境以建立计算图
                    _, calculated_reward, _, _ = env.step(sample_action, eval_mode=True)
                    calculated_rewards.append(calculated_reward)
                
                # 使用有计算图的奖励计算损失
                if calculated_rewards:  # 确保列表非空
                    calculated_rewards_tensor = torch.stack(calculated_rewards)
                    
                    # 使用计算得到的奖励作为损失函数
                    loss = -torch.mean(calculated_rewards_tensor)  # 负号因为我们要最大化奖励
                    
                    # 确保梯度正确传递
                    logger.debug(f"损失计算，形状: {calculated_rewards_tensor.shape}, 值: {loss.item()}")
                    logger.debug(f"损失强制要求梯度: {loss.requires_grad}")
                    
                    # 执行反向传播
                    loss.backward()
                
                # 打印梯度信息，帮助调试
                if total_steps % 50 == 0:
                    grad_norms = []
                    for param in projection_params:
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())
                    if grad_norms:
                        logger.info(f"梯度范数: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, avg={sum(grad_norms)/len(grad_norms):.4f}")
                    else:
                        logger.warning("所有参数的梯度为None")
                
                # 标准梯度更新
                optimizer.step()
                
                # 另外，如果启用了基于奖励的参数更新，直接修改参数
                if use_reward_update:
                    # 计算平均奖励
                    avg_reward = sum(rewards) / len(rewards)
                    # 使用奖励信号更新参数
                    update_projection_params(projection_params, avg_reward, lr_scale=0.005)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # 显示进度
            if total_steps % 50 == 0:
                logger.debug(f"Episode {episode+1}/{num_episodes}, Step {total_steps}, Reward: {reward.item():.4f}")
        
        # 更新探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # 记录回合奖励
        episode_rewards.append(episode_reward.item())
        
        # 更新tqdm描述信息，显示当前奖励和探索率
        tqdm.write(f"Episode {episode+1} completed. Total reward: {episode_reward.item():.4f}, Epsilon: {epsilon:.4f}")
        
        # 每隔一定回合保存模型
        if (episode + 1) % 50 == 0:
            model_save_path = os.path.join(save_dir, f"model_ep{episode+1}.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")
    
    # 训练完成后保存最终模型
    final_model_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 输出训练统计信息
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    logger.info(f"训练完成! 总步数: {total_steps}, 平均奖励: {avg_reward:.4f}")
    
    return model, episode_rewards


# 主函数：将所有组件整合在一起
def main():
    # 1. 注册自定义模块
    register_custom_modules()
    
    # 2. 修改YAML配置
    original_yaml = "dam.yaml"
    modified_yaml = "dam_openvocab.yaml"
    new_yaml_path = modify_yaml_config(original_yaml, modified_yaml)
    
    # 查看修改后的配置文件内容
    with open(new_yaml_path, 'r') as f:
        yaml_content = f.read()
        logger.debug(f"修改后的配置文件内容:\n{yaml_content}")
        
    # 3. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 4. 创建示例类别名称
    class_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
        15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella"
    }
    
    # 5. 使用CLIP获取文本特征
    text_embeddings = get_text_embeddings(class_names, device=device)
    logger.info(f"文本嵌入形状: {text_embeddings.shape}")
    
    # 6. 直接使用原始YOLO模型并手动替换检测头
    from ultralytics import YOLO
    
    # 直接使用原始模型
    model = YOLO("yolo11n.pt")
    
    # 手动替换检测头
    logger.info("手动替换检测头...")
    
    # 提取原始检测头的anchors和stride以及其他属性
    original_detect = model.model.model[-1]
    original_anchors = original_detect.anchors.clone().cpu().numpy()  # 转为numpy数组以避免引用
    original_stride = original_detect.stride.clone() if hasattr(original_detect, 'stride') else None
    
    # 提取输入通道数
    input_channels = []
    if hasattr(original_detect, 'cv2') and isinstance(original_detect.cv2, nn.ModuleList):
        # YOLOv5风格
        for m in original_detect.cv2:
            if hasattr(m, 'in_channels'):
                input_channels.append(m.in_channels)
    elif hasattr(original_detect, 'm') and isinstance(original_detect.m, nn.ModuleList):
        # YOLOv8风格
        for m in original_detect.m:
            if hasattr(m, 'conv') and hasattr(m.conv, 'in_channels'):
                input_channels.append(m.conv.in_channels)
    elif hasattr(original_detect, 'conv') and isinstance(original_detect.conv, nn.ModuleList):
        # 另一种可能的结构
        for m in original_detect.conv:
            if hasattr(m, 'in_channels'):
                input_channels.append(m.in_channels)
    
    # 如果无法从原始检测头中提取，则使用默认值
    if not input_channels:
        logger.warning("无法确定输入通道数，使用默认值: [64, 128, 256]")
        input_channels = [64, 128, 256]
    
    # 打印推断出的通道数
    logger.info(f"推断的输入通道数: {input_channels}")
    
    # 创建新的OpenVocabDetect实例
    detection_head = OpenVocabDetect(
        nc=80,
        anchors=original_anchors,  # 使用numpy数组
        ch=input_channels,  # 使用推断的输入通道数
        semantic_dim=512
    )
    
    # 从原始检测头复制必要的属性
    detection_head.f = original_detect.f if hasattr(original_detect, 'f') else -1  # 输入来源层
    detection_head.i = original_detect.i if hasattr(original_detect, 'i') else len(model.model.model) - 1  # 模块索引
    detection_head.type = original_detect.type if hasattr(original_detect, 'type') else 'OpenVocabDetect'
    
    # 设置stride
    if original_stride is not None:
        detection_head.stride = original_stride
    
    # 设置inplace属性
    if hasattr(original_detect, 'inplace'):
        detection_head.inplace = original_detect.inplace
    
    # 替换检测头
    model.model.model[-1] = detection_head
    logger.info("成功替换为自定义检测头OpenVocabDetect")
    
    # 提取真实的通道数
    ch_list = []
    for conv in model.model.model[-1].reg_conv:
        if hasattr(conv, 'in_channels'):
            ch_list.append(conv.in_channels)
    
    if not ch_list:
        logger.warning("无法从新检测头确定输入通道数，使用默认值[64, 128, 256]")
        ch_list = [64, 128, 256]
    
    logger.info(f"新检测头实际使用的输入通道数: {ch_list}")
    
    # 确保整个模型在正确的设备上
    model.model = model.model.to(device)
    
    # 冻结backbone和neck
    logger.info("冻结backbone和neck，只训练检测头...")
    for m in model.model.model[:-1]:
        for param in m.parameters():
            param.requires_grad = False
            
    # 确保检测头是可训练的
    for param in model.model.model[-1].parameters():
        param.requires_grad = True
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.model.parameters())
    logger.info(f"可训练参数: {trainable_params}，总参数: {total_params}，比例: {trainable_params/total_params:.2%}")
    
    # 7. 创建演示数据集
    logger.info("创建演示数据集...")
    dataset = DemoDataset(device='cpu')  # 明确在CPU上创建数据集
    
    # 8. 开始强化学习训练
    logger.info("开始强化学习训练...")
    
    # 确保模型在正确的设备上
    logger.info(f"将模型移动到设备: {device}")
    model.model = model.model.to(device)
    
    # 训练设置
    reward_weights = {'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1}
    reward_scale = 10.0
    
    # 开始训练
    model_out, rewards = train_rl(
        model_path=model.ckpt_path if hasattr(model, 'ckpt_path') else "yolo11n.pt",
        yaml_path=new_yaml_path,
        dataset=dataset,
        class_names=class_names,
        text_embeddings=text_embeddings,
        num_episodes=100,  # 降低回合数便于测试
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=4,
        semantic_dim=512,
        reward_weights=reward_weights,
        reward_scale=reward_scale,
        save_dir='runs/rl_openvocab',
        use_reward_update=True,  # 启用基于奖励的参数更新
        device=device
    )
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()