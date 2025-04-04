import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import clip
from tqdm import tqdm
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from ultralytics.engine.model import Model

# 确保我们可以导入自定义模型组件
import sys
sys.path.append('.')
try:
    from model import OpenVocabDetect
    from trainer import register_custom_modules, setup_logger, modify_yaml_config
except ImportError:
    print("警告: 无法导入自定义模型组件，请确保model.py和trainer.py在当前路径中")

# 基类和新类的定义
BASE_CLASSES = {
    # 人和交通工具
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 8: 'boat',
    
    # 交通设施
    9: 'traffic light', 11: 'stop sign',
    
    # 家具和居家物品
    13: 'bench', 24: 'backpack', 25: 'umbrella', 27: 'tie', 28: 'suitcase',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    
    # 动物
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    
    # 运动/娱乐用品
    29: 'frisbee', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    
    # 食物和餐具
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    53: 'pizza', 54: 'donut', 55: 'cake'
}

NOVEL_CLASSES = {
    # 人和交通工具
    4: 'airplane', 6: 'train',
    
    # 交通设施
    10: 'fire hydrant', 12: 'parking meter',
    
    # 家具和居家物品
    26: 'handbag', 61: 'toilet', 69: 'oven', 70: 'toaster', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
    
    # 动物
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    
    # 运动/娱乐用品
    30: 'skis', 31: 'snowboard', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    
    # 食物
    48: 'sandwich', 52: 'hot dog'
}

# 设置日志
def setup_logger(name='ZeroShotEval'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

logger = setup_logger()

class FilteredDataset:
    """过滤数据集，只保留指定类别的样本"""
    
    def __init__(self, original_dataset, class_filter, split='train'):
        """
        初始化过滤后的数据集
        
        Args:
            original_dataset: 原始数据集
            class_filter: 要保留的类别ID字典 (BASE_CLASSES 或 NOVEL_CLASSES)
            split: 'train' - 只保留base类别的样本
                   'test' - 保留所有类别的样本，用于测试
        """
        self.dataset = original_dataset
        self.class_filter = class_filter
        self.split = split
        self.logger = setup_logger()
        
        # 根据分割模式和过滤器确定要保留的样本索引
        self.valid_indices = self._get_valid_indices()
        self.logger.info(f"创建{split}数据集，从{len(original_dataset)}个样本中筛选出{len(self.valid_indices)}个有效样本")
        
    def _get_valid_indices(self):
        """获取符合条件的样本索引"""
        valid_indices = []
        
        # 添加进度条
        self.logger.info(f"筛选{self.split}数据集样本...")
        
        # 统计变量
        total_samples = len(self.dataset)
        samples_with_base = 0
        samples_with_novel = 0
        samples_with_both = 0
        samples_with_unknown = 0
        
        for i in tqdm(range(total_samples), desc=f"筛选{self.split}数据集"):
            try:
                sample = self.dataset[i]
                
                # 提取类别信息 - 支持多种COCO数据格式
                classes = None
                
                # 格式 1: {'img': img, 'gt': {'boxes': boxes, 'classes': classes}}
                if isinstance(sample, dict) and 'gt' in sample and 'classes' in sample['gt']:
                    classes = sample['gt']['classes']
                
                # 格式 2: {'img': img, 'label': [cls, boxes]} 或类似格式
                elif isinstance(sample, dict) and 'label' in sample:
                    if isinstance(sample['label'], (list, tuple)) and len(sample['label']) > 0:
                        classes = sample['label'][0]  # 假设第一个元素是类别
                
                # 格式 3: (img, boxes, classes)
                elif isinstance(sample, (list, tuple)) and len(sample) > 2:
                    classes = sample[2]
                
                # 格式 4: {'img': img, 'cls': classes}
                elif isinstance(sample, dict) and 'cls' in sample:
                    classes = sample['cls']
                
                # 如果类别信息还是没提取出来，尝试遍历样本的所有字段
                if classes is None and isinstance(sample, dict):
                    for key, value in sample.items():
                        if key.lower().find('class') >= 0 or key.lower().find('cls') >= 0:
                            classes = value
                            break
                
                # 如果还是找不到类别信息，跳过该样本
                if classes is None:
                    samples_with_unknown += 1
                    continue
                
                # 确保classes是张量或列表等可迭代对象
                if not hasattr(classes, '__iter__'):
                    classes = [classes]
                
                # 检查类别ID - 安全地处理不同类型
                class_ids = []
                for c in classes:
                    # 尝试提取类别ID，处理不同格式
                    if hasattr(c, 'item'):  # 张量
                        class_ids.append(c.item())
                    elif isinstance(c, (int, float)):  # 原始数字
                        class_ids.append(int(c))
                    elif isinstance(c, str) and c.isdigit():  # 字符串数字
                        class_ids.append(int(c))
                
                # 检查样本包含的类别
                has_base = any(c_id in BASE_CLASSES for c_id in class_ids)
                has_novel = any(c_id in NOVEL_CLASSES for c_id in class_ids)
                
                # 更新统计信息
                if has_base:
                    samples_with_base += 1
                if has_novel:
                    samples_with_novel += 1
                if has_base and has_novel:
                    samples_with_both += 1
                
                # 确定是否保留样本
                keep = False
                
                if self.split == 'train':
                    # 训练集：改为更松散的条件 - 包含任何基类就保留
                    # 对于COCO这样的数据集，图片通常只有一个类别，所以这通常意味着不包含新类
                    keep = has_base
                    
                    # 如果严格要求：只保留纯基类样本（不包含任何新类）
                    # keep = has_base and not has_novel
                elif self.split == 'test':
                    # 测试集：保留包含指定类别过滤器中任何类别的样本
                    keep = any(c_id in self.class_filter for c_id in class_ids)
                
                if keep:
                    valid_indices.append(i)
                    
            except Exception as e:
                self.logger.warning(f"处理样本 {i} 时出错: {str(e)}")
        
        # 输出统计信息
        self.logger.info(f"{self.split}数据集统计:")
        self.logger.info(f"总样本数: {total_samples}")
        self.logger.info(f"包含基类的样本: {samples_with_base} ({samples_with_base/total_samples*100:.2f}%)")
        self.logger.info(f"包含新类的样本: {samples_with_novel} ({samples_with_novel/total_samples*100:.2f}%)")
        self.logger.info(f"同时包含基类和新类的样本: {samples_with_both} ({samples_with_both/total_samples*100:.2f}%)")
        self.logger.info(f"无法识别类别的样本: {samples_with_unknown} ({samples_with_unknown/total_samples*100:.2f}%)")
        self.logger.info(f"最终保留的有效样本: {len(valid_indices)} ({len(valid_indices)/total_samples*100:.2f}%)")
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.dataset[real_idx]


class ZeroShotEvaluator:
    """零样本目标检测评估器"""
    
    def __init__(self, model, device='cuda'):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            device: 使用的设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.logger = setup_logger()
        
        # 加载CLIP模型
        self.logger.info("加载CLIP模型...")
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # 获取基类和新类的文本特征
        self.logger.info("获取类别文本特征...")
        self.base_embeddings = self._get_clip_embeddings(list(BASE_CLASSES.values()))
        self.novel_embeddings = self._get_clip_embeddings(list(NOVEL_CLASSES.values()))
        
        # 所有类别的嵌入
        self.all_embeddings = torch.cat([self.base_embeddings, self.novel_embeddings], dim=0)
        self.all_class_names = list(BASE_CLASSES.values()) + list(NOVEL_CLASSES.values())
        
        self.logger.info(f"ZeroShotEvaluator初始化完成，基类特征:{self.base_embeddings.shape}，新类特征:{self.novel_embeddings.shape}")
        
    def _get_clip_embeddings(self, class_names):
        """获取类别的CLIP文本嵌入"""
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            return F.normalize(text_features, dim=-1)
    
    def evaluate(self, test_dataset, iou_threshold=0.5, conf_threshold=0.25):
        """
        评估模型在给定测试集上的零样本性能
        
        Args:
            test_dataset: 测试数据集
            iou_threshold: IoU阈值，用于判断检测是否正确
            conf_threshold: 置信度阈值
            
        Returns:
            评估结果字典，包含基类和新类的性能指标
        """
        self.logger.info(f"开始评估，IoU阈值:{iou_threshold}，置信度阈值:{conf_threshold}")
        
        results = {
            'base': {'TP': 0, 'FP': 0, 'FN': 0, 'AP': []},
            'novel': {'TP': 0, 'FP': 0, 'FN': 0, 'AP': []},
            'all': {'TP': 0, 'FP': 0, 'FN': 0, 'AP': []}
        }
        
        # 遍历测试集
        for batch_idx, batch in enumerate(tqdm(test_dataset, desc="Evaluating")):
            # 准备数据
            # 处理批次中的样本
            for sample in batch:
                # 提取图像和标签
                if isinstance(sample, dict) and 'img' in sample:
                    image = sample['img'].to(self.device)
                    
                    # 确保图像是4D：[batch, channels, height, width]
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    
                    # 获取GT标注
                    if 'gt' in sample and 'boxes' in sample['gt'] and 'classes' in sample['gt']:
                        gt_boxes = sample['gt']['boxes'].to(self.device)
                        gt_classes = sample['gt']['classes'].to(self.device)
                    else:
                        # 如果没有GT信息，跳过该样本
                        continue
                    
                    # 使用模型预测
                    with torch.no_grad():
                        outputs = self.model(image)
                        if isinstance(outputs, list) and len(outputs) == 2:  # 确保输出格式正确
                            box_preds, semantic_features = outputs
                        else:
                            self.logger.warning(f"模型输出格式不正确，期望[boxes, semantics]但得到{type(outputs)}")
                            continue
                    
                    # 确保维度正确
                    sample_boxes = box_preds[0] if box_preds.dim() > 2 else box_preds
                    sample_semantic = semantic_features[0] if semantic_features.dim() > 2 else semantic_features
                    
                    # 使用语义特征计算与所有类别的相似度
                    similarities = torch.matmul(sample_semantic, self.all_embeddings.T)
                    
                    # 获取每个框的最大相似度和对应的类别
                    max_sim, pred_classes = similarities.max(dim=-1)
                    
                    # 应用置信度阈值
                    valid_mask = max_sim > conf_threshold
                    valid_boxes = sample_boxes[valid_mask]
                    valid_classes = pred_classes[valid_mask]
                    valid_scores = max_sim[valid_mask]
                    
                    # 如果没有有效预测，跳过当前样本
                    if len(valid_boxes) == 0:
                        continue
                    
                    # 将预测类别转换为实际类别ID
                    pred_class_ids = []
                    for cls_idx in valid_classes:
                        class_name = self.all_class_names[cls_idx]
                        # 找到对应的类别ID
                        class_id = None
                        for id_, name in BASE_CLASSES.items():
                            if name == class_name:
                                class_id = id_
                                break
                        if class_id is None:
                            for id_, name in NOVEL_CLASSES.items():
                                if name == class_name:
                                    class_id = id_
                                    break
                        if class_id is not None:
                            pred_class_ids.append(class_id)
                    
                    if not pred_class_ids:
                        continue
                        
                    pred_class_ids = torch.tensor(pred_class_ids, device=self.device)
                    
                    # 评估检测结果
                    self._evaluate_predictions(
                        valid_boxes, pred_class_ids, valid_scores,
                        gt_boxes, gt_classes,
                        results, iou_threshold
                    )
        
        # 计算性能指标
        for split in ['base', 'novel', 'all']:
            r = results[split]
            precision = r['TP'] / (r['TP'] + r['FP']) if r['TP'] + r['FP'] > 0 else 0
            recall = r['TP'] / (r['TP'] + r['FN']) if r['TP'] + r['FN'] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            r['precision'] = precision
            r['recall'] = recall
            r['f1'] = f1
            r['mAP'] = np.mean(r['AP']) if r['AP'] else 0
            
            self.logger.info(f"{split}类别结果: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return results
    
    def _evaluate_predictions(self, pred_boxes, pred_classes, pred_scores, 
                              gt_boxes, gt_classes, results, iou_threshold):
        """评估单个样本的预测结果"""
        # 如果没有GT框，所有预测都是FP
        if len(gt_boxes) == 0:
            for cls_id in pred_classes:
                cls_id = cls_id.item()
                if cls_id in BASE_CLASSES:
                    results['base']['FP'] += 1
                    results['all']['FP'] += 1
                elif cls_id in NOVEL_CLASSES:
                    results['novel']['FP'] += 1
                    results['all']['FP'] += 1
            return
        
        # 计算预测框与所有GT框的IoU
        ious = box_iou(pred_boxes, gt_boxes)
        
        # 跟踪已匹配的GT框
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=self.device)
        
        # 按置信度排序处理预测框
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for idx in sorted_indices:
            pred_cls = pred_classes[idx].item()
            max_iou, max_idx = ious[idx].max(dim=0)
            
            # 如果IoU超过阈值且GT框未匹配
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                gt_cls = gt_classes[max_idx].item()
                
                # 仅当类别匹配时才算作TP
                if pred_cls == gt_cls:
                    gt_matched[max_idx] = True
                    
                    # 更新相应类别组的TP
                    if gt_cls in BASE_CLASSES:
                        results['base']['TP'] += 1
                        results['all']['TP'] += 1
                    elif gt_cls in NOVEL_CLASSES:
                        results['novel']['TP'] += 1
                        results['all']['TP'] += 1
                else:
                    # 类别不匹配，算作FP
                    if pred_cls in BASE_CLASSES:
                        results['base']['FP'] += 1
                        results['all']['FP'] += 1
                    elif pred_cls in NOVEL_CLASSES:
                        results['novel']['FP'] += 1
                        results['all']['FP'] += 1
            else:
                # IoU不足，算作FP
                if pred_cls in BASE_CLASSES:
                    results['base']['FP'] += 1
                    results['all']['FP'] += 1
                elif pred_cls in NOVEL_CLASSES:
                    results['novel']['FP'] += 1
                    results['all']['FP'] += 1
        
        # 未匹配的GT框算作FN
        for i, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = gt_classes[i].item()
                if gt_cls in BASE_CLASSES:
                    results['base']['FN'] += 1
                    results['all']['FN'] += 1
                elif gt_cls in NOVEL_CLASSES:
                    results['novel']['FN'] += 1
                    results['all']['FN'] += 1
    
    def evaluate_semantic_consistency(self, test_pairs):
        """
        评估模型在语义关系上的一致性
        
        Args:
            test_pairs: 测试对，每个对包含两个语义相关的类别和它们应有的关系
                      例如：[('man', 'woman', 'gender'), ('dog', 'puppy', 'age')]
        
        Returns:
            语义一致性分数
        """
        self.logger.info(f"评估语义一致性，测试对数量: {len(test_pairs)}")
        consistency_scores = []
        
        for word1, word2, relation in test_pairs:
            # 获取两个词的文本嵌入
            emb1 = self._get_clip_embeddings([word1])[0]
            emb2 = self._get_clip_embeddings([word2])[0]
            
            # 计算它们的差向量
            diff_vector = emb2 - emb1
            
            # 获取模型学习到的相应的语义特征
            if hasattr(self.model.model.model[-1], 'semantic_projection'):
                proj_layer = self.model.model.model[-1].semantic_projection[1]  # 获取线性投影层
                weights = proj_layer.weight.data
                
                # 在权重空间中寻找与差向量最一致的方向
                similarities = F.cosine_similarity(diff_vector.unsqueeze(0), weights, dim=1)
                max_sim = similarities.max().item()
                
                # 记录相似度
                consistency_scores.append((relation, max_sim))
                self.logger.info(f"关系 '{relation}' ({word1}->{word2}) 相似度: {max_sim:.4f}")
            else:
                self.logger.warning(f"模型没有语义投影层，无法评估关系 '{relation}'")
        
        return consistency_scores
    
    def visualize_semantic_space(self, class_subset=None, method='tsne', output_path="semantic_space.png"):
        """
        可视化模型学习的语义空间
        
        Args:
            class_subset: 要可视化的类别子集，如果为None则使用所有类别
            method: 降维方法，'tsne'或'pca'
            output_path: 输出图像路径
        """
        self.logger.info("可视化语义空间...")
        
        # 获取模型的投影层权重
        if hasattr(self.model.model.model[-1], 'semantic_projection'):
            weights = self.model.model.model[-1].semantic_projection[1].weight.data
        else:
            # 如果模型没有明确的投影层，使用文本嵌入
            if class_subset:
                weights = self._get_clip_embeddings(class_subset)
            else:
                weights = self.all_embeddings
        
        # 使用t-SNE降维
        weights_np = weights.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(weights_np)
        
        # 绘制降维结果
        plt.figure(figsize=(12, 10))
        class_names = class_subset if class_subset else self.all_class_names
        
        for i, (point, name) in enumerate(zip(embedded, class_names)):
            x, y = point
            # 用不同颜色标记基类和新类
            color = 'blue' if name in BASE_CLASSES.values() else 'red'
            marker = 'o' if name in BASE_CLASSES.values() else '^'
            plt.scatter(x, y, c=color, marker=marker)
            plt.text(x, y, name, fontsize=8)
        
        plt.title('Semantic Space Visualization (Blue: Base Classes, Red: Novel Classes)')
        plt.tight_layout()
        plt.savefig(output_path)
        self.logger.info(f"语义空间可视化已保存到 {output_path}")
        return plt


def test_semantic_composition(model, device='cuda'):
    """
    测试模型在语义组合上的能力
    
    例如：如果模型见过"红色"和"汽车"，但没见过"红色汽车"，
    测试它是否能够正确识别"红色汽车"
    """
    logger = setup_logger()
    logger.info("测试语义组合能力...")
    
    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # 定义测试情况
    test_compositions = [
        # (基础词1, 基础词2, 组合词)
        ('red', 'car', 'red car'),
        ('small', 'dog', 'small dog'),
        ('wooden', 'chair', 'wooden chair'),
        ('old', 'man', 'old man'),
        ('tall', 'building', 'tall building')
    ]
    
    results = []
    
    for base1, base2, composition in test_compositions:
        # 获取基础词和组合词的CLIP嵌入
        with torch.no_grad():
            base1_text = clip.tokenize(f"a {base1} thing").to(device)
            base2_text = clip.tokenize(f"a {base2}").to(device)
            comp_text = clip.tokenize(f"a {composition}").to(device)
            
            base1_emb = clip_model.encode_text(base1_text)
            base2_emb = clip_model.encode_text(base2_text)
            comp_emb = clip_model.encode_text(comp_text)
            
            # 归一化
            base1_emb = F.normalize(base1_emb, dim=-1)
            base2_emb = F.normalize(base2_emb, dim=-1)
            comp_emb = F.normalize(comp_emb, dim=-1)
        
        # 从模型中提取语义投影信息
        semantic_head = model.model.model[-1]
        if hasattr(semantic_head, 'semantic_projection'):
            proj_layer = semantic_head.semantic_projection[1]  # 假设第二个层是线性层
            
            # 使用模型的投影矩阵进行变换
            transformed_base1 = F.linear(base1_emb, proj_layer.weight)
            transformed_base2 = F.linear(base2_emb, proj_layer.weight)
            
            # 组合转换后的特征 (简单加法组合)
            combined_emb = transformed_base1 + transformed_base2
            combined_emb = F.normalize(combined_emb, dim=-1)
            
            # 将组合的特征投影回原始空间
            # 这里使用伪逆来近似逆变换
            weight_pinv = torch.pinverse(proj_layer.weight)
            projected_combined = F.linear(combined_emb, weight_pinv)
            projected_combined = F.normalize(projected_combined, dim=-1)
            
            # 计算投影后的组合特征与实际组合词的相似度
            similarity = F.cosine_similarity(projected_combined, comp_emb, dim=1)
            
            results.append({
                'composition': f"{base1} + {base2} -> {composition}",
                'similarity': similarity.item()
            })
            
            logger.info(f"组合 '{base1} + {base2} -> {composition}' 相似度: {similarity.item():.4f}")
            
    return results


def test_vector_arithmetic(model, device='cuda'):
    """
    测试语义向量运算能力
    
    例如：测试 man - woman + king ≈ queen 的关系是否成立
    """
    logger = setup_logger()
    logger.info("测试语义向量运算能力...")
    
    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # 定义测试案例
    test_cases = [
        # (A, B, C, D) 测试 A - B + C ≈ D
        ('man', 'woman', 'king', 'queen'),
        ('car', 'cars', 'house', 'houses'),
        ('dog', 'puppy', 'cat', 'kitten'),
        ('apple', 'apples', 'orange', 'oranges'),
        ('man', 'woman', 'boy', 'girl')
    ]
    
    results = []
    
    for a, b, c, expected_d in test_cases:
        # 获取词的CLIP嵌入
        with torch.no_grad():
            a_text = clip.tokenize(f"a {a}").to(device)
            b_text = clip.tokenize(f"a {b}").to(device)
            c_text = clip.tokenize(f"a {c}").to(device)
            d_text = clip.tokenize(f"a {expected_d}").to(device)
            
            a_emb = clip_model.encode_text(a_text)
            b_emb = clip_model.encode_text(b_text)
            c_emb = clip_model.encode_text(c_text)
            d_emb = clip_model.encode_text(d_text)
            
            # 归一化
            a_emb = F.normalize(a_emb, dim=-1)
            b_emb = F.normalize(b_emb, dim=-1)
            c_emb = F.normalize(c_emb, dim=-1)
            d_emb = F.normalize(d_emb, dim=-1)
        
        # 从模型中提取语义投影信息
        semantic_head = model.model.model[-1]
        if hasattr(semantic_head, 'semantic_projection'):
            proj_layer = semantic_head.semantic_projection[1]
            
            # 使用模型的投影矩阵进行变换
            proj_a = F.linear(a_emb, proj_layer.weight)
            proj_b = F.linear(b_emb, proj_layer.weight)
            proj_c = F.linear(c_emb, proj_layer.weight)
            proj_d = F.linear(d_emb, proj_layer.weight)
            
            # 向量运算: a - b + c
            result_vec = proj_a - proj_b + proj_c
            result_vec = F.normalize(result_vec, dim=-1)
            
            # 计算结果与预期d的相似度
            similarity = F.cosine_similarity(result_vec, proj_d, dim=1)
            
            # 同时计算在CLIP原始空间中的相似度作为对照
            clip_result = a_emb - b_emb + c_emb
            clip_result = F.normalize(clip_result, dim=-1)
            clip_similarity = F.cosine_similarity(clip_result, d_emb, dim=1)
            
            results.append({
                'case': f"{a} - {b} + {c} ≈ {expected_d}",
                'model_similarity': similarity.item(),
                'clip_similarity': clip_similarity.item()
            })
            
            logger.info(f"向量运算 '{a} - {b} + {c} ≈ {expected_d}' 模型相似度: {similarity.item():.4f}, CLIP相似度: {clip_similarity.item():.4f}")
    
    return results


def prepare_datasets(coco_dataset, batch_size=16, num_workers=4):
    """
    准备用于训练和测试的数据集
    
    Args:
        coco_dataset: 原始COCO数据集
        batch_size: 批大小
        num_workers: 工作线程数
        
    Returns:
        训练数据加载器和测试数据加载器
    """
    logger = setup_logger()
    logger.info("准备数据集...")
    
    # 创建训练集（只包含基类）
    train_dataset = FilteredDataset(coco_dataset, BASE_CLASSES, split='train')
    
    # 创建基类测试集
    base_test_dataset = FilteredDataset(coco_dataset, BASE_CLASSES, split='test')
    
    # 创建新类测试集
    novel_test_dataset = FilteredDataset(coco_dataset, NOVEL_CLASSES, split='test')
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x  # 保持原始批次格式
    )
    
    base_test_dataloader = DataLoader(
        base_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x  # 保持原始批次格式
    )
    
    novel_test_dataloader = DataLoader(
        novel_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x  # 保持原始批次格式
    )
    
    logger.info(f"数据集准备完成。训练集:{len(train_dataset)}，基类测试集:{len(base_test_dataset)}，新类测试集:{len(novel_test_dataset)}")
    
    return train_dataloader, base_test_dataloader, novel_test_dataloader


def load_model(model_path, base_model_path='yolo11n.pt', device='cuda'):
    """
    加载模型，支持多种格式
    
    Args:
        model_path: 模型路径
        base_model_path: 基础YOLO模型路径，用于重建模型结构
        device: 设备
        
    Returns:
        加载的模型
    """
    logger = setup_logger()
    logger.info(f"正在加载模型: {model_path}")
    
    try:
        # 首先尝试正常加载
        model = Model(model_path)
        logger.info("成功通过Ultralytics Model类加载模型")
        return model
    except Exception as e:
        logger.warning(f"通过Ultralytics Model类加载失败: {str(e)}")
        logger.info("尝试加载state_dict...")

        try:
            # 尝试加载为state_dict
            state_dict = torch.load(model_path, map_location=device)
            
            # 注册自定义模块
            try:
                register_custom_modules()
                logger.info("已注册自定义模块")
            except NameError:
                logger.warning("无法注册自定义模块，这可能会导致问题")
            
            # 加载基础模型
            base_model = Model(base_model_path)
            logger.info(f"已加载基础模型: {base_model_path}")
            
            # 检查模型结构，确保OpenVocabDetect作为检测头
            detection_head = base_model.model.model[-1]
            head_type = type(detection_head).__name__
            
            if head_type != 'OpenVocabDetect':
                logger.warning(f"检测头类型不是OpenVocabDetect，而是{head_type}，尝试替换...")
                
                # 创建自定义检测头并替换原始检测头
                from model import OpenVocabDetect
                
                # 获取通道信息
                ch_list = []
                for m in detection_head.modules():
                    if isinstance(m, nn.Conv2d) and m.in_channels in [64, 128, 256, 512, 1024]:
                        ch_list.append(m.in_channels)
                        
                if not ch_list:
                    ch_list = [64, 128, 256]  # 默认通道数
                
                # 提取原始检测头的属性
                if hasattr(detection_head, 'anchors'):
                    original_anchors = detection_head.anchors.clone().cpu().numpy()
                else:
                    original_anchors = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
                
                # 创建新的检测头
                new_detection_head = OpenVocabDetect(
                    nc=80,
                    anchors=original_anchors,
                    ch=ch_list,
                    semantic_dim=512
                )
                
                # 复制关键属性
                if hasattr(detection_head, 'stride'):
                    new_detection_head.stride = detection_head.stride
                if hasattr(detection_head, 'inplace'):
                    new_detection_head.inplace = detection_head.inplace
                
                # 更换检测头
                base_model.model.model[-1] = new_detection_head
                logger.info("成功替换检测头为OpenVocabDetect")
            
            # 加载state_dict
            if isinstance(state_dict, dict) and 'model' in state_dict:
                # 如果是完整模型字典
                base_model.model.load_state_dict(state_dict['model'])
                logger.info("已加载模型权重 (从'model'键)")
            elif isinstance(state_dict, dict) and any(k.startswith('model.') for k in state_dict.keys()):
                # 如果键以'model.'开头
                # 移除'model.'前缀
                new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
                base_model.model.load_state_dict(new_state_dict, strict=False)
                logger.info("已加载模型权重 (移除'model.'前缀)")
            else:
                # 直接尝试加载
                try:
                    base_model.model.load_state_dict(state_dict, strict=False)
                    logger.info("已直接加载state_dict权重")
                except Exception as e:
                    logger.warning(f"加载state_dict失败: {str(e)}")
                    logger.info("尝试最后的加载方法...")
                    
                    # 创建新的state_dict
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('model.'):
                            new_state_dict[k[6:]] = v  # 删除'model.'前缀
                        else:
                            new_state_dict[k] = v
                    
                    # 尝试加载
                    base_model.model.load_state_dict(new_state_dict, strict=False)
                    logger.info("成功加载模型权重")
            
            # 确保模型在正确的设备上
            base_model.model = base_model.model.to(device)
            
            return base_model
            
        except Exception as e:
            logger.error(f"无法加载模型: {str(e)}")
            raise e


def run_evaluation(model_path, test_dataloader, base_model_path='yolo11n.pt', output_dir='evaluation_results'):
    """
    运行完整的零样本评估
    
    Args:
        model_path: 模型路径
        test_dataloader: 测试数据加载器
        base_model_path: 基础YOLO模型路径
        output_dir: 输出目录
    """
    logger = setup_logger()
    logger.info(f"加载模型: {model_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型 - 使用改进的加载函数
    model = load_model(model_path, base_model_path)
    model.eval()
    
    # 创建评估器
    evaluator = ZeroShotEvaluator(model)
    
    # 运行基本评估
    logger.info("正在评估检测性能...")
    results = evaluator.evaluate(test_dataloader)
    
    # 保存结果
    with open(os.path.join(output_dir, 'detection_results.txt'), 'w') as f:
        f.write("零样本检测结果:\n")
        f.write("============================\n")
        f.write(f"基类 - 精确率: {results['base']['precision']:.4f}, 召回率: {results['base']['recall']:.4f}, F1: {results['base']['f1']:.4f}\n")
        f.write(f"新类 - 精确率: {results['novel']['precision']:.4f}, 召回率: {results['novel']['recall']:.4f}, F1: {results['novel']['f1']:.4f}\n")
        f.write(f"所有类 - 精确率: {results['all']['precision']:.4f}, 召回率: {results['all']['recall']:.4f}, F1: {results['all']['f1']:.4f}\n")
    
    # 测试语义一致性
    logger.info("正在评估语义一致性...")
    test_pairs = [
        ('man', 'woman', 'gender'),
        ('dog', 'puppy', 'age'),
        ('bicycle', 'motorcycle', 'motor'),
        ('cup', 'bowl', 'container'),
        ('cat', 'tiger', 'wild')
    ]
    consistency_scores = evaluator.evaluate_semantic_consistency(test_pairs)
    
    # 保存结果
    with open(os.path.join(output_dir, 'semantic_consistency.txt'), 'w') as f:
        f.write("语义一致性结果:\n")
        f.write("============================\n")
        for relation, score in consistency_scores:
            f.write(f"关系 '{relation}': {score:.4f}\n")
    
    # 测试语义组合能力
    logger.info("正在评估语义组合能力...")
    composition_results = test_semantic_composition(model)
    
    # 保存结果
    with open(os.path.join(output_dir, 'semantic_composition.txt'), 'w') as f:
        f.write("语义组合结果:\n")
        f.write("============================\n")
        for result in composition_results:
            f.write(f"组合: {result['composition']}\n")
            f.write(f"  相似度: {result['similarity']:.4f}\n")
    
    # 测试向量运算能力
    logger.info("正在评估语义向量运算能力...")
    arithmetic_results = test_vector_arithmetic(model)
    
    # 保存结果
    with open(os.path.join(output_dir, 'vector_arithmetic.txt'), 'w') as f:
        f.write("语义向量运算结果:\n")
        f.write("============================\n")
        for result in arithmetic_results:
            f.write(f"测试: {result['case']}\n")
            f.write(f"  模型相似度: {result['model_similarity']:.4f}\n")
            f.write(f"  CLIP相似度: {result['clip_similarity']:.4f}\n")
    
    # 可视化语义空间
    logger.info("正在可视化语义空间...")
    _ = evaluator.visualize_semantic_space(output_path=os.path.join(output_dir, 'semantic_space.png'))
    
    logger.info(f"评估完成，结果已保存到 {output_dir} 目录")
    
    return {
        'detection': results,
        'consistency': consistency_scores,
        'composition': composition_results,
        'arithmetic': arithmetic_results
    }


if __name__ == "__main__":
    # 该模块可以单独运行，用于评估已训练模型
    import argparse
    
    parser = argparse.ArgumentParser(description="零样本目标检测评估")
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--base-model', type=str, default='yolo11n.pt', help='基础YOLO模型路径')
    parser.add_argument('--data', type=str, required=True, help='COCO数据集路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载数据集
    from ultralytics.data.dataset import YOLODataset
    
    data_dict = {
        'path': args.data,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in list(BASE_CLASSES.items()) + list(NOVEL_CLASSES.items())}
    }
    
    test_dataset = YOLODataset(
        img_path=os.path.join(args.data, data_dict['val']),
        data=data_dict
    )
    
    # 准备数据集
    _, base_test_loader, novel_test_loader = prepare_datasets(test_dataset, batch_size=args.batch_size)
    
    # 运行评估
    run_evaluation(args.model, novel_test_loader, base_model_path=args.base_model, output_dir=args.output_dir)