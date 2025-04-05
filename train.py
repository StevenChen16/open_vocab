import os
import torch
import torch.nn as nn
import argparse
import yaml
import logging
import numpy as np
from tqdm import tqdm
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.model import Model

# 从trainer.py导入必要的方法
from trainer import (
    register_custom_modules, 
    modify_yaml_config, 
    get_text_embeddings, 
    setup_logger,
    DetectionRLEnv,
    ReplayBuffer,
    update_projection_params,
    SemanticHead
)

# 从zero_shot_eval导入零样本评估组件
from zero_shot_eval import (
    BASE_CLASSES,
    NOVEL_CLASSES,
    FilteredDataset,
    prepare_datasets,
    run_evaluation
)

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train open-vocab detector with RL")
    parser.add_argument('--data', type=str, default='data/coco8.yaml', help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='data loading workers')
    parser.add_argument('--save-dir', type=str, default='runs/train_openvocab', help='save directory')
    parser.add_argument('--semantic-dim', type=int, default=512, help='semantic feature dimension')
    parser.add_argument('--reward-scale', type=float, default=10.0, help='reward scaling factor')
    parser.add_argument('--zero-shot', action='store_true', help='enable zero-shot evaluation')
    parser.add_argument('--eval-interval', type=int, default=10, help='evaluation interval epochs')
    return parser.parse_args()

# COCO数据集适配器
class COCOAdapter:
    """
    适配COCO数据集到RL环境所需的格式
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.logger = setup_logger()
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # 获取原始数据
        batch = self.dataset[idx]
        
        # 提取图像和标签
        if isinstance(batch, dict):
            image = batch['img'] if 'img' in batch else batch['images']
            boxes = batch['bboxes'] if 'bboxes' in batch else batch.get('gt_bboxes', None)
            classes = batch['cls'] if 'cls' in batch else batch.get('gt_classes', None)
        else:
            # 尝试解析元组或列表格式
            image = batch[0] if len(batch) > 0 else torch.zeros((3, 640, 640))
            boxes = batch[1] if len(batch) > 1 else torch.zeros((0, 4))
            classes = batch[2] if len(batch) > 2 else torch.zeros((0,), dtype=torch.long)
        
        # 确保boxes和classes是张量
        if boxes is None or not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            boxes = torch.zeros((0, 4))
        if classes is None or not isinstance(classes, torch.Tensor) or classes.numel() == 0:
            classes = torch.zeros((0,), dtype=torch.long)
            
        # 构建适配后的数据格式
        return {
            'img': image,
            'gt': {
                'boxes': boxes,
                'classes': classes
            },
            'pred_boxes': boxes  # 初始预测框与GT相同
        }

def load_coco_dataset(data_yaml, img_size=640):
    """
    加载COCO数据集
    
    Args:
        data_yaml: 数据集配置文件路径
        img_size: 图像大小
        
    Returns:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        class_names: 类别名称字典
    """
    logger = setup_logger()
    
    # 创建数据配置
    data_dict = {
        'path': 'D:/workstation/ML/SAD/datasets/coco8',
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
        }
    }
    
    # 如果提供了YAML文件，则加载它
    if os.path.exists(data_yaml):
        with open(data_yaml, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            # 合并YAML配置
            for k, v in yaml_dict.items():
                data_dict[k] = v
    
    logger.info(f"数据集路径: {data_dict['path']}")
    
    # 提取数据集路径
    path = data_dict['path']
    train_path = os.path.join(path, data_dict['train'])
    val_path = os.path.join(path, data_dict['val'])
    
    # 创建训练数据集
    logger.info(f"加载训练数据集: {train_path}")
    train_dataset = YOLODataset(
        img_path=train_path,
        augment=True,  # 启用数据增强
        rect=False,  # 不使用矩形训练
        cache=False,
        stride=32,
        pad=0.0,
        data=data_dict
    )
    
    # 创建验证数据集
    logger.info(f"加载验证数据集: {val_path}")
    val_dataset = YOLODataset(
        img_path=val_path,
        augment=False,  # 验证集不需要增强
        rect=True,  # 使用矩形推理
        cache=False,
        stride=32,
        pad=0.0,
        data=data_dict
    )
    
    # 提取类别名称
    class_names = data_dict.get('names', {})
    logger.info(f"加载了 {len(class_names)} 个类别名称")
    
    # 使用适配器包装数据集
    train_adapter = COCOAdapter(train_dataset)
    val_adapter = COCOAdapter(val_dataset)
    
    logger.info(f"训练集大小: {len(train_adapter)}, 验证集大小: {len(val_adapter)}")
    
    return train_adapter, val_adapter, class_names, train_dataset, val_dataset

def train_rl_direct(
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
    zero_shot_eval=False,    # 是否启用零样本评估
    eval_interval=10,        # 评估间隔
    eval_datasets=None,      # 评估数据集
):
    """
    直接实现的RL训练函数，绕过原始train_rl函数的问题
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化日志
    logger = setup_logger()
    logger.info(f"开始RL训练，设备: {device}")
    
    # 首先注册自定义模块
    register_custom_modules()
    logger.info("已注册自定义模块")
    
    # 获取类别文本特征
    text_embeddings = get_text_embeddings(class_names, semantic_dim, device)
    logger.info(f"文本特征准备完成，形状: {text_embeddings.shape}")
    
    # 加载预训练模型
    model = Model(model_path)
    logger.info(f"已加载预训练模型: {model_path}")
    
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
    
    # 创建RL环境
    env = DetectionRLEnv(
        model=model.model,
        text_embeddings=text_embeddings,
        dataset=dataset,
        semantic_dim=semantic_dim,
        reward_weights=reward_weights,
        reward_scale=reward_scale,
        device=device,
        ch_list=ch_list,
        conf_threshold=0.05  # 使用较低的置信度阈值
    )
    logger.info("RL环境创建完成")
    
    # 定义投影层优化器
    projection_params = []
    detection_head = model.model.model[-1]
    
    # 收集语义投影层参数
    if hasattr(detection_head, 'semantic_projection'):
        logger.info(f"找到semantic_projection层，添加参数...")
        projection_params.extend(list(detection_head.semantic_projection.parameters()))
    
    if hasattr(detection_head, 'semantic_extract'):
        logger.info(f"找到semantic_extract层，添加参数...")
        for extract_module in detection_head.semantic_extract:
            params = list(extract_module.parameters())
            projection_params.extend(params)
    
    # 搜索可能的语义相关层
    if not projection_params:
        logger.warning("未找到标准语义层，搜索其他可能的语义相关层...")
        # 搜索所有可能的线性层或卷积层作为语义相关层
        for name, module in detection_head.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params = list(module.parameters())
                if params:
                    logger.info(f"找到可能的语义层 {name}，添加 {len(params)} 个参数")
                    projection_params.extend(params)
    
    # 如果仍然没有参数，创建一个新的语义投影头
    if not projection_params:
        logger.warning("未找到任何可训练参数，创建自定义语义投影层...")
        # 创建一个简单的语义投影头
        semantic_head = SemanticHead(
            input_dim=semantic_dim,
            hidden_dim=semantic_dim * 2,
            output_dim=semantic_dim
        ).to(device)
        
        # 将投影头添加到模型中
        model.semantic_head = semantic_head
        
        # 添加参数
        projection_params.extend(list(semantic_head.parameters()))
        
        logger.info(f"创建了自定义语义投影头，参数数量: {len(list(semantic_head.parameters()))}")
    
    # 检查参数数量
    if not projection_params:
        # 如果仍然没有参数，创建一个虚拟参数以避免优化器错误
        logger.warning("仍然没有找到参数，创建虚拟参数...")
        dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        model.dummy_param = dummy_param
        projection_params.append(dummy_param)
    
    # 输出参数统计
    logger.info(f"总共找到 {len(projection_params)} 个可训练参数")
    
    # 确保所有参数都在设备上
    for param in projection_params:
        if param.device != torch.device(device):
            param.data = param.data.to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(projection_params, lr=learning_rate)
    logger.info(f"优化器已创建，学习率: {learning_rate}")
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer()
    
    # RL训练循环
    total_steps = 0
    episode_rewards = []
    current_epsilon = epsilon  # 初始探索率
    
    # 使用tqdm进度条
    for episode in tqdm(range(num_episodes), desc="Training RL", unit="episode"):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 探索或利用
            if np.random.random() < current_epsilon:
                # 探索: 创建随机投影矩阵
                action = []
                for ch in env.ch_list:
                    # 改进随机投影矩阵初始化
                    proj_matrix = torch.randn(ch, semantic_dim, device=device)
                    # 正交初始化
                    if ch <= semantic_dim:
                        q, _ = torch.linalg.qr(proj_matrix.T)
                        proj_matrix = q.T[:ch]
                    else:
                        q, _ = torch.linalg.qr(proj_matrix)
                        proj_matrix = q[:ch]
                    
                    # 确保保持梯度
                    proj_matrix = proj_matrix.detach().clone().requires_grad_(True)
                    action.append(proj_matrix)
            else:
                # 利用: 使用当前模型的投影参数
                with torch.no_grad():
                    # 获取语义投影层的权重
                    action = []
                    
                    # 尝试提取投影矩阵
                    if hasattr(detection_head, 'semantic_projection'):
                        for name, param in detection_head.semantic_projection.named_parameters():
                            if 'weight' in name and '0.weight' in name:
                                # 克隆参数并启用梯度
                                proj_matrix = param.t().detach().clone().requires_grad_(True)
                                action.append(proj_matrix)
                                break
                    
                    if not action and hasattr(detection_head, 'semantic_extract'):
                        for extract in detection_head.semantic_extract:
                            for m in extract.modules():
                                if isinstance(m, nn.Conv2d) and m.out_channels == semantic_dim:
                                    weight = m.weight.data
                                    proj_matrix = weight.view(weight.shape[0], weight.shape[1], -1).mean(dim=2).t()
                                    # 克隆并启用梯度
                                    proj_matrix = proj_matrix.detach().clone().requires_grad_(True)
                                    action.append(proj_matrix)
                    
                    # 如果以上方法都失败，使用随机初始化
                    if not action:
                        for ch in env.ch_list:
                            proj_matrix = torch.randn(ch, semantic_dim, device=device)
                            # 正交初始化
                            if ch <= semantic_dim:
                                q, _ = torch.linalg.qr(proj_matrix.T)
                                proj_matrix = q.T[:ch]
                            else:
                                q, _ = torch.linalg.qr(proj_matrix)
                                proj_matrix = q[:ch]
                            # 确保保持梯度
                            proj_matrix = proj_matrix.requires_grad_(True)
                            action.append(proj_matrix)
                    
                    # 确保创建足够的矩阵
                    while len(action) < len(env.ch_list):
                        ch = env.ch_list[len(action)]
                        proj_matrix = torch.randn(ch, semantic_dim, device=device)
                        # 正交初始化
                        if ch <= semantic_dim:
                            q, _ = torch.linalg.qr(proj_matrix.T)
                            proj_matrix = q.T[:ch]
                        else:
                            q, _ = torch.linalg.qr(proj_matrix)
                            proj_matrix = q[:ch]
                        # 确保保持梯度
                        proj_matrix = proj_matrix.requires_grad_(True)
                        action.append(proj_matrix)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 经验回放学习
            if len(replay_buffer) > batch_size:
                # 从回放缓冲区采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 为计算梯度，重新执行动作并获取奖励
                optimizer.zero_grad()
                
                # 为了建立计算图，重新执行每个动作获取与模型参数相关的奖励
                calculated_rewards = []
                
                for i in range(batch_size):
                    sample_action = actions[i]
                    
                    # 确保动作保持梯度
                    if isinstance(sample_action, list):
                        # 对于列表，确保每个矩阵都保持梯度
                        sample_action_with_grad = []
                        for mat in sample_action:
                            # 确保每个矩阵都有梯度
                            if isinstance(mat, torch.Tensor) and not mat.requires_grad:
                                mat = mat.detach().clone().requires_grad_(True)
                            sample_action_with_grad.append(mat)
                        sample_action = sample_action_with_grad
                    elif isinstance(sample_action, torch.Tensor) and not sample_action.requires_grad:
                        # 对于单个张量，确保它有梯度
                        sample_action = sample_action.detach().clone().requires_grad_(True)
                    
                    # 使用eval_mode=True，仅计算奖励而不更新环境状态
                    _, calculated_reward, _, _ = env.step(sample_action, eval_mode=True)
                    
                    # 检查奖励是否有梯度信息
                    if isinstance(calculated_reward, torch.Tensor):
                        # 确保奖励保持梯度
                        if not calculated_reward.requires_grad:
                            logger.warning(f"奖励没有梯度信息，创建新的可导奖励")
                            # 创建新的与原始奖励值相同但有梯度的张量
                            proxy_reward = calculated_reward.detach().clone().requires_grad_(True)
                            # 另一种方法是使用原始参数直接计算，确保保持梯度连接
                            for action_tensor in sample_action:
                                if action_tensor.requires_grad:
                                    # 找到一个可导的参数，将其连接到奖励上
                                    proxy_reward = proxy_reward + 0.0 * action_tensor.sum()
                            calculated_rewards.append(proxy_reward)
                        else:
                            calculated_rewards.append(calculated_reward)
                
                # 计算损失并更新参数
                if calculated_rewards:
                    # 处理不同形状的张量
                    if all(r.dim() == 0 for r in calculated_rewards):
                        # 所有张量都是标量，可以用stack
                        rewards_tensor = torch.stack(calculated_rewards)
                    else:
                        # 确保每个奖励都是标量
                        rewards_tensor = torch.stack([r.mean() if r.dim() > 0 else r for r in calculated_rewards])
                    
                    # 计算损失 - 我们要最大化奖励，所以取负
                    loss = -torch.mean(rewards_tensor)
                    
                    # 检查损失是否有梯度
                    if not loss.requires_grad:
                        logger.warning("损失没有梯度信息，无法反向传播")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 检查梯度
                    grad_exists = False
                    for param in projection_params:
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            grad_exists = True
                            break
                    
                    if not grad_exists:
                        logger.warning("没有产生有效梯度，跳过参数更新")
                        continue
                    
                    # 应用梯度
                    optimizer.step()
                    
                    logger.debug(f"计算了损失值: {loss.item():.4f}，来自 {len(calculated_rewards)} 个奖励样本")
                
                # 如果启用了基于奖励的参数更新，直接修改参数
                if use_reward_update:
                    # 使用原始rewards（不是计算图连接的）
                    valid_rewards = [r for r in rewards if isinstance(r, torch.Tensor) and r.numel() > 0]
                    if valid_rewards:
                        avg_reward = sum(r.item() for r in valid_rewards) / len(valid_rewards)
                        # 使用奖励信号更新参数
                        update_projection_params(projection_params, avg_reward, lr_scale=0.005)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # 显示进度
            if total_steps % 50 == 0:
                logger.debug(f"Episode {episode+1}/{num_episodes}, Step {total_steps}, Reward: {reward.item():.4f}")
        
        # 更新探索率
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        
        # 记录回合奖励
        if isinstance(episode_reward, torch.Tensor):
            episode_rewards.append(episode_reward.item())
        else:
            episode_rewards.append(float(episode_reward))
        
        # 输出当前进度
        if (episode + 1) % 10 == 0 or episode == 0:
            logger.info(f"Episode {episode+1}/{num_episodes} 完成。总奖励: {episode_rewards[-1]:.4f}, 探索率: {current_epsilon:.4f}")
        
        # 每隔一定回合保存模型
        if (episode + 1) % 50 == 0 or (episode + 1) == num_episodes:
            model_save_path = os.path.join(save_dir, f"model_ep{episode+1}.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"模型已保存至 {model_save_path}")
        
        # 零样本评估
        if zero_shot_eval and eval_datasets and (episode + 1) % eval_interval == 0:
            logger.info(f"进行零样本评估 (Episode {episode+1})...")
            eval_dir = os.path.join(save_dir, f"eval_ep{episode+1}")
            os.makedirs(eval_dir, exist_ok=True)
            
            # 保存当前模型
            current_model_path = os.path.join(eval_dir, "current_model.pt")
            torch.save(model.state_dict(), current_model_path)
            
            # 执行评估 - 对基类和新类分别评估
            base_results = run_evaluation(
                model_path=current_model_path,
                test_dataloader=eval_datasets['base'],
                output_dir=os.path.join(eval_dir, "base_classes")
            )
            
            novel_results = run_evaluation(
                model_path=current_model_path,
                test_dataloader=eval_datasets['novel'],
                output_dir=os.path.join(eval_dir, "novel_classes")
            )
            
            # 记录评估结果
            base_f1 = base_results['detection']['base']['f1']
            novel_f1 = novel_results['detection']['novel']['f1']
            logger.info(f"零样本评估结果 - 基类F1: {base_f1:.4f}, 新类F1: {novel_f1:.4f}")
    
    # 训练完成后保存最终模型
    final_model_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存至 {final_model_path}")
    
    # 保存奖励历史
    reward_path = os.path.join(save_dir, "rewards.npy")
    np.save(reward_path, np.array(episode_rewards))
    
    # 输出训练统计信息
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    logger.info(f"训练完成! 总步数: {total_steps}, 平均奖励: {avg_reward:.4f}")
    
    # 最终零样本评估
    if zero_shot_eval and eval_datasets:
        logger.info("进行最终零样本评估...")
        eval_dir = os.path.join(save_dir, "final_evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        final_results = run_evaluation(
            model_path=final_model_path,
            test_dataloader=eval_datasets['novel'],
            output_dir=eval_dir
        )
        
        # 输出最终评估结果
        base_precision = final_results['detection']['base']['precision']
        base_recall = final_results['detection']['base']['recall']
        base_f1 = final_results['detection']['base']['f1']
        
        novel_precision = final_results['detection']['novel']['precision']
        novel_recall = final_results['detection']['novel']['recall']
        novel_f1 = final_results['detection']['novel']['f1']
        
        logger.info("最终零样本评估结果:")
        logger.info(f"基类 - 精确率: {base_precision:.4f}, 召回率: {base_recall:.4f}, F1: {base_f1:.4f}")
        logger.info(f"新类 - 精确率: {novel_precision:.4f}, 召回率: {novel_recall:.4f}, F1: {novel_f1:.4f}")
    
    return model, episode_rewards

def main(args):
    # 初始化日志
    logger = setup_logger()
    logger.info(f"开始训练开放词汇目标检测器，使用设备: {args.device}")
    
    # 注册自定义模块
    register_custom_modules()
    
    # 创建自定义YAML配置
    yaml_path = os.path.join(args.save_dir, "custom_openvocab.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    modified_yaml = modify_yaml_config("dam.yaml", yaml_path, args.semantic_dim)
    
    # 设置设备
    device = args.device
    
    # 加载数据集
    logger.info(f"加载COCO数据集: {args.data}")
    train_dataset, val_dataset, class_names, raw_train_dataset, raw_val_dataset = load_coco_dataset(
        data_yaml=args.data,
        img_size=args.img_size
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置奖励权重
    reward_weights = {'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1}
    
    # 准备零样本评估数据集
    eval_datasets = None
    if args.zero_shot:
        logger.info("准备零样本评估数据集...")
        train_loader, base_test_loader, novel_test_loader = prepare_datasets(
            raw_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        
        eval_datasets = {
            'base': base_test_loader,
            'novel': novel_test_loader
        }
        
        logger.info(f"零样本评估数据集准备完成")
    
    # 开始强化学习训练 - 使用我们直接实现的RL训练函数
    logger.info("开始强化学习训练...")
    model_out, rewards = train_rl_direct(
        model_path=args.model,
        yaml_path=modified_yaml,
        dataset=train_dataset,
        class_names=class_names,
        num_episodes=args.epochs,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=args.batch_size,
        semantic_dim=args.semantic_dim,
        reward_weights=reward_weights,
        reward_scale=args.reward_scale,
        save_dir=args.save_dir,
        device=device,
        use_reward_update=True,
        zero_shot_eval=args.zero_shot,
        eval_interval=args.eval_interval,
        eval_datasets=eval_datasets
    )
    
    # 保存奖励历史
    reward_path = os.path.join(args.save_dir, "rewards.npy")
    np.save(reward_path, rewards)
    logger.info(f"奖励历史已保存至: {reward_path}")
    
    # 输出奖励曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards during training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(args.save_dir, 'reward_curve.png'))
    logger.info("奖励曲线已保存")
    
    logger.info("训练流程已完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)