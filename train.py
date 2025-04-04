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
    train_rl, 
    setup_logger,
    SemanticHead
)

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train open-vocab detector with RL")
    parser.add_argument('--data', type=str, default='coco8.yaml', help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='data loading workers')
    parser.add_argument('--save-dir', type=str, default='runs/train_openvocab', help='save directory')
    parser.add_argument('--semantic-dim', type=int, default=512, help='semantic feature dimension')
    parser.add_argument('--reward-scale', type=float, default=10.0, help='reward scaling factor')
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
        try:
            with open(data_yaml, 'r') as f:
                yaml_dict = yaml.safe_load(f)
                # 合并YAML配置
                for k, v in yaml_dict.items():
                    data_dict[k] = v
        except Exception as e:
            logger.warning(f"无法加载YAML配置文件 {data_yaml}: {e}")
    
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
    
    return train_adapter, val_adapter, class_names

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
    train_dataset, val_dataset, class_names = load_coco_dataset(
        data_yaml=args.data,
        img_size=args.img_size
    )
    
    # 获取类别文本特征
    logger.info("获取CLIP文本特征...")
    text_embeddings = get_text_embeddings(class_names, semantic_dim=args.semantic_dim, device=device)
    
    # 加载预训练模型
    logger.info(f"加载预训练模型: {args.model}")
    model = Model(args.model)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置奖励权重
    reward_weights = {'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1}
    
    # 创建一个包装函数来修复train_rl中的问题
    def train_rl_fixed(*args, **kwargs):
        """修复版的train_rl函数，处理张量堆叠问题"""
        # 修改trainer.py中的函数
        original_train_rl = train_rl
        
        # 保存原始函数的代码
        original_code = original_train_rl.__code__
        
        # 运行原始函数，但捕获特定错误
        try:
            return original_train_rl(*args, **kwargs)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                logger.warning("捕获到tensor stack错误，尝试使用修复版本...")
                
                # 手动进行强化学习训练
                # 提取参数
                model_path = kwargs.get('model_path')
                yaml_path = kwargs.get('yaml_path')
                dataset = kwargs.get('dataset')
                class_names = kwargs.get('class_names')
                num_episodes = kwargs.get('num_episodes', 100)
                learning_rate = kwargs.get('learning_rate', 1e-4)
                batch_size = kwargs.get('batch_size', 32)
                semantic_dim = kwargs.get('semantic_dim', 512)
                reward_weights = kwargs.get('reward_weights', {'accuracy': 0.6, 'semantic': 0.3, 'exploration': 0.1})
                reward_scale = kwargs.get('reward_scale', 10.0)
                save_dir = kwargs.get('save_dir', 'runs/rl_train')
                device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                use_reward_update = kwargs.get('use_reward_update', True)
                
                from ultralytics.engine.model import Model
                from trainer import (
                    setup_logger, register_custom_modules, get_text_embeddings,
                    DetectionRLEnv, ReplayBuffer, update_projection_params
                )
                
                # 初始化日志
                logger = setup_logger()
                
                # 首先注册自定义模块
                register_custom_modules()
                
                # 获取类别文本特征
                text_embeddings = get_text_embeddings(class_names, semantic_dim, device)
                
                # 加载预训练模型
                model = Model(model_path)
                
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
                
                # 创建RL环境
                env = DetectionRLEnv(
                    model=model.model,
                    text_embeddings=text_embeddings,
                    dataset=dataset,
                    semantic_dim=semantic_dim,
                    reward_weights=reward_weights,
                    reward_scale=reward_scale,
                    device=device,
                    ch_list=ch_list
                )
                
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
                
                # 如果仍然没有参数，创建一个新的语义投影层
                if not projection_params:
                    logger.warning("未找到任何可训练参数，创建自定义语义投影层...")
                    # 创建一个简单的语义投影层
                    semantic_proj = nn.Sequential(
                        nn.Linear(semantic_dim, semantic_dim * 2),
                        nn.ReLU(),
                        nn.Linear(semantic_dim * 2, semantic_dim)
                    ).to(device)
                    
                    # 将投影层添加到模型中
                    model.semantic_proj = semantic_proj
                    
                    # 添加参数
                    projection_params.extend(list(semantic_proj.parameters()))
                    
                    logger.info(f"创建了自定义语义投影层，参数数量: {len(projection_params)}")
                
                # 检查参数数量
                if not projection_params:
                    # 如果仍然没有参数，创建一个虚拟参数以避免优化器错误
                    logger.warning("仍然没有找到参数，创建虚拟参数...")
                    dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
                    model.dummy_param = dummy_param
                    projection_params.append(dummy_param)
                
                # 输出参数统计
                logger.info(f"总共找到 {len(projection_params)} 个可训练参数")
                
                # 创建优化器
                optimizer = torch.optim.Adam(projection_params, lr=learning_rate)
                
                # 创建经验回放缓冲区
                replay_buffer = ReplayBuffer()
                
                # RL训练循环
                total_steps = 0
                episode_rewards = []
                epsilon = 0.3  # 初始探索率
                
                # 使用tqdm进度条
                for episode in tqdm(range(num_episodes), desc="Training RL", unit="episode"):
                    # 重置环境
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        # 探索或利用
                        if np.random.random() < epsilon:
                            # 探索
                            action = []
                            for ch in env.ch_list:
                                action.append(torch.randn(ch, semantic_dim, device=env.device))
                        else:
                            # 利用
                            with torch.no_grad():
                                # 获取语义投影层的权重
                                action = []
                                detection_head = model.model.model[-1]
                                
                                # 尝试提取投影矩阵
                                if hasattr(detection_head, 'semantic_projection'):
                                    for name, param in detection_head.semantic_projection.named_parameters():
                                        if 'weight' in name and '0.weight' in name:
                                            action.append(param.t())
                                            break
                                
                                if not action and hasattr(detection_head, 'semantic_extract'):
                                    for extract in detection_head.semantic_extract:
                                        for m in extract.modules():
                                            if isinstance(m, torch.nn.Conv2d) and m.out_channels == semantic_dim:
                                                weight = m.weight.data
                                                proj_matrix = weight.view(weight.shape[0], weight.shape[1], -1).mean(dim=2).t()
                                                action.append(proj_matrix)
                                
                                # 如果以上方法都失败，使用随机初始化
                                if not action:
                                    for ch in env.ch_list:
                                        action.append(torch.randn(ch, semantic_dim, device=env.device))
                                
                                # 确保创建足够的矩阵
                                while len(action) < len(env.ch_list):
                                    ch = env.ch_list[len(action)]
                                    action.append(torch.randn(ch, semantic_dim, device=env.device))
                        
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
                            
                            # 为了建立计算图，重新训练每个动作获取与模型参数相关的奖励
                            calculated_rewards = []
                            for i in range(batch_size):
                                sample_action = actions[i]
                                
                                try:
                                    _, calculated_reward, _, _ = env.step(sample_action, eval_mode=True)
                                    calculated_rewards.append(calculated_reward)
                                except Exception as e:
                                    logger.warning(f"计算奖励时出错: {e}")
                            
                            # 修复：过滤掉空或无效的奖励
                            calculated_rewards = [r for r in calculated_rewards if isinstance(r, torch.Tensor) and r.numel() > 0]
                            
                            # 使用有计算图的奖励计算损失
                            if calculated_rewards:
                                try:
                                    # 检查所有张量的形状
                                    shapes = [r.shape for r in calculated_rewards]
                                    all_same_shape = all(s == shapes[0] for s in shapes)
                                    
                                    if all_same_shape:
                                        # 形状相同，使用stack
                                        calculated_rewards_tensor = torch.stack(calculated_rewards)
                                    else:
                                        # 形状不同，使用flatten和cat
                                        calculated_rewards_tensor = torch.cat([r.view(-1) for r in calculated_rewards])
                                    
                                    # 计算损失
                                    loss = -torch.mean(calculated_rewards_tensor)
                                    
                                    # 执行反向传播
                                    loss.backward()
                                except Exception as e:
                                    logger.warning(f"损失计算出错: {e}")
                            
                            # 标准梯度更新
                            optimizer.step()
                            
                            # 如果启用了基于奖励的参数更新，直接修改参数
                            if use_reward_update:
                                # 计算平均奖励
                                avg_reward = sum(r.item() for r in rewards if isinstance(r, torch.Tensor)) / len(rewards)
                                # 使用奖励信号更新参数
                                update_projection_params(projection_params, avg_reward, lr_scale=0.005)
                        
                        state = next_state
                        episode_reward += reward
                        total_steps += 1
                        
                        # 显示进度
                        if total_steps % 50 == 0:
                            logger.debug(f"Episode {episode+1}/{num_episodes}, Step {total_steps}, Reward: {reward.item():.4f}")
                    
                    # 更新探索率
                    epsilon = max(0.05, epsilon * 0.995)
                    
                    # 记录回合奖励
                    episode_rewards.append(episode_reward.item())
                    
                    # 每隔一定回合保存模型
                    if (episode + 1) % 50 == 0:
                        model_save_path = os.path.join(save_dir, f"model_ep{episode+1}.pt")
                        torch.save(model.state_dict(), model_save_path)
                        logger.info(f"Model saved to {model_save_path}")
                
                # 训练完成
                return model, episode_rewards
            else:
                # 如果是其他错误，重新抛出
                raise
    
    # 开始强化学习训练
    logger.info("开始强化学习训练...")
    try:
        model_out, rewards = train_rl_fixed(
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
            use_reward_update=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, "model_final.pt")
        torch.save(model_out.state_dict(), final_model_path)
        logger.info(f"训练完成，最终模型已保存至: {final_model_path}")
        
        # 保存奖励历史
        reward_path = os.path.join(args.save_dir, "rewards.npy")
        np.save(reward_path, rewards)
        logger.info(f"奖励历史已保存至: {reward_path}")
        
        # 输出奖励曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title('Rewards during training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(args.save_dir, 'reward_curve.png'))
            logger.info("奖励曲线已保存")
        except ImportError:
            logger.warning("无法导入matplotlib，跳过绘制奖励曲线")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("训练流程已完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)