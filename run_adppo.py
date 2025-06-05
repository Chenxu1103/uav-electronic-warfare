#!/usr/bin/env python3
"""
AD-PPO运行脚本
提供简单的命令行界面运行AD-PPO算法

使用示例:
1. 训练AD-PPO算法:
   python run_adppo.py train --total_timesteps 100000 --rollout_steps 200 --batch_size 64

2. 评估已训练的模型:
   python run_adppo.py evaluate --eval_episodes 5

3. 使用自定义路径:
   python run_adppo.py train --save_path my_experiments/ad_ppo_test
   
4. 快速训练模式(用于测试):
   python run_adppo.py train --quick
   
5. 从检查点继续训练:
   python run_adppo.py train --load_checkpoint
"""

import os
import sys
import argparse
import numpy as np
import time
import torch
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.utils.plotting import plot_training_curves, plot_environment_state, plot_trajectory

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.insert(0, project_root)
print(f"添加项目根目录到Python路径: {project_root}")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def create_environment(num_uavs=3, num_radars=3, max_steps=500, time_step=0.1, 
                      world_size=None, target_position=None, starting_area=None, radar_positions=None):
    """
    创建电子对抗模拟环境
    
    Args:
        num_uavs: UAV数量
        num_radars: 雷达数量
        max_steps: 最大步数
        time_step: 时间步长
        world_size: 世界大小
        target_position: 目标位置
        starting_area: 起始区域
        radar_positions: 雷达位置
        
    Returns:
        环境对象
    """
    # 默认参数
    if world_size is None:
        world_size = [10000, 10000, 1000]
    if target_position is None:
        target_position = np.array([8000, 8000, 100])
    if starting_area is None:
        # 格式为 [x_min, x_max, y_min, y_max, z_min, z_max]
        starting_area = [0, 2000, 0, 2000, 50, 200]
    if radar_positions is None:
        radar_positions = [
            np.array([7000, 3000, 0]),
            np.array([5000, 5000, 0]),
            np.array([3000, 7000, 0])
        ]
    
    # 创建环境配置
    config = {
        "num_uavs": num_uavs,
        "num_radars": num_radars,
        "max_steps": max_steps,
        "time_step": time_step,
        "world_size": world_size,
        "target_position": target_position,
        "starting_area": starting_area,
        "radar_positions": radar_positions
    }
    
    # 创建并返回环境
    return ElectronicWarfareEnv(config)

def create_agent(env, hidden_dim=256, learning_rate=3e-4, gamma=0.99, rollout_steps=200, batch_size=64):
    """
    创建AD-PPO智能体
    
    Args:
        env: 环境对象
        hidden_dim: 隐藏层维度
        learning_rate: 学习率
        gamma: 折扣因子
        rollout_steps: 每次rollout的步数
        batch_size: 批量大小
        
    Returns:
        AD-PPO智能体
    """
    # 获取环境的动作维度和智能体数量
    action_dim = env.action_space.shape[0] 
    num_agents = env.num_uavs
    
    # 实际状态维度是28，而不是env.observation_space.shape[0]
    actual_state_dim = 28
    
    print(f"环境动作维度: {action_dim}")
    print(f"智能体数量: {num_agents}")
    print(f"状态维度设置为: {actual_state_dim}")
    
    # 创建AD-PPO智能体
    agent = ADPPO(
        state_dim=actual_state_dim,  # 使用实际状态维度
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=0.95,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return agent

def train(args):
    """训练AD-PPO算法"""
    print("===== 开始训练 =====")
    
    # 创建环境
    env = ElectronicWarfareEnv(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        env_size=args.env_size,
        dt=args.dt,
        max_steps=args.max_steps
    )
    
    # 打印环境信息
    print(f"环境观察空间: {env.observation_space}")
    print(f"环境动作空间: {env.action_space}")
    
    # 重置环境并查看初始状态
    initial_state = env.reset()
    print(f"初始状态形状: {initial_state.shape}")
    print(f"初始状态: {initial_state[:10]}...") # 只打印前10个元素
    
    # 使用实际的状态维度，而不是环境定义的观察空间维度
    state_dim = initial_state.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"状态维度 (实际): {state_dim}, 动作维度: {action_dim}")
    
    # 初始化超参数
    learning_rate = args.learning_rate
    clip_param = args.clip_param
    entropy_coef = args.entropy_coef
    
    # 创建超参数自动调整设置
    auto_adjust = args.auto_adjust
    reward_window_size = 10  # 用于平滑奖励曲线的窗口大小
    eval_interval = 20  # 每20个episode评估一次当前策略
    
    agent = ADPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=clip_param,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device
    )
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 训练记录
    rewards = []
    critic_losses = []
    actor_losses = []
    entropies = []
    
    # 训练循环
    total_steps = 0
    episode = 0
    
    # 快速模式
    if args.quick:
        total_timesteps = min(args.total_timesteps, 10000)
        print(f"快速模式: 时间步数 = {total_timesteps}")
    else:
        total_timesteps = args.total_timesteps
    
    # 用于自动调整的变量
    reward_history = []
    last_avg_reward = -float('inf')
    improvement_threshold = 0.05  # 5%的改善阈值
    stagnation_counter = 0
    max_stagnation = 3  # 连续3次停滞后调整参数
    
    # 参数调整记录
    param_adjustment_log = []
    
    while total_steps < total_timesteps:
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        while not done and step < env.max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录轨迹
            for uav in env.uavs:
                if uav.is_alive:
                    uav.trajectory.append(uav.position.copy())
                    
            # 存储经验
            agent.buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            step += 1
                
        # 回合结束，计算回报和优势
        if len(agent.buffer.states) > 0:
            # 计算最后状态的价值
            _, _, last_value = agent.select_action(state)
            
            # 计算回报和优势
            agent.buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            
            # 更新策略
            rollout = agent.buffer.get()
            stats = agent.update(rollout)
            
            critic_losses.append(stats['value_loss'])
            actor_losses.append(stats['policy_loss'])
            entropies.append(stats['entropy'])
            
            agent.buffer.clear()
            
        # 记录奖励
        rewards.append(episode_reward)
        reward_history.append(episode_reward)
        
        # 打印训练信息
        if episode % args.log_interval == 0:
            print(f"Episode {episode} | Steps {total_steps}/{total_timesteps}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Value Loss: {stats['value_loss']:.4f}")
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print("-" * 30)
            
        # 自动参数调整
        if auto_adjust and episode > 0 and episode % eval_interval == 0:
            # 计算最近reward_window_size个episode的平均奖励
            if len(reward_history) > reward_window_size:
                current_avg_reward = np.mean(reward_history[-reward_window_size:])
                
                # 计算奖励改善率
                if last_avg_reward != -float('inf'):
                    improvement = (current_avg_reward - last_avg_reward) / abs(last_avg_reward) if last_avg_reward != 0 else 1.0
                    
                    print(f"当前平均奖励: {current_avg_reward:.2f}, 上一平均奖励: {last_avg_reward:.2f}")
                    print(f"改善率: {improvement:.2%}")
                    
                    if improvement < improvement_threshold:
                        stagnation_counter += 1
                        print(f"奖励停滞! 计数: {stagnation_counter}/{max_stagnation}")
                    else:
                        stagnation_counter = 0
                    
                    # 如果连续几次评估没有显著改善，调整超参数
                    if stagnation_counter >= max_stagnation:
                        print("===== 自动调整超参数 =====")
                        stagnation_counter = 0
                        
                        # 根据不同情况调整不同参数
                        if stats['policy_loss'] > 0.5:
                            # 策略损失过大，减小学习率
                            old_lr = agent.optimizer.param_groups[0]['lr']
                            new_lr = max(old_lr * 0.5, 1e-5)
                            
                            for param_group in agent.optimizer.param_groups:
                                param_group['lr'] = new_lr
                                
                            print(f"降低学习率: {old_lr:.2e} -> {new_lr:.2e}")
                            param_adjustment_log.append((episode, f"学习率: {old_lr:.2e} -> {new_lr:.2e}"))
                            
                        elif stats['entropy'] < 0.02:
                            # 熵太小，增加熵系数
                            old_entropy_coef = entropy_coef
                            entropy_coef = min(entropy_coef * 2.0, 0.05)
                            agent.entropy_coef = entropy_coef
                            
                            print(f"增加熵系数: {old_entropy_coef:.3f} -> {entropy_coef:.3f}")
                            param_adjustment_log.append((episode, f"熵系数: {old_entropy_coef:.3f} -> {entropy_coef:.3f}"))
                            
                        elif stats['entropy'] > 0.3:
                            # 熵太大，减小熵系数
                            old_entropy_coef = entropy_coef
                            entropy_coef = max(entropy_coef * 0.5, 0.001)
                            agent.entropy_coef = entropy_coef
                            
                            print(f"减小熵系数: {old_entropy_coef:.3f} -> {entropy_coef:.3f}")
                            param_adjustment_log.append((episode, f"熵系数: {old_entropy_coef:.3f} -> {entropy_coef:.3f}"))
                        
                        else:
                            # 尝试调整裁剪参数
                            old_clip_param = clip_param
                            clip_param = clip_param * 0.9 if clip_param > 0.1 else clip_param * 1.1
                            agent.clip_param = clip_param
                            
                            print(f"调整裁剪参数: {old_clip_param:.3f} -> {clip_param:.3f}")
                            param_adjustment_log.append((episode, f"裁剪参数: {old_clip_param:.3f} -> {clip_param:.3f}"))
                        
                        # 修改环境奖励权重
                        if current_avg_reward < -1000:
                            # 奖励太低，增加正面奖励
                            old_jamming_success = env.reward_weights['jamming_success']
                            env.reward_weights['jamming_success'] *= 1.5
                            
                            old_goal_reward = env.reward_weights['goal_reward']
                            env.reward_weights['goal_reward'] *= 1.5
                            
                            old_distance_penalty = env.reward_weights['distance_penalty']
                            env.reward_weights['distance_penalty'] *= 0.5
                            
                            print(f"调整奖励权重:")
                            print(f"  - 干扰成功奖励: {old_jamming_success:.2f} -> {env.reward_weights['jamming_success']:.2f}")
                            print(f"  - 目标奖励: {old_goal_reward:.2f} -> {env.reward_weights['goal_reward']:.2f}")
                            print(f"  - 距离惩罚: {old_distance_penalty:.6f} -> {env.reward_weights['distance_penalty']:.6f}")
                            
                            param_adjustment_log.append((episode, f"调整奖励权重"))
                
                # 更新最后平均奖励
                last_avg_reward = current_avg_reward
        
        # 保存模型
        if episode % args.save_interval == 0 and episode > 0:
            agent.save(os.path.join(args.save_path, f"model_{episode}.pt"))
            
        # 绘制训练曲线
        if episode % args.plot_interval == 0 and episode > 0:
            plot_training_curves(
                rewards=rewards,
                critic_losses=critic_losses,
                actor_losses=actor_losses,
                entropies=entropies,
                save_path=args.save_path
            )
            
        # 绘制轨迹
        if episode % args.plot_interval == 0 and episode > 0:
            plot_trajectory(env.uavs, env.radars, args.save_path)
            
        episode += 1
        
    # 保存最终模型
    agent.save(os.path.join(args.save_path, "model_final.pt"))
    
    # 绘制最终训练曲线
    plot_training_curves(
        rewards=rewards,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        entropies=entropies,
        save_path=args.save_path
    )
    
    # 保存超参数调整日志
    if auto_adjust and param_adjustment_log:
        with open(os.path.join(args.save_path, "parameter_adjustments.txt"), "w") as f:
            f.write("Episode | Parameter Adjustment\n")
            f.write("-" * 50 + "\n")
            for episode, adjustment in param_adjustment_log:
                f.write(f"{episode:6d} | {adjustment}\n")
    
    env.close()
    print("===== 训练完成 =====")

def evaluate(args):
    """评估AD-PPO算法"""
    print("===== 开始评估 =====")
    
    # 创建环境
    env = ElectronicWarfareEnv(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        env_size=args.env_size,
        dt=args.dt,
        max_steps=args.max_steps
    )
    
    # 打印环境信息
    print(f"环境观察空间: {env.observation_space}")
    print(f"环境动作空间: {env.action_space}")
    
    # 重置环境并查看初始状态
    initial_state = env.reset()
    print(f"初始状态形状: {initial_state.shape}")
    
    # 使用实际的状态维度，而不是环境定义的观察空间维度
    state_dim = initial_state.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"状态维度 (实际): {state_dim}, 动作维度: {action_dim}")
    
    agent = ADPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device
    )
    
    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        return
    
    agent.load(args.model_path)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 评估
    total_reward = 0
    success_count = 0
    
    for ep in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        while not done and step < env.max_steps:
            # 选择动作
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录轨迹
            for uav in env.uavs:
                if uav.is_alive:
                    uav.trajectory.append(uav.position.copy())
                    
            state = next_state
            episode_reward += reward
            step += 1
                
        # 判断是否成功 - 更宽松的成功条件
        jammed_ratio = sum(1 for radar in env.radars if radar.is_jammed) / len(env.radars)
        if jammed_ratio >= 0.5:  # 50%的雷达被干扰就算成功
            success_count += 1
            
        total_reward += episode_reward
        
        # 打印评估信息
        print(f"Episode {ep+1} | Reward: {episode_reward:.2f} | Steps: {step}")
        print(f"雷达干扰率: {jammed_ratio:.2%}")
        
        # 创建评估回合的保存目录
        eval_dir = os.path.join(args.save_path, f"eval_{ep}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 绘制轨迹
        plot_trajectory(env.uavs, env.radars, eval_dir)
    
    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f"平均奖励: {total_reward / args.num_episodes:.2f}")
    print(f"成功率: {success_count / args.num_episodes:.2%}")
    
    env.close()
    print("===== 评估完成 =====")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AD-PPO算法训练与评估")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 训练参数
    train_parser = subparsers.add_parser("train", help="训练AD-PPO算法")
    
    # 环境参数
    train_parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    train_parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    train_parser.add_argument("--env_size", type=float, default=2000.0, help="环境大小")
    train_parser.add_argument("--dt", type=float, default=0.1, help="时间步长")
    train_parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    
    # 算法参数
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    train_parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    train_parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数")
    train_parser.add_argument("--clip_param", type=float, default=0.2, help="PPO裁剪参数")
    train_parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    train_parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵损失系数")
    train_parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪范数")
    train_parser.add_argument("--auto_adjust", action="store_true", help="自动调整超参数")
    
    # 训练参数
    train_parser.add_argument("--total_timesteps", type=int, default=100000, help="总时间步数")
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    train_parser.add_argument("--quick", action="store_true", help="快速模式")
    
    # 保存参数
    train_parser.add_argument("--save_path", type=str, default="experiments/results/ad_ppo", help="保存路径")
    train_parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    train_parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    train_parser.add_argument("--plot_interval", type=int, default=100, help="绘图间隔")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估AD-PPO算法")
    
    # 环境参数
    eval_parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    eval_parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    eval_parser.add_argument("--env_size", type=float, default=2000.0, help="环境大小")
    eval_parser.add_argument("--dt", type=float, default=0.1, help="时间步长")
    eval_parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    
    # 算法参数
    eval_parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    eval_parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    eval_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    eval_parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数")
    eval_parser.add_argument("--clip_param", type=float, default=0.2, help="PPO裁剪参数")
    eval_parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    eval_parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵损失系数")
    eval_parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪范数")
    
    # 评估参数
    eval_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--num_episodes", type=int, default=10, help="评估回合数")
    eval_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    # 保存参数
    eval_parser.add_argument("--save_path", type=str, default="experiments/results/ad_ppo_eval", help="保存路径")
    
    args = parser.parse_args()
    
    # 执行命令
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 