#!/usr/bin/env python3
"""
AD-PPO 训练脚本
此脚本专门用于训练动作依赖的近端策略优化(AD-PPO)算法
"""

import os
import sys
import time
import numpy as np
import torch
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.utils.plotting import plot_training_curves, plot_environment_state, plot_trajectory

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.insert(0, project_root)
print(f"添加项目根目录到Python路径: {project_root}")

# 导入项目模块
from src.models import ECMEnvironment
from src.algorithms.ad_ppo import ADPPO
from src.utils import plot_training_curves

def train(args):
    """
    训练AD-PPO算法
    
    Args:
        args: 命令行参数
    """
    # 创建环境
    env = ElectronicWarfareEnv(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        env_size=args.env_size,
        dt=args.dt,
        max_steps=args.max_steps
    )
    
    # 创建算法
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
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
    
    while total_steps < args.total_timesteps:
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        while not done:
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
            
            # 可视化环境状态
            if args.render and total_steps % args.render_interval == 0:
                plot_environment_state(env.uavs, env.radars, args.save_path, total_steps)
                
        # 更新策略
        if len(agent.buffer.states) >= args.batch_size:
            rollout = agent.buffer.get()
            stats = agent.update(rollout)
            
            critic_losses.append(stats['value_loss'])
            actor_losses.append(stats['policy_loss'])
            entropies.append(stats['entropy'])
            
            agent.buffer.clear()
            
        # 记录奖励
        rewards.append(episode_reward)
        
        # 打印训练信息
        if episode % args.log_interval == 0:
            print(f"Episode {episode}")
            print(f"Total Steps: {total_steps}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward: {np.mean(rewards[-args.log_interval:]):.2f}")
            print(f"Critic Loss: {stats['value_loss']:.4f}")
            print(f"Actor Loss: {stats['policy_loss']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print("-" * 50)
            
        # 保存模型
        if episode % args.save_interval == 0:
            agent.save(os.path.join(args.save_path, f"ad_ppo_{episode}.pt"))
            
        # 绘制训练曲线
        if episode % args.plot_interval == 0:
            plot_training_curves(
                rewards=rewards,
                critic_losses=critic_losses,
                actor_losses=actor_losses,
                entropies=entropies,
                save_path=args.save_path
            )
            
        # 绘制轨迹
        if episode % args.plot_interval == 0:
            plot_trajectory(env.uavs, env.radars, args.save_path)
            
        episode += 1
        
    # 保存最终模型
    agent.save(os.path.join(args.save_path, "ad_ppo_final.pt"))
    
    # 绘制最终训练曲线
    plot_training_curves(
        rewards=rewards,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        entropies=entropies,
        save_path=args.save_path
    )
    
    # 绘制最终轨迹
    plot_trajectory(env.uavs, env.radars, args.save_path)
    
    env.close()
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # 环境参数
    parser.add_argument("--num_uavs", type=int, default=3)
    parser.add_argument("--num_radars", type=int, default=2)
    parser.add_argument("--env_size", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=200)
    
    # 算法参数
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    
    # 日志和保存参数
    parser.add_argument("--save_path", type=str, default="experiments/results/ad_ppo")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--plot_interval", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_interval", type=int, default=10)
    
    args = parser.parse_args()
    train(args) 