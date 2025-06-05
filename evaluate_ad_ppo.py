import os
import numpy as np
import torch
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.utils.plotting import plot_environment_state, plot_trajectory

def evaluate(args):
    """
    评估AD-PPO算法
    
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
    
    # 加载模型
    agent.load(args.model_path)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 评估记录
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    # 评估循环
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        while not done:
            # 选择动作
            action = agent.select_action(state, deterministic=True)[0]
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录轨迹
            for uav in env.uavs:
                if uav.is_alive:
                    uav.trajectory.append(uav.position.copy())
                    
            state = next_state
            episode_reward += reward
            step += 1
            
            # 可视化环境状态
            if args.render:
                plot_environment_state(env.uavs, env.radars, args.save_path, step)
                
        # 记录评估结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        success_rates.append(1.0 if info.get('success', False) else 0.0)
        
        # 打印评估信息
        print(f"Episode {episode}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {step}")
        print(f"Success: {info.get('success', False)}")
        print("-" * 50)
        
        # 绘制轨迹
        plot_trajectory(env.uavs, env.radars, args.save_path, episode)
        
    # 打印评估统计信息
    print("\n评估统计信息:")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"成功率: {np.mean(success_rates):.2%}")
    
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
    
    # 评估参数
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    
    # 保存参数
    parser.add_argument("--save_path", type=str, default="experiments/results/ad_ppo_eval")
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    evaluate(args) 