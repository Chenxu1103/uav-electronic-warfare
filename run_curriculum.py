#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG
from src.utils.plotting import plot_trajectory
from src.utils.buffer import RolloutBuffer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="课程学习训练电子对抗算法")
    
    # 训练参数
    parser.add_argument("--algorithm", type=str, default="ad_ppo", choices=["ad_ppo", "maddpg"], help="训练的算法")
    parser.add_argument("--episodes_per_stage", type=int, default=100, help="每个阶段的训练回合数")
    parser.add_argument("--eval_interval", type=int, default=20, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="每次评估的回合数")
    parser.add_argument("--save_interval", type=int, default=50, help="保存间隔")
    
    # 环境参数
    parser.add_argument("--num_uavs", type=int, default=3, help="UAV数量")
    parser.add_argument("--env_size", type=float, default=2000.0, help="环境大小")
    parser.add_argument("--dt", type=float, default=0.1, help="时间步长")
    parser.add_argument("--max_steps", type=int, default=200, help="每回合最大步数")
    
    # 算法参数
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵正则化系数")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪")
    parser.add_argument("--buffer_size", type=int, default=10000, help="MADDPG经验回放缓冲区大小")
    parser.add_argument("--batch_size", type=int, default=128, help="MADDPG批次大小")
    parser.add_argument("--tau", type=float, default=0.01, help="MADDPG目标网络软更新参数")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称")
    
    return parser.parse_args()

def evaluate_agent(agent, env, agent_type, num_episodes=5, save_path=None):
    """评估智能体性能"""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    total_reward = 0
    success_count = 0
    partial_success_count = 0
    avg_jamming_ratio = 0
    min_distances = []  # 记录最小距离
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        episode_min_distances = []
        
        while not done and step < env.max_steps:
            # 选择动作
            if agent_type == 'ad_ppo':
                action, _, _ = agent.select_action(state, deterministic=True)
            else:  # maddpg
                # 分割状态为每个智能体的状态
                state_dim = state.shape[0] // env.num_uavs
                agent_states = []
                for i in range(env.num_uavs):
                    agent_states.append(state[i*state_dim:(i+1)*state_dim])
                agent_states = np.array(agent_states)
                
                # 选择动作
                actions = agent.select_action(agent_states, add_noise=False)
                action = np.concatenate(actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录轨迹
            for uav in env.uavs:
                if uav.is_alive:
                    uav.trajectory.append(uav.position.copy())
            
            # 记录每步的最小距离
            step_min_distance = float('inf')
            for uav in env.uavs:
                if not uav.is_alive:
                    continue
                for radar in env.radars:
                    distance = np.linalg.norm(uav.position[:2] - radar.position[:2])
                    step_min_distance = min(step_min_distance, distance)
            episode_min_distances.append(step_min_distance)
                    
            state = next_state
            episode_reward += reward
            step += 1
                
        # 计算雷达干扰率
        jammed_radar_count = sum(1 for radar in env.radars if radar.is_jammed)
        jammed_ratio = jammed_radar_count / len(env.radars)
        avg_jamming_ratio += jammed_ratio
        
        # 打印详细的雷达状态信息
        print("\n===== 雷达状态详情 =====")
        for i, radar in enumerate(env.radars):
            print(f"雷达 {i+1}: 位置 {radar.position}, 干扰状态: {'已干扰' if radar.is_jammed else '未干扰'}, 干扰功率: {radar.jamming_power:.4f}")
            
        # 打印UAV状态信息
        print("\n===== UAV状态详情 =====")
        for i, uav in enumerate(env.uavs):
            if uav.is_alive:
                min_distance = float('inf')
                closest_radar = -1
                for j, radar in enumerate(env.radars):
                    distance = np.linalg.norm(uav.position[:2] - radar.position[:2])
                    if distance < min_distance:
                        min_distance = distance
                        closest_radar = j
                print(f"UAV {i+1}: 位置 {uav.position}, 能量: {uav.energy:.2f}, 干扰状态: {'开启' if uav.is_jamming else '关闭'}, 与雷达{closest_radar+1}最近, 距离: {min_distance:.2f}")
            else:
                print(f"UAV {i+1}: 已失效")
        
        # 修改成功标准：将50%以上雷达被干扰也计算为成功
        if jammed_ratio >= 0.5:  # 半数以上雷达被干扰就认为成功
            success_count += 1
            
        # 记录部分成功（至少有一个雷达被干扰）
        if jammed_ratio > 0:
            partial_success_count += 1
        
        # 记录最小距离
        if episode_min_distances:
            min_distances.append(min(episode_min_distances))
            
        total_reward += episode_reward
        print(f"\nEpisode {ep+1} | Reward: {episode_reward:.2f} | Steps: {step}")
        print(f"雷达干扰率: {jammed_ratio:.2%}")
        print(f"本回合最小距离: {min(episode_min_distances) if episode_min_distances else 'N/A'}")
        
        # 如果有保存路径，绘制最后一个回合的轨迹
        if save_path and ep == num_episodes - 1:
            try:
                plot_trajectory(env.uavs, env.radars, save_path)
            except:
                print("无法绘制轨迹，跳过图像生成")
            
    # 计算平均奖励和成功率
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes * 100
    partial_success_rate = partial_success_count / num_episodes * 100
    avg_jamming_ratio = avg_jamming_ratio / num_episodes * 100
    avg_min_distance = np.mean(min_distances) if min_distances else float('inf')
    
    print(f"\n平均奖励: {avg_reward:.2f}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"部分成功率(>0%): {partial_success_rate:.2f}%")
    print(f"平均雷达干扰率: {avg_jamming_ratio:.2f}%")
    print(f"平均最小距离: {avg_min_distance:.2f}")
    
    return avg_reward, success_rate, avg_jamming_ratio, avg_min_distance

def train_stage(agent, env, stage_config, args, save_path, start_episode=0):
    """训练指定阶段"""
    print(f"===== 开始训练阶段 {stage_config['stage']} =====")
    print(f"阶段设置: {stage_config}")
    
    # 训练数据记录
    training_data = {
        'algorithm': args.algorithm,
        'stage': stage_config['stage'],
        'episodes': [],
        'rewards': [],
        'eval_episodes': [],
        'eval_rewards': [],
        'success_rates': [],
        'jamming_ratios': [],
        'min_distances': []
    }
    
    if args.algorithm == 'ad_ppo':
        training_function = train_ad_ppo_stage
    else:
        training_function = train_maddpg_stage
        
    return training_function(agent, env, stage_config, args, save_path, training_data, start_episode)

def train_ad_ppo_stage(agent, env, stage_config, args, save_path, training_data, start_episode=0):
    """训练AD-PPO算法指定阶段"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for episode in range(start_episode, args.episodes_per_stage):
        # 重置环境并获取初始状态
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 保存单个回合的经验
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []
        episode_values = []
        step = 0
        
        # 收集单个回合的经验
        while not done and step < env.max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 保存经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            step += 1
            
        # 计算last_value（用于GAE计算）
        _, _, last_value = agent.select_action(state)
        
        # 创建RolloutBuffer并添加经验
        buffer = RolloutBuffer()
        for i in range(len(episode_states)):
            buffer.add(
                episode_states[i],
                episode_actions[i],
                episode_rewards[i],
                episode_states[i+1] if i < len(episode_states) - 1 else state,  # next_state
                episode_dones[i],
                episode_log_probs[i],
                episode_values[i]
            )
        
        # 计算优势和回报
        buffer.compute_returns_and_advantages(last_value, args.gamma, args.gae_lambda)
        
        # 获取经验数据
        rollout = buffer.get()
        
        # 更新策略
        update_info = agent.update(rollout)
        
        # 记录训练数据
        training_data['episodes'].append(episode)
        training_data['rewards'].append(episode_reward)
        
        # 打印训练信息
        policy_loss = update_info.get('policy_loss', 0)
        value_loss = update_info.get('value_loss', 0)
        entropy = update_info.get('entropy', 0)
        
        print(f"Stage {stage_config['stage']} | Episode {episode} | Reward: {episode_reward:.2f}")
        print(f"Value Loss: {value_loss:.4f}")
        print(f"Policy Loss: {policy_loss:.4f}")
        print(f"Entropy: {entropy:.4f}")
        print("-" * 30)
        
        # 评估模型
        if episode % args.eval_interval == 0 or episode == args.episodes_per_stage - 1:
            eval_save_path = os.path.join(save_path, f"eval_stage{stage_config['stage']}_{episode}")
            os.makedirs(eval_save_path, exist_ok=True)
            
            avg_reward, success_rate, jamming_ratio, min_distance = evaluate_agent(
                agent=agent, 
                env=env,
                agent_type='ad_ppo',
                num_episodes=args.eval_episodes,
                save_path=eval_save_path
            )
            
            # 记录评估数据
            training_data['eval_episodes'].append(episode)
            training_data['eval_rewards'].append(avg_reward)
            training_data['success_rates'].append(success_rate)
            training_data['jamming_ratios'].append(jamming_ratio)
            training_data['min_distances'].append(min_distance)
            
            # 绘制训练进度
            plot_training_progress(training_data, save_path)
        
        # 保存模型检查点
        if episode % args.save_interval == 0 or episode == args.episodes_per_stage - 1:
            model_path = os.path.join(save_path, f"stage{stage_config['stage']}_model_{episode}.pt")
            agent.save(model_path)
            print(f"模型已保存到 {model_path}")
    
    # 保存阶段最终模型
    final_model_path = os.path.join(save_path, f"stage{stage_config['stage']}_model_final.pt")
    agent.save(final_model_path)
    print(f"阶段{stage_config['stage']}最终模型已保存到 {final_model_path}")
    
    return agent, training_data

def train_maddpg_stage(agent, env, stage_config, args, save_path, training_data, start_episode=0):
    """训练MADDPG算法指定阶段"""
    state_dim_per_agent = env.observation_space.shape[0] // args.num_uavs
    action_dim_per_agent = env.action_space.shape[0] // args.num_uavs
    
    for episode in range(start_episode, args.episodes_per_stage):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 分割状态为每个智能体的状态
        states = []
        for i in range(args.num_uavs):
            states.append(state[i*state_dim_per_agent:(i+1)*state_dim_per_agent])
        states = np.array(states)
        
        while not done:
            # 选择动作
            actions = agent.select_action(states)
            action = np.concatenate(actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 分割下一个状态
            next_states = []
            for i in range(args.num_uavs):
                next_states.append(next_state[i*state_dim_per_agent:(i+1)*state_dim_per_agent])
            next_states = np.array(next_states)
            
            # 存储转换
            agent.store_transition(states, actions, reward, next_states, done)
            
            # 更新智能体
            if len(agent.replay_buffer) > args.batch_size:
                critic_loss, actor_loss = agent.update()
            
            states = next_states
            episode_reward += reward
        
        # 记录训练数据
        training_data['episodes'].append(episode)
        training_data['rewards'].append(episode_reward)
        
        # 打印训练信息
        critic_loss, actor_loss = agent.get_statistics()
        print(f"Stage {stage_config['stage']} | Episode {episode} | Reward: {episode_reward:.2f}")
        print(f"Critic Loss: {critic_loss:.4f}")
        print(f"Actor Loss: {actor_loss:.4f}")
        print("-" * 30)
        
        # 评估模型
        if episode % args.eval_interval == 0 or episode == args.episodes_per_stage - 1:
            eval_save_path = os.path.join(save_path, f"eval_stage{stage_config['stage']}_{episode}")
            os.makedirs(eval_save_path, exist_ok=True)
            
            avg_reward, success_rate, jamming_ratio, min_distance = evaluate_agent(
                agent=agent, 
                env=env,
                agent_type='maddpg',
                num_episodes=args.eval_episodes,
                save_path=eval_save_path
            )
            
            # 记录评估数据
            training_data['eval_episodes'].append(episode)
            training_data['eval_rewards'].append(avg_reward)
            training_data['success_rates'].append(success_rate)
            training_data['jamming_ratios'].append(jamming_ratio)
            training_data['min_distances'].append(min_distance)
            
            # 绘制训练进度
            plot_training_progress(training_data, save_path)
        
        # 保存模型检查点
        if episode % args.save_interval == 0 or episode == args.episodes_per_stage - 1:
            model_dir = os.path.join(save_path, f"stage{stage_config['stage']}_model_{episode}")
            os.makedirs(model_dir, exist_ok=True)
            agent.save(model_dir)
            print(f"模型已保存到 {model_dir}")
    
    # 保存阶段最终模型
    final_model_dir = os.path.join(save_path, f"stage{stage_config['stage']}_model_final")
    os.makedirs(final_model_dir, exist_ok=True)
    agent.save(final_model_dir)
    print(f"阶段{stage_config['stage']}最终模型已保存到 {final_model_dir}")
    
    return agent, training_data

def plot_training_progress(training_data, save_path):
    """绘制训练进度"""
    plt.figure(figsize=(15, 15))
    
    # 训练奖励
    plt.subplot(4, 1, 1)
    plt.plot(training_data['episodes'], training_data['rewards'], 'b-')
    plt.title(f"Training Rewards - {training_data['algorithm']} - Stage {training_data['stage']}")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # 评估奖励
    plt.subplot(4, 1, 2)
    plt.plot(training_data['eval_episodes'], training_data['eval_rewards'], 'r-o')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # 成功率
    plt.subplot(4, 1, 3)
    plt.plot(training_data['eval_episodes'], training_data['success_rates'], 'g-o')
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    # 最小距离
    plt.subplot(4, 1, 4)
    plt.plot(training_data['eval_episodes'], training_data['min_distances'], 'm-o')
    plt.title('Average Minimum Distance')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'stage{training_data["stage"]}_progress.png'))
    plt.close()

def create_curriculum_stages():
    """创建课程学习阶段设置"""
    stages = [
        {
            'stage': 1,
            'num_radars': 1,  # 第一阶段：只有一个雷达
            'env_size': 1000.0,  # 较小的环境
            'target_success_rate': 50.0,  # 目标成功率
            'success_threshold': 0.5,  # 干扰率阈值
            'description': '一个雷达，小环境，基础训练'
        },
        {
            'stage': 2,
            'num_radars': 1,
            'env_size': 2000.0,  # 扩大环境
            'target_success_rate': 50.0,
            'success_threshold': 0.5,
            'description': '一个雷达，大环境，提高导航能力'
        },
        {
            'stage': 3,
            'num_radars': 2,  # 两个雷达
            'env_size': 2000.0,
            'target_success_rate': 30.0,
            'success_threshold': 0.5,  # 半数雷达被干扰
            'description': '两个雷达，多目标干扰训练'
        },
        {
            'stage': 4,
            'num_radars': 3,  # 三个雷达
            'env_size': 2000.0,
            'target_success_rate': 20.0,
            'success_threshold': 0.5,  # 半数雷达被干扰
            'description': '三个雷达，复杂多目标干扰训练'
        }
    ]
    return stages

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name if args.exp_name else f"curriculum_{args.algorithm}_{timestamp}"
    save_path = os.path.join("experiments", "curriculum", exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(save_path, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    # 创建课程学习阶段
    stages = create_curriculum_stages()
    
    # 初始化环境
    env = None
    all_training_data = []
    
    # 逐阶段训练
    for stage_config in stages:
        print(f"\n\n===== 开始阶段 {stage_config['stage']}: {stage_config['description']} =====\n")
        
        # 创建新环境
        env = ElectronicWarfareEnv(
            num_uavs=args.num_uavs,
            num_radars=stage_config['num_radars'],
            env_size=stage_config['env_size'],
            dt=args.dt,
            max_steps=args.max_steps
        )
        
        # 获取实际状态维度
        initial_state = env.reset()
        state_dim = initial_state.shape[0]
        action_dim = env.action_space.shape[0]
        
        # 为每个阶段创建新的智能体
        if args.algorithm == "ad_ppo":
            print(f"阶段 {stage_config['stage']} 状态维度: {state_dim}, 动作维度: {action_dim}")
            
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
        else:
            state_dim_per_agent = state_dim // args.num_uavs
            action_dim_per_agent = action_dim // args.num_uavs
            print(f"阶段 {stage_config['stage']} 每个智能体状态维度: {state_dim_per_agent}, 动作维度: {action_dim_per_agent}")
            
            agent = MADDPG(
                num_agents=args.num_uavs,
                state_dim=state_dim_per_agent,
                action_dim=action_dim_per_agent,
                hidden_dim=args.hidden_dim,
                lr=args.learning_rate,
                gamma=args.gamma,
                tau=args.tau,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                device=args.device
            )
        
        # 为该阶段创建保存目录
        stage_save_path = os.path.join(save_path, f"stage{stage_config['stage']}")
        os.makedirs(stage_save_path, exist_ok=True)
        
        # 训练当前阶段
        agent, stage_training_data = train_stage(agent, env, stage_config, args, stage_save_path)
        all_training_data.append(stage_training_data)
        
        # 评估当前阶段的表现
        final_eval_path = os.path.join(stage_save_path, "final_evaluation")
        avg_reward, success_rate, jamming_ratio, min_distance = evaluate_agent(
            agent=agent,
            env=env,
            agent_type=args.algorithm,
            num_episodes=10,  # 更多回合进行最终评估
            save_path=final_eval_path
        )
        
        # 保存阶段结果
        stage_results = {
            'stage': stage_config['stage'],
            'description': stage_config['description'],
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'jamming_ratio': jamming_ratio,
            'min_distance': min_distance,
            'target_success_rate': stage_config['target_success_rate']
        }
        
        with open(os.path.join(stage_save_path, "results.txt"), "w") as f:
            for key, value in stage_results.items():
                f.write(f"{key}: {value}\n")
        
        # 检查是否达到目标成功率
        if success_rate >= stage_config['target_success_rate']:
            print(f"阶段 {stage_config['stage']} 成功! 达到目标成功率: {success_rate:.2f}% >= {stage_config['target_success_rate']}%")
        else:
            print(f"阶段 {stage_config['stage']} 未达到目标成功率: {success_rate:.2f}% < {stage_config['target_success_rate']}%")
            print("继续下一阶段训练...")
    
    print(f"===== 课程学习训练完成，结果保存在 {save_path} =====")

if __name__ == "__main__":
    main() 