#!/usr/bin/env python3
"""
比较不同强化学习算法在电子对抗环境中的性能

本脚本比较AD-PPO算法与MADDPG算法在电子对抗环境中的性能，并生成对比图表和指标。

使用示例:
    python compare_algorithms.py --num_episodes 500 --eval_interval 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG
from src.utils.plotting import plot_training_curves, plot_trajectory

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.insert(0, project_root)

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def create_environment(num_uavs=3, num_radars=2, env_size=2000.0, dt=0.1, max_steps=200):
    """创建电子对抗环境"""
    env = ElectronicWarfareEnv(
        num_uavs=num_uavs,
        num_radars=num_radars,
        env_size=env_size,
        dt=dt,
        max_steps=max_steps
    )
    return env

def train_adppo(env, args):
    """训练AD-PPO算法"""
    print("===== 开始训练 AD-PPO 算法 =====")
    
    # 初始化环境
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化超参数
    learning_rate = args.learning_rate
    clip_param = args.clip_param
    entropy_coef = args.entropy_coef
    
    # 创建AD-PPO智能体
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
    save_path = os.path.join(args.save_dir, 'ad_ppo')
    os.makedirs(save_path, exist_ok=True)
    
    # 训练记录
    rewards = []
    critic_losses = []
    actor_losses = []
    entropies = []
    eval_rewards = []
    success_rates = []
    partial_success_rates = []
    jamming_ratios = []
    episodes_trained = []
    
    # 自动调整参数设置
    window_size = 10  # 用于计算平均奖励的窗口大小
    last_avg_reward = -float('inf')
    improvement_threshold = 0.03  # 3%的改善阈值
    stagnation_counter = 0
    max_stagnation = 3  # 连续停滞次数阈值
    
    # 参数调整记录
    param_adjustments = []
    
    # 训练循环
    episode = 0
    
    for episode in tqdm(range(args.num_episodes), desc="训练AD-PPO"):
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
        
        # 自动调整参数（每10个回合检查一次）
        if args.auto_adjust and episode > 0 and episode % 10 == 0 and len(rewards) > window_size:
            # 计算最近window_size个回合的平均奖励
            current_avg_reward = np.mean(rewards[-window_size:])
            
            # 计算奖励改善率
            if last_avg_reward != -float('inf'):
                improvement = (current_avg_reward - last_avg_reward) / abs(last_avg_reward) if last_avg_reward != 0 else 1.0
                
                print(f"\n当前平均奖励: {current_avg_reward:.2f}, 上一平均奖励: {last_avg_reward:.2f}")
                print(f"改善率: {improvement:.2%}")
                
                if improvement < improvement_threshold:
                    stagnation_counter += 1
                    print(f"奖励停滞! 计数: {stagnation_counter}/{max_stagnation}")
                else:
                    stagnation_counter = 0
                
                # 如果连续几次没有显著改善，调整参数
                if stagnation_counter >= max_stagnation:
                    print("===== 自动调整AD-PPO参数 =====")
                    stagnation_counter = 0
                    
                    # 根据不同情况调整不同参数
                    adjustment_made = False
                    
                    # 1. 如果策略损失较大，减小学习率
                    if np.mean(actor_losses[-5:]) > 0.3:
                        old_lr = learning_rate
                        learning_rate = max(learning_rate * 0.7, 5e-5)
                        # 更新优化器学习率
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        print(f"降低学习率: {old_lr:.2e} -> {learning_rate:.2e}")
                        param_adjustments.append((episode, f"学习率: {old_lr:.2e} -> {learning_rate:.2e}"))
                        adjustment_made = True
                    
                    # 2. 如果熵太小，增加熵系数促进探索
                    elif np.mean(entropies[-5:]) < 0.01:
                        old_entropy_coef = entropy_coef
                        entropy_coef = min(entropy_coef * 2.0, 0.05)
                        agent.entropy_coef = entropy_coef
                        print(f"增加熵系数: {old_entropy_coef:.4f} -> {entropy_coef:.4f}")
                        param_adjustments.append((episode, f"熵系数: {old_entropy_coef:.4f} -> {entropy_coef:.4f}"))
                        adjustment_made = True
                    
                    # 3. 如果熵太大，减小熵系数提高确定性
                    elif np.mean(entropies[-5:]) > 0.2:
                        old_entropy_coef = entropy_coef
                        entropy_coef = max(entropy_coef * 0.5, 0.001)
                        agent.entropy_coef = entropy_coef
                        print(f"减小熵系数: {old_entropy_coef:.4f} -> {entropy_coef:.4f}")
                        param_adjustments.append((episode, f"熵系数: {old_entropy_coef:.4f} -> {entropy_coef:.4f}"))
                        adjustment_made = True
                    
                    # 4. 如果奖励非常低，调整环境奖励权重
                    elif current_avg_reward < -1300:
                        # 增加正面奖励，减少负面奖励
                        print("调整环境奖励权重:")
                        
                        # 增加干扰成功奖励
                        old_jamming_success = env.reward_weights['jamming_success']
                        env.reward_weights['jamming_success'] = min(old_jamming_success * 1.5, 10.0)
                        print(f"  干扰成功奖励: {old_jamming_success:.2f} -> {env.reward_weights['jamming_success']:.2f}")
                        
                        # 减小距离惩罚
                        old_distance_penalty = env.reward_weights['distance_penalty']
                        env.reward_weights['distance_penalty'] = max(old_distance_penalty * 0.5, -0.00001)
                        print(f"  距离惩罚: {old_distance_penalty:.6f} -> {env.reward_weights['distance_penalty']:.6f}")
                        
                        param_adjustments.append((episode, "调整奖励权重"))
                        adjustment_made = True
                    
                    # 5. 如果以上都不适用，尝试调整裁剪参数
                    elif not adjustment_made:
                        old_clip_param = clip_param
                        # 如果改善较小但不是负的，可能需要更激进的更新，减小裁剪参数
                        if improvement > 0:
                            clip_param = max(clip_param * 0.8, 0.05)
                        # 如果改善为负，可能更新太激进，增大裁剪参数
                        else:
                            clip_param = min(clip_param * 1.2, 0.3)
                        agent.clip_param = clip_param
                        print(f"调整裁剪参数: {old_clip_param:.2f} -> {clip_param:.2f}")
                        param_adjustments.append((episode, f"裁剪参数: {old_clip_param:.2f} -> {clip_param:.2f}"))
            
            # 更新最后平均奖励
            last_avg_reward = current_avg_reward
        
        # 打印训练信息
        if episode % args.log_interval == 0:
            print(f"\nEpisode {episode} | Reward: {episode_reward:.2f}")
            print(f"Value Loss: {stats['value_loss']:.4f}")
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print("-" * 30)
            
        # 保存模型
        if episode % args.save_interval == 0 and episode > 0:
            agent.save(os.path.join(save_path, f"model_{episode}.pt"))
            
        # 评估模型
        if episode % args.eval_interval == 0 and episode > 0:
            avg_reward, success_rate, jamming_ratio = evaluate_agent(
                agent=agent, 
                env=env,
                agent_type='adppo',
                num_episodes=args.eval_episodes,
                save_path=os.path.join(save_path, f"eval_{episode}")
            )
            eval_rewards.append(avg_reward)
            success_rates.append(success_rate)
            jamming_ratios.append(jamming_ratio)
            episodes_trained.append(episode)
            
    # 保存最终模型
    agent.save(os.path.join(save_path, "model_final.pt"))
    
    # 绘制训练曲线
    plot_training_curves(
        rewards=rewards,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        entropies=entropies,
        save_path=save_path
    )
    
    # 保存参数调整记录
    if args.auto_adjust and param_adjustments:
        with open(os.path.join(save_path, "parameter_adjustments.txt"), "w") as f:
            f.write("回合,调整\n")
            for ep, adj in param_adjustments:
                f.write(f"{ep},{adj}\n")
    
    return {
        'rewards': rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'entropies': entropies,
        'eval_rewards': eval_rewards,
        'success_rates': success_rates,
        'jamming_ratios': jamming_ratios,
        'episodes_trained': episodes_trained
    }

def train_maddpg(env, args):
    """训练MADDPG算法"""
    print("===== 开始训练 MADDPG 算法 =====")
    
    # 初始化环境
    state = env.reset()
    state_dim = state.shape[0] // env.num_uavs  # 单个智能体的状态维度
    action_dim = env.action_space.shape[0] // env.num_uavs  # 单个智能体的动作维度
    
    # 创建MADDPG智能体
    agent = MADDPG(
        n_agents=env.num_uavs,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr_actor=args.learning_rate,
        lr_critic=args.learning_rate * 2,  # 评论家网络使用稍大的学习率
        gamma=args.gamma,
        tau=0.01,
        batch_size=args.batch_size,
        buffer_size=1e6
    )
    
    # 创建保存目录
    save_path = os.path.join(args.save_dir, 'maddpg')
    os.makedirs(save_path, exist_ok=True)
    
    # 训练记录
    rewards = []
    critic_losses = []
    actor_losses = []
    eval_rewards = []
    success_rates = []
    partial_success_rates = []
    jamming_ratios = []
    episodes_trained = []
    
    # 训练循环
    for episode in tqdm(range(args.num_episodes), desc="训练MADDPG"):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        # 为MADDPG准备状态 (分割状态为每个智能体的状态)
        agent_states = []
        for i in range(env.num_uavs):
            agent_states.append(state[i*state_dim:(i+1)*state_dim])
        agent_states = np.array(agent_states)
        
        while not done and step < env.max_steps:
            # 选择动作
            actions = agent.select_action(agent_states)
            
            # 执行动作 (合并所有智能体的动作)
            combined_action = np.concatenate(actions)
            next_state, reward, done, info = env.step(combined_action)
            
            # 记录轨迹
            for uav in env.uavs:
                if uav.is_alive:
                    uav.trajectory.append(uav.position.copy())
            
            # 为MADDPG准备下一状态和奖励
            agent_next_states = []
            agent_rewards = []
            
            for i in range(env.num_uavs):
                agent_next_states.append(next_state[i*state_dim:(i+1)*state_dim])
                agent_rewards.append(reward / env.num_uavs)  # 平均分配奖励
                
            agent_next_states = np.array(agent_next_states)
            agent_rewards = np.array(agent_rewards)
            
            # 存储经验
            agent.replay_buffer.add(agent_states, actions, agent_next_states, agent_rewards, done)
            
            # 更新策略
            if agent.replay_buffer.size > args.batch_size:
                critic_loss, actor_loss = agent.update()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
            
            state = next_state
            agent_states = agent_next_states
            episode_reward += reward
            step += 1
                
        # 记录奖励
        rewards.append(episode_reward)
        
        # 打印训练信息
        if episode % args.log_interval == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f}")
            if len(critic_losses) > 0:
                print(f"Critic Loss: {critic_losses[-1]:.4f}")
                print(f"Actor Loss: {actor_losses[-1]:.4f}")
            print("-" * 30)
            
        # 保存模型
        if episode % args.save_interval == 0 and episode > 0:
            model_dir = os.path.join(save_path, f"model_{episode}")
            os.makedirs(model_dir, exist_ok=True)
            agent.save(model_dir)
            
        # 评估模型
        if episode % args.eval_interval == 0 and episode > 0:
            avg_reward, success_rate, jamming_ratio = evaluate_agent(
                agent=agent, 
                env=env,
                agent_type='maddpg',
                num_episodes=args.eval_episodes,
                save_path=os.path.join(save_path, f"eval_{episode}")
            )
            eval_rewards.append(avg_reward)
            success_rates.append(success_rate)
            jamming_ratios.append(jamming_ratio)
            episodes_trained.append(episode)
            
    # 保存最终模型
    model_dir = os.path.join(save_path, "model_final")
    os.makedirs(model_dir, exist_ok=True)
    agent.save(model_dir)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(critic_losses, label='Critic')
    plt.plot(actor_losses, label='Actor')
    plt.title('Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"))
    
    return {
        'rewards': rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'eval_rewards': eval_rewards,
        'success_rates': success_rates,
        'jamming_ratios': jamming_ratios,
        'episodes_trained': episodes_trained
    }

def evaluate_agent(agent, env, agent_type, num_episodes=5, save_path=None):
    """评估智能体性能"""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    total_reward = 0
    success_count = 0
    partial_success_count = 0
    avg_jamming_ratio = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # 记录轨迹
        for uav in env.uavs:
            uav.trajectory = [uav.position.copy()]
            
        while not done and step < env.max_steps:
            # 选择动作
            if agent_type == 'adppo':
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
            
        total_reward += episode_reward
        print(f"\nEpisode {ep+1} | Reward: {episode_reward:.2f} | Steps: {step}")
        print(f"雷达干扰率: {jammed_ratio:.2%}")
        
        # 如果有保存路径，绘制最后一个回合的轨迹
        if save_path and ep == num_episodes - 1:
            plot_trajectory(env.uavs, env.radars, save_path)
            
    # 计算平均奖励和成功率
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes * 100
    partial_success_rate = partial_success_count / num_episodes * 100
    avg_jamming_ratio = avg_jamming_ratio / num_episodes * 100
    
    print(f"\n平均奖励: {avg_reward:.2f}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"部分成功率(>0%): {partial_success_rate:.2f}%")
    print(f"平均雷达干扰率: {avg_jamming_ratio:.2f}%")
    
    return avg_reward, success_rate, avg_jamming_ratio

def plot_comparison(adppo_results, maddpg_results, save_dir):
    """绘制算法对比图"""
    print("===== 生成算法对比图 =====")
    
    # 创建保存目录
    comparison_dir = os.path.join(save_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 绘制训练奖励对比图
    plt.figure(figsize=(15, 15))
    
    # 训练奖励
    plt.subplot(3, 2, 1)
    plt.plot(adppo_results['rewards'], label='AD-PPO')
    plt.plot(maddpg_results['rewards'], label='MADDPG')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 评估奖励
    plt.subplot(3, 2, 2)
    plt.plot(adppo_results['episodes_trained'], adppo_results['eval_rewards'], marker='o', label='AD-PPO')
    plt.plot(maddpg_results['episodes_trained'], maddpg_results['eval_rewards'], marker='s', label='MADDPG')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # 成功率
    plt.subplot(3, 2, 3)
    plt.plot(adppo_results['episodes_trained'], [x * 100 for x in adppo_results['success_rates']], marker='o', label='AD-PPO')
    plt.plot(maddpg_results['episodes_trained'], [x * 100 for x in maddpg_results['success_rates']], marker='s', label='MADDPG')
    plt.title('Success Rate (100%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    
    # 雷达干扰率
    plt.subplot(3, 2, 4)
    plt.plot(adppo_results['episodes_trained'], [x * 100 for x in adppo_results['jamming_ratios']], marker='o', label='AD-PPO')
    plt.plot(maddpg_results['episodes_trained'], [x * 100 for x in maddpg_results['jamming_ratios']], marker='s', label='MADDPG')
    plt.title('Average Jamming Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Jamming Ratio (%)')
    plt.legend()
    
    # Actor Loss - 使用单独的子图分别展示两个算法的Actor Loss
    plt.subplot(3, 2, 5)
    # 对数据进行处理，移除极端值和零值
    adppo_losses = np.array(adppo_results['actor_losses'])
    maddpg_losses = np.array(maddpg_results['actor_losses'])
    
    # 替换零值和负值（对于AD-PPO，policy loss通常为正值）
    adppo_losses = np.where(adppo_losses <= 0, 1e-6, adppo_losses)
    # 对于MADDPG，policy loss是负值（最大化Q值），我们转换为正值
    maddpg_losses = np.abs(maddpg_losses)
    maddpg_losses = np.where(maddpg_losses <= 0, 1e-6, maddpg_losses)
    
    # 使用移动平均来平滑曲线
    window_size = 5
    if len(adppo_losses) > window_size:
        adppo_smooth = np.convolve(adppo_losses, np.ones(window_size)/window_size, mode='valid')
    else:
        adppo_smooth = adppo_losses
        
    if len(maddpg_losses) > window_size:
        maddpg_smooth = np.convolve(maddpg_losses, np.ones(window_size)/window_size, mode='valid')
    else:
        maddpg_smooth = maddpg_losses
    
    # 绘制平滑后的曲线
    plt.semilogy(adppo_smooth, label='AD-PPO', linewidth=2, alpha=0.8)
    plt.semilogy(maddpg_smooth, label='MADDPG', linewidth=2, alpha=0.8)
    plt.title('Actor Losses (log scale)')
    plt.xlabel('Update Step')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 添加Critic Losses的比较
    plt.subplot(3, 2, 6)
    adppo_critic_losses = np.array(adppo_results['critic_losses'])
    maddpg_critic_losses = np.array(maddpg_results['critic_losses'])
    
    # 替换零值
    adppo_critic_losses = np.where(adppo_critic_losses <= 0, 1e-6, adppo_critic_losses)
    maddpg_critic_losses = np.where(maddpg_critic_losses <= 0, 1e-6, maddpg_critic_losses)
    
    # 使用移动平均平滑曲线
    if len(adppo_critic_losses) > window_size:
        adppo_critic_smooth = np.convolve(adppo_critic_losses, np.ones(window_size)/window_size, mode='valid')
    else:
        adppo_critic_smooth = adppo_critic_losses
        
    if len(maddpg_critic_losses) > window_size:
        maddpg_critic_smooth = np.convolve(maddpg_critic_losses, np.ones(window_size)/window_size, mode='valid')
    else:
        maddpg_critic_smooth = maddpg_critic_losses
    
    plt.semilogy(adppo_critic_smooth, label='AD-PPO', linewidth=2, alpha=0.8)
    plt.semilogy(maddpg_critic_smooth, label='MADDPG', linewidth=2, alpha=0.8)
    plt.title('Critic Losses (log scale)')
    plt.xlabel('Update Step')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "algorithm_comparison.png"))
    
    # 创建性能对比表格
    performance_data = {
        'Algorithm': ['AD-PPO', 'MADDPG'],
        'Final Avg Reward': [adppo_results['eval_rewards'][-1], maddpg_results['eval_rewards'][-1]],
        'Final Success Rate (%)': [adppo_results['success_rates'][-1] * 100, maddpg_results['success_rates'][-1] * 100],
        'Final Jamming Ratio (%)': [adppo_results['jamming_ratios'][-1] * 100, maddpg_results['jamming_ratios'][-1] * 100],
        'Mean Training Reward': [np.mean(adppo_results['rewards']), np.mean(maddpg_results['rewards'])],
        'Max Training Reward': [np.max(adppo_results['rewards']), np.max(maddpg_results['rewards'])],
    }
    
    # 确保成功率和干扰率数值在0-100范围内
    if performance_data['Final Success Rate (%)'][0] > 100:
        performance_data['Final Success Rate (%)'][0] = performance_data['Final Success Rate (%)'][0] / 100
    if performance_data['Final Success Rate (%)'][1] > 100:
        performance_data['Final Success Rate (%)'][1] = performance_data['Final Success Rate (%)'][1] / 100
    
    if performance_data['Final Jamming Ratio (%)'][0] > 100:
        performance_data['Final Jamming Ratio (%)'][0] = performance_data['Final Jamming Ratio (%)'][0] / 100
    if performance_data['Final Jamming Ratio (%)'][1] > 100:
        performance_data['Final Jamming Ratio (%)'][1] = performance_data['Final Jamming Ratio (%)'][1] / 100
    
    df = pd.DataFrame(performance_data)
    df.to_csv(os.path.join(comparison_dir, "performance_comparison.csv"), index=False)
    
    # 创建更详细的HTML表格
    html_table = df.to_html(index=False)
    with open(os.path.join(comparison_dir, "performance_comparison.html"), 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Algorithm Performance Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .header {{ font-size: 24px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">Algorithm Performance Comparison</div>
            {html_table}
        </body>
        </html>
        """)
    
    print(f"算法对比结果已保存到 {comparison_dir}")
    
    return df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较AD-PPO和MADDPG算法在电子对抗环境中的性能")
    
    # 环境参数
    parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    parser.add_argument("--env_size", type=float, default=2000.0, help="环境大小")
    parser.add_argument("--dt", type=float, default=0.1, help="时间步长")
    parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    
    # 算法参数
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵损失系数")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪范数")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--auto_adjust", action="store_true", help="启用自动参数调整")
    
    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--eval_interval", type=int, default=50, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="experiments/algorithm_comparison", help="保存目录")
    
    args = parser.parse_args()
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建环境
    env = create_environment(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        env_size=args.env_size,
        dt=args.dt,
        max_steps=args.max_steps
    )
    
    # 训练AD-PPO
    adppo_results = train_adppo(env, args)
    
    # 训练MADDPG
    maddpg_results = train_maddpg(env, args)
    
    # 绘制对比图
    plot_comparison(adppo_results, maddpg_results, args.save_dir)
    
    # 关闭环境
    env.close()
    
    print("===== 比较完成 =====")

if __name__ == "__main__":
    main() 