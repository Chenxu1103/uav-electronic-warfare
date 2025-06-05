#!/usr/bin/env python3
"""
自动参数调整工具

用于自动搜索和优化强化学习算法的超参数，以获得更好的性能。
支持AD-PPO和MADDPG算法，可以通过网格搜索或贝叶斯优化方法寻找最佳参数组合。

使用示例:
    # 网格搜索
    python autotuner.py grid --algorithm ad_ppo --episodes 100
    
    # 贝叶斯优化
    python autotuner.py bayesian --algorithm ad_ppo --trials 10
"""

import os
import sys
import argparse
import numpy as np
import json
import time
import torch
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 尝试导入贝叶斯优化库
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG
from src.utils.plotting import plot_training_curves

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

def evaluate_adppo(env, params, num_episodes=50, eval_episodes=5):
    """
    使用给定参数训练AD-PPO算法并评估性能
    
    Args:
        env: 环境对象
        params: 参数字典
        num_episodes: 训练回合数
        eval_episodes: 评估回合数
        
    Returns:
        dict: 包含性能指标的字典
    """
    print(f"正在评估参数: {params}")
    
    # 初始化环境
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建AD-PPO智能体
    agent = ADPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=params['hidden_dim'],
        lr=params['learning_rate'],
        gamma=params['gamma'],
        gae_lambda=params['gae_lambda'],
        clip_param=params['clip_param'],
        value_loss_coef=params['value_loss_coef'],
        entropy_coef=params['entropy_coef'],
        max_grad_norm=params['max_grad_norm'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 修改环境的奖励权重
    env.reward_weights.update({
        'jamming_success': params['jamming_success_reward'],
        'partial_success': params['partial_success_reward'],
        'distance_penalty': params['distance_penalty'],
        'energy_penalty': params['energy_penalty'],
        'goal_reward': params['goal_reward'],
        'reward_scale': params['reward_scale']
    })
    
    # 训练记录
    rewards = []
    
    # 训练循环
    for episode in tqdm(range(num_episodes), desc=f"训练参数集"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
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
                
        # 回合结束，更新策略
        if len(agent.buffer.states) > 0:
            # 计算最后状态的价值
            _, _, last_value = agent.select_action(state)
            
            # 计算回报和优势
            agent.buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            
            # 更新策略
            rollout = agent.buffer.get()
            agent.update(rollout)
            
            agent.buffer.clear()
            
        # 记录奖励
        rewards.append(float(episode_reward))  # 确保是Python标准类型，而不是numpy类型
    
    # 评估训练后的模型
    eval_rewards = []
    jamming_ratios = []
    
    for ep in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 确定性策略
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
                
        # 计算雷达干扰率
        jammed_radar_count = sum(1 for radar in env.radars if radar.is_jammed)
        jammed_ratio = jammed_radar_count / len(env.radars)
        jamming_ratios.append(float(jammed_ratio))  # 确保是Python标准类型
        eval_rewards.append(float(episode_reward))  # 确保是Python标准类型
    
    # 计算性能指标
    mean_reward = float(np.mean(rewards[-10:]))  # 最后10回合的平均奖励
    eval_mean_reward = float(np.mean(eval_rewards))
    mean_jamming_ratio = float(np.mean(jamming_ratios))
    
    print(f"训练平均奖励: {mean_reward:.2f}")
    print(f"评估平均奖励: {eval_mean_reward:.2f}")
    print(f"平均干扰率: {mean_jamming_ratio:.2%}")
    
    return {
        'train_reward': mean_reward,
        'eval_reward': eval_mean_reward,
        'jamming_ratio': mean_jamming_ratio,
        'params': params
    }

def grid_search(args):
    """
    执行网格搜索找到最佳参数
    
    Args:
        args: 命令行参数
    """
    print("===== 开始网格搜索 =====")
    
    # 创建环境
    env = create_environment(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        max_steps=args.max_steps
    )
    
    # 参数网格
    if args.algorithm == 'ad_ppo':
        param_grid = {
            'learning_rate': [1e-4, 3e-4, 5e-4],
            'hidden_dim': [128, 256],
            'gamma': [0.99],
            'gae_lambda': [0.9, 0.95],
            'clip_param': [0.1, 0.2],
            'value_loss_coef': [0.5],
            'entropy_coef': [0.005, 0.01, 0.02],
            'max_grad_norm': [0.5],
            # 环境奖励参数
            'jamming_success_reward': [5.0, 10.0],
            'partial_success_reward': [10.0, 20.0],
            'distance_penalty': [-0.001, -0.0001],
            'energy_penalty': [-0.1],
            'goal_reward': [50.0, 100.0],
            'reward_scale': [0.1, 0.2]
        }
    else:  # maddpg
        param_grid = {
            'learning_rate': [1e-4, 3e-4, 5e-4],
            'hidden_dim': [128, 256],
            'gamma': [0.99],
            'tau': [0.01, 0.05],
            'batch_size': [64, 128],
            # 环境奖励参数
            'jamming_success_reward': [5.0, 10.0],
            'partial_success_reward': [10.0, 20.0],
            'distance_penalty': [-0.001, -0.0001],
            'energy_penalty': [-0.1],
            'goal_reward': [50.0, 100.0],
            'reward_scale': [0.1, 0.2]
        }
    
    # 如果指定了快速模式，减少参数网格
    if args.quick:
        for k in param_grid:
            if len(param_grid[k]) > 1:
                param_grid[k] = param_grid[k][:1]
        
        num_episodes = min(args.episodes, 10)
        eval_episodes = 2
    else:
        num_episodes = args.episodes
        eval_episodes = 5
    
    # 创建参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))
    
    print(f"总共 {len(param_combinations)} 种参数组合")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"grid_search_{args.algorithm}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 执行网格搜索
    results = []
    
    for i, combo in enumerate(param_combinations):
        print(f"\n===== 测试参数组合 {i+1}/{len(param_combinations)} =====")
        
        # 构建参数字典
        params = {keys[j]: combo[j] for j in range(len(keys))}
        
        # 评估参数
        result = evaluate_adppo(
            env=env,
            params=params,
            num_episodes=num_episodes,
            eval_episodes=eval_episodes
        )
        
        results.append(result)
        
        # 保存中间结果
        with open(os.path.join(save_dir, "grid_search_results.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    # 按评估奖励排序
    results.sort(key=lambda x: x['eval_reward'], reverse=True)
    
    # 打印最佳参数
    print("\n===== 网格搜索结果 =====")
    print(f"最佳参数组合:")
    for k, v in results[0]['params'].items():
        print(f"  {k}: {v}")
    print(f"训练奖励: {results[0]['train_reward']:.2f}")
    print(f"评估奖励: {results[0]['eval_reward']:.2f}")
    print(f"干扰率: {results[0]['jamming_ratio']:.2%}")
    
    # 绘制参数重要性图
    plot_parameter_importance(results, save_dir)
    
    return results

def bayesian_optimization(args):
    """
    执行贝叶斯优化找到最佳参数
    
    Args:
        args: 命令行参数
    """
    if not SKOPT_AVAILABLE:
        print("错误: 贝叶斯优化需要scikit-optimize库。请安装: pip install scikit-optimize")
        return
    
    print("===== 开始贝叶斯优化 =====")
    
    # 创建环境
    env = create_environment(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        max_steps=args.max_steps
    )
    
    # 定义参数空间
    if args.algorithm == 'ad_ppo':
        space = [
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
            Integer(64, 512, name='hidden_dim'),
            Real(0.9, 0.999, name='gamma'),
            Real(0.8, 0.99, name='gae_lambda'),
            Real(0.05, 0.3, name='clip_param'),
            Real(0.1, 1.0, name='value_loss_coef'),
            Real(0.001, 0.05, name='entropy_coef'),
            Real(0.1, 1.0, name='max_grad_norm'),
            Real(1.0, 20.0, name='jamming_success_reward'),
            Real(5.0, 30.0, name='partial_success_reward'),
            Real(-0.01, -0.00001, name='distance_penalty'),
            Real(-0.5, -0.01, name='energy_penalty'),
            Real(10.0, 200.0, name='goal_reward'),
            Real(0.05, 0.5, name='reward_scale')
        ]
    else:  # maddpg
        space = [
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
            Integer(64, 512, name='hidden_dim'),
            Real(0.9, 0.999, name='gamma'),
            Real(0.001, 0.1, name='tau'),
            Integer(32, 256, name='batch_size'),
            Real(1.0, 20.0, name='jamming_success_reward'),
            Real(5.0, 30.0, name='partial_success_reward'),
            Real(-0.01, -0.00001, name='distance_penalty'),
            Real(-0.5, -0.01, name='energy_penalty'),
            Real(10.0, 200.0, name='goal_reward'),
            Real(0.05, 0.5, name='reward_scale')
        ]
    
    # 创建目标函数
    @use_named_args(space)
    def objective(**params):
        result = evaluate_adppo(
            env=env,
            params=params,
            num_episodes=args.episodes,
            eval_episodes=3
        )
        # 最大化评估奖励
        return -result['eval_reward']
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"bayesian_opt_{args.algorithm}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 执行贝叶斯优化
    res = gp_minimize(
        objective,
        space,
        n_calls=args.trials,
        random_state=SEED,
        verbose=True
    )
    
    # 提取最佳参数
    best_params = {space[i].name: res.x[i] for i in range(len(space))}
    
    # 打印最佳参数
    print("\n===== 贝叶斯优化结果 =====")
    print(f"最佳参数组合:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"最佳评估奖励: {-res.fun:.2f}")
    
    # 保存结果
    with open(os.path.join(save_dir, "bayesian_results.json"), "w") as f:
        json.dump({
            'best_params': best_params,
            'best_reward': -res.fun,
            'all_results': [{
                'params': {space[i].name: x[i] for i in range(len(space))},
                'reward': -y
            } for x, y in zip(res.x_iters, res.func_vals)]
        }, f, indent=4)
    
    return best_params

def plot_parameter_importance(results, save_dir):
    """
    绘制参数重要性图表
    
    Args:
        results: 网格搜索结果
        save_dir: 保存目录
    """
    if not results:
        return
    
    # 提取参数和性能指标
    params_data = {}
    for result in results:
        params = result['params']
        reward = result['eval_reward']
        
        for key, value in params.items():
            if key not in params_data:
                params_data[key] = {'values': [], 'rewards': []}
            
            params_data[key]['values'].append(value)
            params_data[key]['rewards'].append(reward)
    
    # 计算每个参数对奖励的影响
    param_impact = {}
    for key, data in params_data.items():
        unique_values = list(set(data['values']))
        if len(unique_values) <= 1:
            continue
            
        avg_rewards = []
        for val in unique_values:
            indices = [i for i, v in enumerate(data['values']) if v == val]
            avg_reward = np.mean([data['rewards'][i] for i in indices])
            avg_rewards.append(avg_reward)
        
        # 计算最大值与最小值的差距作为参数影响度
        impact = max(avg_rewards) - min(avg_rewards)
        param_impact[key] = impact
    
    # 按影响度排序
    sorted_params = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)
    
    # 绘制参数重要性条形图
    plt.figure(figsize=(12, 6))
    params = [p[0] for p in sorted_params]
    impacts = [p[1] for p in sorted_params]
    
    plt.bar(params, impacts)
    plt.title('Parameter Importance')
    plt.xlabel('Parameter')
    plt.ylabel('Impact on Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_importance.png'))
    
    # 为主要参数绘制箱线图
    top_params = params[:min(5, len(params))]
    
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(top_params):
        plt.subplot(2, 3, i+1)
        
        data = params_data[param]
        values = np.array(data['values'])
        rewards = np.array(data['rewards'])
        
        unique_values = sorted(list(set(values)))
        value_groups = [rewards[values == val] for val in unique_values]
        
        plt.boxplot(value_groups, labels=[str(v) for v in unique_values])
        plt.title(f'Impact of {param}')
        plt.xlabel(param)
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_boxplots.png'))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超参数自动调整工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    subparsers.required = True
    
    # 网格搜索命令
    grid_parser = subparsers.add_parser("grid", help="网格搜索")
    grid_parser.add_argument("--algorithm", type=str, choices=['ad_ppo', 'maddpg'], default='ad_ppo', help="算法")
    grid_parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    grid_parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    grid_parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    grid_parser.add_argument("--episodes", type=int, default=50, help="每个参数组合的训练回合数")
    grid_parser.add_argument("--save_dir", type=str, default="experiments/parameter_tuning", help="保存目录")
    grid_parser.add_argument("--quick", action="store_true", help="快速模式（减少参数组合）")
    
    # 贝叶斯优化命令
    bayes_parser = subparsers.add_parser("bayesian", help="贝叶斯优化")
    bayes_parser.add_argument("--algorithm", type=str, choices=['ad_ppo', 'maddpg'], default='ad_ppo', help="算法")
    bayes_parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    bayes_parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    bayes_parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    bayes_parser.add_argument("--episodes", type=int, default=50, help="每个参数组合的训练回合数")
    bayes_parser.add_argument("--trials", type=int, default=20, help="贝叶斯优化尝试次数")
    bayes_parser.add_argument("--save_dir", type=str, default="experiments/parameter_tuning", help="保存目录")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 执行命令
    if args.command == "grid":
        results = grid_search(args)
    elif args.command == "bayesian":
        results = bayesian_optimization(args)
    else:
        parser.print_help()
        return
    
    print("参数调整完成！")

if __name__ == "__main__":
    main() 