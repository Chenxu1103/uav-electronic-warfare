#!/usr/bin/env python3
"""
使用自动调优后的最佳参数运行算法

本脚本允许用户使用自动参数调整工具找到的最佳参数运行算法比较。
可以直接从自动调整的结果文件中加载参数，也可以手动指定参数文件。

使用示例:
    # 使用最近的参数调整结果
    python run_autotuned.py --latest

    # 指定参数文件
    python run_autotuned.py --params_file experiments/parameter_tuning/grid_search_ad_ppo_20230510_123456/grid_search_results.json

    # 手动指定超参数（以AD-PPO为例）
    python run_autotuned.py --algorithm ad_ppo --learning_rate 1e-4 --hidden_dim 512
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
from datetime import datetime
from compare_algorithms import train_adppo, train_maddpg, evaluate_agent, plot_comparison, create_environment
import pandas as pd

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def find_latest_params_file():
    """
    查找最近的参数文件
    
    Returns:
        str: 最近参数文件的路径，如果没有找到则返回None
    """
    # 搜索所有参数调整结果文件
    search_dirs = [
        os.path.join("experiments", "parameter_tuning"),
        os.path.join("experiments", "algorithm_comparison")
    ]
    
    param_files = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file in ["grid_search_results.json", "bayesian_results.json", "parameter_adjustments.txt"]:
                    param_files.append(os.path.join(root, file))
    
    if not param_files:
        return None
    
    # 按修改时间排序
    param_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return param_files[0]

def load_params(params_file):
    """
    从参数文件加载最佳参数
    
    Args:
        params_file: 参数文件路径
        
    Returns:
        dict: 参数字典
    """
    if not os.path.exists(params_file):
        print(f"错误: 参数文件 '{params_file}' 不存在")
        return None
    
    try:
        # 检查文件类型
        if params_file.endswith(".json"):
            with open(params_file, "r") as f:
                data = json.load(f)
                
            # 提取最佳参数
            if "best_params" in data:  # 贝叶斯优化结果
                return data["best_params"]
            elif isinstance(data, list) and len(data) > 0:  # 网格搜索结果
                # 按评估奖励排序
                data.sort(key=lambda x: x.get('eval_reward', x.get('train_reward', 0)), reverse=True)
                return data[0]['params']
            else:
                print(f"警告: 无法从文件中提取参数")
                return None
        
        elif params_file.endswith(".txt"):  # 参数调整记录
            # 读取最后一个参数调整
            params = {}
            with open(params_file, "r") as f:
                lines = f.readlines()
                
            if len(lines) < 2:
                print(f"警告: 参数调整记录文件为空")
                return None
                
            # 解析最后一次调整
            for line in reversed(lines[1:]):  # 跳过标题行
                if ":" in line:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        episode, adjustment = parts[0], parts[1]
                        
                        # 解析参数调整
                        if "学习率" in adjustment:
                            value = adjustment.split("->")[1].strip()
                            params["learning_rate"] = float(value)
                        elif "熵系数" in adjustment:
                            value = adjustment.split("->")[1].strip()
                            params["entropy_coef"] = float(value)
                        elif "裁剪参数" in adjustment:
                            value = adjustment.split("->")[1].strip()
                            params["clip_param"] = float(value)
            
            return params
        
        else:
            print(f"警告: 不支持的文件类型 '{params_file}'")
            return None
            
    except Exception as e:
        print(f"错误: 加载参数时出错: {str(e)}")
        return None

def run_with_params(args):
    """
    使用指定参数运行算法比较
    
    Args:
        args: 命令行参数
    """
    # 创建环境
    env = create_environment(
        num_uavs=args.num_uavs,
        num_radars=args.num_radars,
        env_size=args.env_size,
        dt=args.dt,
        max_steps=args.max_steps
    )
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存使用的参数
    with open(os.path.join(save_dir, "used_parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 训练AD-PPO算法
    print("===== 训练AD-PPO算法 =====")
    adppo_results = train_adppo(env, args)
    
    # 训练MADDPG算法
    print("===== 训练MADDPG算法 =====")
    maddpg_results = train_maddpg(env, args)
    
    # 绘制对比图
    plot_comparison(adppo_results, maddpg_results, save_dir)
    
    # 创建比较表
    performance_data = {
        'Algorithm': ['AD-PPO', 'MADDPG'],
        'Final Avg Reward': [adppo_results['eval_rewards'][-1], maddpg_results['eval_rewards'][-1]],
        'Final Success Rate (%)': [adppo_results['success_rates'][-1], maddpg_results['success_rates'][-1]],
        'Final Jamming Ratio (%)': [adppo_results['jamming_ratios'][-1], maddpg_results['jamming_ratios'][-1]],
        'Mean Training Reward': [np.mean(adppo_results['rewards']), np.mean(maddpg_results['rewards'])],
        'Max Training Reward': [np.max(adppo_results['rewards']), np.max(maddpg_results['rewards'])],
    }
    
    # 确保成功率和干扰率数值在0-1范围内，然后乘以100转为百分比
    for i in range(2):
        if performance_data['Final Success Rate (%)'][i] > 1:
            performance_data['Final Success Rate (%)'][i] = performance_data['Final Success Rate (%)'][i] / 100
        performance_data['Final Success Rate (%)'][i] = performance_data['Final Success Rate (%)'][i] * 100
            
        if performance_data['Final Jamming Ratio (%)'][i] > 1:
            performance_data['Final Jamming Ratio (%)'][i] = performance_data['Final Jamming Ratio (%)'][i] / 100
        performance_data['Final Jamming Ratio (%)'][i] = performance_data['Final Jamming Ratio (%)'][i] * 100
    
    df = pd.DataFrame(performance_data)
    comparison_dir = os.path.join(args.save_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    df.to_csv(os.path.join(comparison_dir, "performance_comparison.csv"), index=False)
    
    # 关闭环境
    env.close()
    
    print(f"===== 比较完成，结果保存在 {save_dir} =====")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用自动调优参数运行算法比较")
    
    # 参数来源选项
    parser.add_argument("--latest", action="store_true", help="使用最近的参数调整结果")
    parser.add_argument("--params_file", type=str, help="参数文件路径")
    
    # 环境参数
    parser.add_argument("--num_uavs", type=int, default=3, help="无人机数量")
    parser.add_argument("--num_radars", type=int, default=2, help="雷达数量")
    parser.add_argument("--env_size", type=float, default=2000.0, help="环境大小")
    parser.add_argument("--dt", type=float, default=0.1, help="时间步长")
    parser.add_argument("--max_steps", type=int, default=200, help="最大步数")
    
    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--eval_interval", type=int, default=50, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="experiments/autotuned_runs", help="保存目录")
    
    # 一些常用参数，方便手动调整
    parser.add_argument("--algorithm", type=str, choices=["ad_ppo", "maddpg", "both"], default="both", help="要运行的算法")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--hidden_dim", type=int, help="隐藏层维度")
    parser.add_argument("--gamma", type=float, help="折扣因子")
    parser.add_argument("--clip_param", type=float, help="PPO裁剪参数")
    parser.add_argument("--entropy_coef", type=float, help="熵系数")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    
    # AD-PPO特有参数
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪范数")
    
    # MADDPG特有参数
    parser.add_argument("--tau", type=float, default=0.01, help="软更新系数")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="经验回放缓冲区大小")
    
    # 环境奖励参数
    parser.add_argument("--jamming_success_reward", type=float, help="干扰成功奖励")
    parser.add_argument("--partial_success_reward", type=float, help="部分干扰成功奖励")
    parser.add_argument("--distance_penalty", type=float, help="距离惩罚系数")
    parser.add_argument("--energy_penalty", type=float, help="能量惩罚系数")
    parser.add_argument("--goal_reward", type=float, help="目标达成奖励")
    parser.add_argument("--reward_scale", type=float, help="奖励缩放因子")
    
    # 辅助参数
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔（回合）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（回合）")
    parser.add_argument("--plot_interval", type=int, default=100, help="绘图间隔（回合）")
    
    # 自动调整标志
    parser.add_argument("--auto_adjust", action="store_true", help="启用自动参数调整")
    
    args = parser.parse_args()
    
    # 加载参数
    params = {}
    
    if args.latest:
        params_file = find_latest_params_file()
        if params_file:
            print(f"使用最近的参数文件: {params_file}")
            params = load_params(params_file)
        else:
            print("警告: 未找到参数文件，使用默认参数")
            
    elif args.params_file:
        params = load_params(args.params_file)
        if not params:
            print("警告: 无法从指定的参数文件加载参数，使用默认参数")
    
    # 将加载的参数更新到args
    if params:
        for key, value in params.items():
            if not getattr(args, key, None):  # 只更新未手动指定的参数
                setattr(args, key, value)
        
        print("使用加载的参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # 设置奖励参数的默认值（如果未指定）
    if not args.jamming_success_reward:
        args.jamming_success_reward = 5.0
    if not args.partial_success_reward:
        args.partial_success_reward = 10.0
    if not args.distance_penalty:
        args.distance_penalty = -0.001
    if not args.energy_penalty:
        args.energy_penalty = -0.1
    if not args.goal_reward:
        args.goal_reward = 50.0
    if not args.reward_scale:
        args.reward_scale = 0.1
        
    # 设置算法参数的默认值（如果未指定）
    if not args.learning_rate:
        args.learning_rate = 3e-4
    if not args.hidden_dim:
        args.hidden_dim = 256
    if not args.gamma:
        args.gamma = 0.99
    if not args.clip_param:
        args.clip_param = 0.2
    if not args.entropy_coef:
        args.entropy_coef = 0.01
    if not args.batch_size:
        args.batch_size = 64
    
    # 使用更新后的参数运行
    run_with_params(args)

if __name__ == "__main__":
    main() 