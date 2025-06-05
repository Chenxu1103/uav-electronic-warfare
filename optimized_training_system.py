#!/usr/bin/env python3
"""
优化训练系统 - 逐步接近论文理想数据

本系统包含以下优化:
1. 改进的算法实现
2. 优化的环境设计
3. 智能训练策略
4. 自动超参数调优
5. 渐进式训练流程
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG

class OptimizedTrainingSystem:
    """优化的训练系统"""
    
    def __init__(self, config=None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.env = self._create_optimized_environment()
        self.performance_tracker = PerformanceTracker()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.curriculum_trainer = CurriculumTrainer()
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            # 环境参数
            'num_uavs': 3,
            'num_radars': 2,
            'env_size': 2000.0,
            'max_steps': 200,
            
            # 训练参数
            'total_episodes': 2000,
            'evaluation_interval': 50,
            'save_interval': 100,
            'batch_size': 64,
            
            # 优化参数
            'target_performance': {
                'reconnaissance_completion': 0.90,
                'safe_zone_time': 2.0,
                'reconnaissance_cooperation': 35.0,
                'jamming_cooperation': 30.0,
                'jamming_failure_rate': 25.0
            },
            
            # 自适应参数
            'adaptive_learning': True,
            'curriculum_learning': True,
            'auto_tuning': True
        }
    
    def _create_optimized_environment(self):
        """创建优化的环境"""
        env = ElectronicWarfareEnv(
            num_uavs=self.config['num_uavs'],
            num_radars=self.config['num_radars'],
            env_size=self.config['env_size'],
            max_steps=self.config['max_steps']
        )
        
        # 优化奖励权重
        env.reward_weights.update({
            'jamming_success': 100.0,           # 增加干扰成功奖励
            'partial_success': 50.0,            # 部分成功奖励
            'distance_penalty': -0.00005,       # 减小距离惩罚
            'energy_penalty': -0.005,           # 减小能量惩罚
            'detection_penalty': -0.1,          # 减小检测惩罚
            'death_penalty': -1.0,              # 减小死亡惩罚
            'goal_reward': 1000.0,              # 增加目标奖励
            'coordination_reward': 50.0,        # 增加协调奖励
            'stealth_reward': 1.0,              # 增加隐身奖励
            'approach_reward': 15.0,            # 增加接近奖励
            'jamming_attempt_reward': 8.0,      # 增加尝试干扰奖励
            'reward_scale': 0.8,                # 增加奖励缩放
            'min_reward': -10.0,                # 调整最小奖励
            'max_reward': 150.0,                # 增加最大奖励
        })
        
        return env
    
    def train_optimized_adppo(self):
        """训练优化的AD-PPO算法"""
        print("🚀 开始优化AD-PPO训练...")
        
        # 创建优化的AD-PPO智能体
        agent = self._create_optimized_adppo()
        
        # 渐进式训练
        results = self.curriculum_trainer.progressive_training(
            agent=agent,
            env=self.env,
            config=self.config,
            performance_tracker=self.performance_tracker
        )
        
        return agent, results
    
    def train_optimized_maddpg(self):
        """训练优化的MADDPG算法"""
        print("🚀 开始优化MADDPG训练...")
        
        # 创建优化的MADDPG智能体
        agent = self._create_optimized_maddpg()
        
        # 渐进式训练
        results = self.curriculum_trainer.progressive_training(
            agent=agent,
            env=self.env,
            config=self.config,
            performance_tracker=self.performance_tracker,
            algorithm_type='maddpg'
        )
        
        return agent, results
    
    def _create_optimized_adppo(self):
        """创建优化的AD-PPO智能体"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # 使用优化的超参数
        optimized_params = self.hyperparameter_optimizer.get_optimal_adppo_params()
        
        agent = ADPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=optimized_params['hidden_dim'],
            lr=optimized_params['learning_rate'],
            gamma=optimized_params['gamma'],
            gae_lambda=optimized_params['gae_lambda'],
            clip_param=optimized_params['clip_param'],
            value_loss_coef=optimized_params['value_loss_coef'],
            entropy_coef=optimized_params['entropy_coef'],
            max_grad_norm=optimized_params['max_grad_norm'],
            device='cpu'
        )
        
        return agent
    
    def _create_optimized_maddpg(self):
        """创建优化的MADDPG智能体"""
        state_dim = self.env.observation_space.shape[0] // self.env.num_uavs
        action_dim = self.env.action_space.shape[0] // self.env.num_uavs
        
        # 使用优化的超参数
        optimized_params = self.hyperparameter_optimizer.get_optimal_maddpg_params()
        
        agent = MADDPG(
            n_agents=self.env.num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=optimized_params['hidden_dim'],
            lr_actor=optimized_params['lr_actor'],
            lr_critic=optimized_params['lr_critic'],
            gamma=optimized_params['gamma'],
            tau=optimized_params['tau'],
            batch_size=optimized_params['batch_size'],
            buffer_size=int(optimized_params['buffer_size'])
        )
        
        return agent

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.metrics_history = {
            'rewards': deque(maxlen=window_size),
            'success_rates': deque(maxlen=window_size),
            'jamming_ratios': deque(maxlen=window_size),
            'cooperation_rates': deque(maxlen=window_size),
            'completion_rates': deque(maxlen=window_size)
        }
        
    def update(self, metrics):
        """更新性能指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_trend(self, metric_name):
        """获取指标趋势"""
        if metric_name not in self.metrics_history:
            return 0
        
        values = list(self.metrics_history[metric_name])
        if len(values) < 2:
            return 0
        
        # 计算趋势斜率
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def get_current_performance(self):
        """获取当前性能"""
        performance = {}
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                performance[metric] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'trend': self.get_trend(metric)
                }
        return performance

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self):
        self.adppo_search_space = {
            'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],
            'hidden_dim': [128, 256, 512],
            'gamma': [0.95, 0.99, 0.995],
            'gae_lambda': [0.9, 0.95, 0.98],
            'clip_param': [0.1, 0.2, 0.3],
            'value_loss_coef': [0.25, 0.5, 1.0],
            'entropy_coef': [0.005, 0.01, 0.02],
            'max_grad_norm': [0.3, 0.5, 1.0]
        }
        
        self.maddpg_search_space = {
            'lr_actor': [1e-4, 3e-4, 5e-4],
            'lr_critic': [3e-4, 5e-4, 1e-3],
            'hidden_dim': [128, 256, 512],
            'gamma': [0.95, 0.99, 0.995],
            'tau': [0.005, 0.01, 0.02],
            'batch_size': [32, 64, 128],
            'buffer_size': [5e5, 1e6, 2e6]
        }
        
        self.best_params = {
            'adppo': None,
            'maddpg': None
        }
        
    def get_optimal_adppo_params(self):
        """获取优化的AD-PPO参数"""
        if self.best_params['adppo'] is None:
            # 返回经过调优的默认参数
            self.best_params['adppo'] = {
                'learning_rate': 3e-4,
                'hidden_dim': 256,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5
            }
        return self.best_params['adppo']
    
    def get_optimal_maddpg_params(self):
        """获取优化的MADDPG参数"""
        if self.best_params['maddpg'] is None:
            # 返回经过调优的默认参数
            self.best_params['maddpg'] = {
                'lr_actor': 3e-4,
                'lr_critic': 5e-4,
                'hidden_dim': 256,
                'gamma': 0.99,
                'tau': 0.01,
                'batch_size': 64,
                'buffer_size': 1e6
            }
        return self.best_params['maddpg']
    
    def bayesian_optimization(self, algorithm_type, objective_func, n_trials=20):
        """贝叶斯优化超参数"""
        search_space = self.adppo_search_space if algorithm_type == 'adppo' else self.maddpg_search_space
        
        best_score = -float('inf')
        best_params = None
        
        for trial in range(n_trials):
            # 随机采样参数（简化版贝叶斯优化）
            params = {}
            for key, values in search_space.items():
                params[key] = random.choice(values)
            
            # 评估参数
            score = objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            print(f"Trial {trial+1}/{n_trials}: Score = {score:.4f}")
        
        self.best_params[algorithm_type] = best_params
        return best_params, best_score

class CurriculumTrainer:
    """课程学习训练器"""
    
    def __init__(self):
        self.difficulty_levels = [
            {'num_radars': 1, 'env_size': 1000, 'max_steps': 150},  # 简单
            {'num_radars': 2, 'env_size': 1500, 'max_steps': 175},  # 中等
            {'num_radars': 2, 'env_size': 2000, 'max_steps': 200},  # 困难
            {'num_radars': 3, 'env_size': 2000, 'max_steps': 200},  # 专家
        ]
        
    def progressive_training(self, agent, env, config, performance_tracker, algorithm_type='adppo'):
        """渐进式训练"""
        print("📚 开始课程学习训练...")
        
        all_results = []
        total_episodes = 0
        
        for level, difficulty in enumerate(self.difficulty_levels):
            print(f"\n🎯 训练难度等级 {level+1}/4: {difficulty}")
            
            # 调整环境难度
            self._adjust_environment_difficulty(env, difficulty)
            
            # 计算该难度级别的训练回合数
            episodes_for_level = config['total_episodes'] // len(self.difficulty_levels)
            
            # 训练该难度级别
            level_results = self._train_level(
                agent=agent,
                env=env,
                episodes=episodes_for_level,
                level=level,
                algorithm_type=algorithm_type,
                performance_tracker=performance_tracker
            )
            
            all_results.extend(level_results)
            total_episodes += episodes_for_level
            
            # 检查是否达到该级别的通过标准
            if self._check_level_completion(level_results):
                print(f"✅ 完成难度等级 {level+1}")
            else:
                print(f"⚠️ 难度等级 {level+1} 需要更多训练")
        
        print(f"🎉 课程学习完成! 总训练回合: {total_episodes}")
        return all_results
    
    def _adjust_environment_difficulty(self, env, difficulty):
        """调整环境难度"""
        # 重新初始化环境参数
        env.num_radars = difficulty['num_radars']
        env.env_size = difficulty['env_size']
        env.max_steps = difficulty['max_steps']
        
        # 重新初始化雷达
        env.radars = []
        for i in range(env.num_radars):
            from src.models.radar_model import Radar
            position = np.random.uniform(
                [-env.env_size/4, -env.env_size/4, 0],
                [env.env_size/4, env.env_size/4, 100]
            )
            radar = Radar(position=position)
            env.radars.append(radar)
    
    def _train_level(self, agent, env, episodes, level, algorithm_type, performance_tracker):
        """训练特定难度级别"""
        level_results = []
        
        for episode in range(episodes):
            # 训练一个回合
            episode_result = self._train_episode(agent, env, algorithm_type)
            level_results.append(episode_result)
            
            # 更新性能跟踪
            performance_tracker.update({
                'rewards': episode_result['reward'],
                'success_rates': episode_result['success'],
                'jamming_ratios': episode_result['jamming_ratio']
            })
            
            # 打印进度
            if episode % 20 == 0:
                avg_reward = np.mean([r['reward'] for r in level_results[-20:]])
                print(f"  级别 {level+1} - 回合 {episode}/{episodes}, 平均奖励: {avg_reward:.2f}")
        
        return level_results
    
    def _train_episode(self, agent, env, algorithm_type):
        """训练单个回合"""
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < env.max_steps:
            if algorithm_type == 'adppo':
                action, log_prob, value = agent.select_action(state)
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
                
            else:  # maddpg
                state_dim = state.shape[0] // env.num_uavs
                agent_states = []
                for i in range(env.num_uavs):
                    agent_states.append(state[i*state_dim:(i+1)*state_dim])
                agent_states = np.array(agent_states)
                
                actions = agent.select_action(agent_states)
                combined_action = np.concatenate(actions)
                next_state, reward, done, info = env.step(combined_action)
                
                # 准备MADDPG的经验
                agent_next_states = []
                agent_rewards = []
                for i in range(env.num_uavs):
                    agent_next_states.append(next_state[i*state_dim:(i+1)*state_dim])
                    agent_rewards.append(reward / env.num_uavs)
                
                agent_next_states = np.array(agent_next_states)
                agent_rewards = np.array(agent_rewards)
                
                # 存储经验
                agent.replay_buffer.add(agent_states, actions, agent_next_states, agent_rewards, done)
                
                # 更新策略
                if agent.replay_buffer.size > 64:
                    agent.update()
            
            state = next_state
            total_reward += reward
            step += 1
        
        # AD-PPO的回合结束处理
        if algorithm_type == 'adppo' and len(agent.buffer.states) > 0:
            _, _, last_value = agent.select_action(state)
            agent.buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            rollout = agent.buffer.get()
            agent.update(rollout)
            agent.buffer.clear()
        
        # 计算性能指标
        jammed_count = sum(1 for radar in env.radars if radar.is_jammed)
        jamming_ratio = jammed_count / len(env.radars)
        success = jamming_ratio >= 0.5
        
        return {
            'reward': total_reward,
            'success': success,
            'jamming_ratio': jamming_ratio,
            'steps': step
        }
    
    def _check_level_completion(self, level_results):
        """检查级别完成情况"""
        if len(level_results) < 20:
            return False
        
        # 检查最近20个回合的表现
        recent_results = level_results[-20:]
        avg_reward = np.mean([r['reward'] for r in recent_results])
        success_rate = np.mean([r['success'] for r in recent_results])
        
        # 设定通过标准
        return avg_reward > 300 and success_rate > 0.3

class AdaptiveEnvironmentOptimizer:
    """自适应环境优化器"""
    
    def __init__(self, env):
        self.env = env
        self.performance_history = []
        
    def optimize_rewards(self, performance_metrics):
        """根据性能指标优化奖励函数"""
        # 如果干扰成功率太低，增加干扰奖励
        if performance_metrics.get('jamming_ratio', 0) < 0.3:
            self.env.reward_weights['jamming_success'] *= 1.1
            self.env.reward_weights['jamming_attempt_reward'] *= 1.05
        
        # 如果协作率太低，增加协作奖励
        if performance_metrics.get('cooperation_rate', 0) < 0.2:
            self.env.reward_weights['coordination_reward'] *= 1.1
        
        # 如果完成度太低，减少惩罚
        if performance_metrics.get('completion_rate', 0) < 0.5:
            self.env.reward_weights['distance_penalty'] *= 0.9
            self.env.reward_weights['energy_penalty'] *= 0.9
        
        print(f"🔧 环境奖励权重已自适应调整")

def main():
    """主函数 - 运行优化训练系统"""
    print("🎯 启动优化训练系统...")
    
    # 创建优化训练系统
    training_system = OptimizedTrainingSystem()
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/optimized_training/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练AD-PPO
    print("\n🚀 开始优化AD-PPO训练...")
    adppo_agent, adppo_results = training_system.train_optimized_adppo()
    
    # 训练MADDPG
    print("\n🚀 开始优化MADDPG训练...")
    maddpg_agent, maddpg_results = training_system.train_optimized_maddpg()
    
    # 保存模型
    adppo_agent.save(os.path.join(results_dir, "optimized_adppo_model.pt"))
    torch.save(maddpg_agent.state_dict(), os.path.join(results_dir, "optimized_maddpg_model.pt"))
    
    # 评估最终性能
    print("\n📊 评估最终性能...")
    final_evaluation = evaluate_optimized_performance(
        adppo_agent, maddpg_agent, training_system.env, results_dir
    )
    
    print(f"✅ 优化训练完成! 结果保存在: {results_dir}")
    return final_evaluation

def evaluate_optimized_performance(adppo_agent, maddpg_agent, env, save_dir):
    """评估优化后的性能"""
    from enhanced_performance_comparison import collect_detailed_episode_data, create_table_5_2_comparison
    
    print("正在评估优化后的性能...")
    
    # 收集详细性能数据
    adppo_results = collect_detailed_episode_data(adppo_agent, env, 'adppo', 50)
    maddpg_results = collect_detailed_episode_data(maddpg_agent, env, 'maddpg', 50)
    
    # 生成对比报告
    comparison_df = create_table_5_2_comparison(adppo_results, maddpg_results, save_dir)
    
    # 打印改进情况
    print("\n📈 优化效果总结:")
    print("=" * 60)
    print(f"{'指标':<25} {'AD-PPO':<15} {'MADDPG':<15} {'目标值':<15}")
    print("-" * 60)
    print(f"{'侦察任务完成度':<25} {adppo_results['reconnaissance_completion']:<15.3f} {maddpg_results['reconnaissance_completion']:<15.3f} {'0.90':<15}")
    print(f"{'安全区域开辟时间':<25} {adppo_results['safe_zone_development_time']:<15.1f} {maddpg_results['safe_zone_development_time']:<15.1f} {'2.0':<15}")
    print(f"{'侦察协作率(%)':<25} {adppo_results['reconnaissance_cooperation_rate']:<15.1f} {maddpg_results['reconnaissance_cooperation_rate']:<15.1f} {'35.0':<15}")
    print(f"{'干扰协作率(%)':<25} {adppo_results['jamming_cooperation_rate']:<15.1f} {maddpg_results['jamming_cooperation_rate']:<15.1f} {'30.0':<15}")
    print(f"{'干扰动作失效率(%)':<25} {adppo_results['jamming_failure_rate']:<15.1f} {maddpg_results['jamming_failure_rate']:<15.1f} {'25.0':<15}")
    
    return {
        'adppo': adppo_results,
        'maddpg': maddpg_results,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    main() 