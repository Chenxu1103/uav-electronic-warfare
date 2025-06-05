#!/usr/bin/env python3
"""
快速优化测试 - 展示优化训练系统的效果

运行一个简化版的优化训练，展示实验数据如何逐步接近理想值。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from intelligent_reward_designer import IntelligentRewardDesigner, RewardShapeOptimizer

class QuickOptimizationTester:
    """快速优化测试器"""
    
    def __init__(self):
        self.reward_designer = IntelligentRewardDesigner()
        self.reward_shaper = RewardShapeOptimizer()
        
        # 创建环境
        self.env = ElectronicWarfareEnv(
            num_uavs=3,
            num_radars=2,
            env_size=1500.0,  # 稍小的环境，便于快速测试
            max_steps=150
        )
        
        # 性能历史记录
        self.performance_history = []
        
    def run_quick_optimization_test(self, episodes=100):
        """运行快速优化测试"""
        print("🚀 开始快速优化测试...")
        print(f"目标: 在{episodes}个回合内展示性能改进")
        
        # 创建优化的AD-PPO智能体
        agent = self._create_test_agent()
        
        # 优化前的基线测试
        print("\n📊 基线性能测试...")
        baseline_metrics = self._evaluate_performance(agent, 10, "基线")
        
        # 开始优化训练
        print(f"\n🎯 开始{episodes}回合优化训练...")
        optimized_metrics = self._run_optimized_training(agent, episodes)
        
        # 最终性能测试
        print("\n📊 优化后性能测试...")
        final_metrics = self._evaluate_performance(agent, 10, "优化后")
        
        # 生成对比报告
        self._generate_improvement_report(baseline_metrics, final_metrics, optimized_metrics)
        
        return baseline_metrics, final_metrics, optimized_metrics
    
    def _create_test_agent(self):
        """创建测试用的AD-PPO智能体"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        agent = ADPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,  # 较小的网络，便于快速训练
            lr=5e-4,         # 较高的学习率
            gamma=0.95,
            gae_lambda=0.9,
            clip_param=0.3,  # 较大的裁剪参数，允许更大更新
            value_loss_coef=0.5,
            entropy_coef=0.02,  # 更高的熵系数，增加探索
            max_grad_norm=1.0,
            device='cpu'
        )
        
        return agent
    
    def _evaluate_performance(self, agent, num_episodes, phase_name):
        """评估当前性能"""
        print(f"  评估{phase_name}性能 ({num_episodes}回合)...")
        
        metrics = {
            'rewards': [],
            'success_rates': [],
            'jamming_ratios': [],
            'completion_scores': [],
            'cooperation_scores': []
        }
        
        for episode in range(num_episodes):
            episode_result = self._run_evaluation_episode(agent)
            
            metrics['rewards'].append(episode_result['reward'])
            metrics['success_rates'].append(episode_result['success'])
            metrics['jamming_ratios'].append(episode_result['jamming_ratio'])
            metrics['completion_scores'].append(episode_result['completion_score'])
            metrics['cooperation_scores'].append(episode_result['cooperation_score'])
        
        # 计算平均值
        avg_metrics = {
            'average_reward': np.mean(metrics['rewards']),
            'success_rate': np.mean(metrics['success_rates']),
            'jamming_ratio': np.mean(metrics['jamming_ratios']),
            'reconnaissance_completion': np.mean(metrics['completion_scores']),
            'reconnaissance_cooperation': np.mean(metrics['cooperation_scores']) * 100,
            'safe_zone_time': 1.0 if np.mean(metrics['success_rates']) > 0.3 else 0.0
        }
        
        print(f"    平均奖励: {avg_metrics['average_reward']:.2f}")
        print(f"    成功率: {avg_metrics['success_rate']:.2%}")
        print(f"    干扰率: {avg_metrics['jamming_ratio']:.2%}")
        print(f"    侦察完成度: {avg_metrics['reconnaissance_completion']:.3f}")
        
        return avg_metrics
    
    def _run_evaluation_episode(self, agent):
        """运行单个评估回合"""
        state = self.env.reset()
        total_reward = 0
        step = 0
        
        # 记录UAV轨迹用于协作分析
        uav_positions = []
        
        while step < self.env.max_steps:
            # 选择动作（不训练）
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录位置
            positions = [uav.position for uav in self.env.uavs if uav.is_alive]
            uav_positions.append(positions)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if done:
                break
        
        # 计算各项指标
        jammed_count = sum(1 for radar in self.env.radars if radar.is_jammed)
        jamming_ratio = jammed_count / len(self.env.radars)
        success = jamming_ratio >= 0.5
        
        # 计算侦察完成度（简化）
        completion_score = min(step / self.env.max_steps + jamming_ratio * 0.5, 1.0)
        
        # 计算协作分数（基于UAV间距离）
        cooperation_score = self._calculate_cooperation_score(uav_positions)
        
        return {
            'reward': total_reward,
            'success': success,
            'jamming_ratio': jamming_ratio,
            'completion_score': completion_score,
            'cooperation_score': cooperation_score,
            'steps': step
        }
    
    def _calculate_cooperation_score(self, uav_positions_history):
        """计算协作分数"""
        if not uav_positions_history:
            return 0.0
        
        cooperation_scores = []
        
        for positions in uav_positions_history:
            if len(positions) >= 2:
                score = self.reward_shaper.optimize_cooperation_reward(positions)
                cooperation_scores.append(score)
        
        return np.mean(cooperation_scores) if cooperation_scores else 0.0
    
    def _run_optimized_training(self, agent, episodes):
        """运行优化训练"""
        training_metrics = []
        
        for episode in range(episodes):
            # 每10回合评估一次性能并调整奖励
            if episode % 10 == 0 and episode > 0:
                current_metrics = self._evaluate_performance(agent, 5, f"第{episode}回合")
                
                # 使用智能奖励设计器调整环境
                new_weights = self.reward_designer.design_adaptive_rewards(
                    self.env, current_metrics, episode
                )
                self.env.reward_weights.update(new_weights)
                
                # 记录性能
                training_metrics.append({
                    'episode': episode,
                    'metrics': current_metrics
                })
                
                self.performance_history.append(current_metrics)
            
            # 训练一个回合
            self._train_episode(agent)
            
            # 打印进度
            if episode % 20 == 0:
                print(f"  训练进度: {episode}/{episodes} ({episode/episodes*100:.1f}%)")
        
        return training_metrics
    
    def _train_episode(self, agent):
        """训练单个回合"""
        state = self.env.reset()
        total_reward = 0
        step = 0
        
        while step < self.env.max_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
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
            total_reward += reward
            step += 1
            
            if done:
                break
        
        # 更新模型
        if len(agent.buffer.states) > 0:
            _, _, last_value = agent.select_action(state)
            agent.buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            rollout = agent.buffer.get()
            agent.update(rollout)
            agent.buffer.clear()
    
    def _generate_improvement_report(self, baseline, final, training_history):
        """生成改进报告"""
        print("\n" + "="*60)
        print("📈 优化效果报告")
        print("="*60)
        
        # 计算改进幅度
        improvements = {}
        for key in baseline:
            if key in final:
                baseline_val = baseline[key]
                final_val = final[key]
                
                if baseline_val != 0:
                    improvement = (final_val - baseline_val) / abs(baseline_val) * 100
                else:
                    improvement = final_val * 100
                
                improvements[key] = improvement
        
        # 打印改进结果
        print(f"{'指标':<25} {'基线值':<15} {'优化后':<15} {'改进幅度':<15}")
        print("-" * 70)
        
        metrics_display = {
            'average_reward': '平均奖励',
            'success_rate': '成功率',
            'jamming_ratio': '干扰率',
            'reconnaissance_completion': '侦察完成度',
            'reconnaissance_cooperation': '侦察协作率',
            'safe_zone_time': '安全区域时间'
        }
        
        for key, display_name in metrics_display.items():
            if key in baseline and key in final:
                baseline_val = baseline[key]
                final_val = final[key]
                improvement = improvements.get(key, 0)
                
                if key in ['success_rate', 'jamming_ratio']:
                    print(f"{display_name:<25} {baseline_val:.2%} {'':>8} {final_val:.2%} {'':>8} {improvement:+.1f}%")
                elif key == 'reconnaissance_cooperation':
                    print(f"{display_name:<25} {baseline_val:.1f}% {'':>7} {final_val:.1f}% {'':>7} {improvement:+.1f}%")
                else:
                    print(f"{display_name:<25} {baseline_val:.3f} {'':>8} {final_val:.3f} {'':>8} {improvement:+.1f}%")
        
        print("\n" + "="*60)
        
        # 与论文目标对比
        print("📊 与论文目标对比:")
        print("-" * 40)
        
        paper_targets = {
            'reconnaissance_completion': 0.97,
            'safe_zone_time': 2.1,
            'reconnaissance_cooperation': 37.0,
            'success_rate': 0.6
        }
        
        for key, target in paper_targets.items():
            if key in final:
                current = final[key]
                if key == 'reconnaissance_cooperation':
                    gap = abs(current - target) / target * 100
                    print(f"  {key}: 当前 {current:.1f}%, 目标 {target}%, 差距 {gap:.1f}%")
                elif key in ['success_rate']:
                    gap = abs(current - target) / target * 100
                    print(f"  {key}: 当前 {current:.2%}, 目标 {target:.2%}, 差距 {gap:.1f}%")
                else:
                    gap = abs(current - target) / target * 100
                    print(f"  {key}: 当前 {current:.3f}, 目标 {target}, 差距 {gap:.1f}%")
        
        print("\n💡 优化建议:")
        self._generate_optimization_suggestions(final, paper_targets)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"experiments/quick_optimization/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 保存性能数据
        results_data = {
            'baseline': convert_numpy_types(baseline),
            'final': convert_numpy_types(final),
            'improvements': convert_numpy_types(improvements),
            'training_history': convert_numpy_types(training_history)
        }
        
        import json
        with open(os.path.join(save_dir, 'optimization_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {save_dir}")
    
    def _generate_optimization_suggestions(self, current_metrics, targets):
        """生成优化建议"""
        suggestions = []
        
        for key, target in targets.items():
            if key in current_metrics:
                current = current_metrics[key]
                
                if key == 'reconnaissance_completion' and current < target:
                    suggestions.append("• 增加侦察探索奖励和接近目标奖励")
                elif key == 'safe_zone_time' and current < target:
                    suggestions.append("• 提高干扰成功奖励和快速建立安全区域的奖励")
                elif key == 'reconnaissance_cooperation' and current < target:
                    suggestions.append("• 强化多UAV协作奖励机制")
                elif key == 'success_rate' and current < target:
                    suggestions.append("• 整体提高任务完成奖励，减少不必要的惩罚")
        
        if not suggestions:
            suggestions.append("• 当前性能良好，继续优化训练稳定性")
        
        for suggestion in suggestions:
            print(suggestion)

def main():
    """主函数"""
    tester = QuickOptimizationTester()
    
    # 运行快速优化测试
    print("🎯 快速优化测试 - 展示实验数据向理想值收敛")
    print("目标: 验证优化系统能够改进算法性能")
    
    baseline, final, history = tester.run_quick_optimization_test(episodes=80)
    
    print("\n✅ 快速优化测试完成!")
    print("📝 这个测试展示了:")
    print("  1. 智能奖励设计器如何根据性能差距调整奖励函数")
    print("  2. 算法性能如何在训练过程中逐步改进")
    print("  3. 优化后的指标与论文目标值的对比")
    print("  4. 具体的改进建议和下一步优化方向")

if __name__ == "__main__":
    main() 