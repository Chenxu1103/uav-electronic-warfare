#!/usr/bin/env python3
"""
高级优化测试系统 - 实现真实有效的性能改进

集成增强干扰系统，确保获得真实的干扰率、成功率等关键指标。
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
from enhanced_jamming_system import EnhancedJammingSystem, RealTimePerformanceCalculator

class AdvancedOptimizationTester:
    """高级优化测试器"""
    
    def __init__(self):
        self.reward_designer = IntelligentRewardDesigner()
        self.reward_shaper = RewardShapeOptimizer()
        self.jamming_system = EnhancedJammingSystem()
        self.performance_calculator = RealTimePerformanceCalculator()
        
        # 创建优化的环境
        self.env = self._create_enhanced_environment()
        
        # 性能历史记录
        self.performance_history = []
        
    def _create_enhanced_environment(self):
        """创建增强的环境"""
        env = ElectronicWarfareEnv(
            num_uavs=3,
            num_radars=2,
            env_size=1500.0,
            max_steps=150
        )
        
        # 优化奖励权重 - 强化干扰导向
        env.reward_weights.update({
            'jamming_success': 200.0,           # 大幅增加干扰成功奖励
            'partial_success': 100.0,           # 部分成功奖励
            'jamming_attempt_reward': 50.0,     # 增加尝试干扰奖励
            'approach_reward': 30.0,            # 增加接近奖励
            'coordination_reward': 80.0,        # 增加协调奖励
            'goal_reward': 500.0,               # 目标完成奖励
            'distance_penalty': -0.00001,       # 减小距离惩罚
            'energy_penalty': -0.001,           # 减小能量惩罚
            'detection_penalty': -0.05,         # 减小检测惩罚
            'death_penalty': -50.0,             # 减小死亡惩罚
            'stealth_reward': 10.0,             # 隐身奖励
            'reward_scale': 1.0,                # 奖励缩放
            'min_reward': -20.0,                # 最小奖励
            'max_reward': 300.0,                # 最大奖励
        })
        
        return env
    
    def run_advanced_optimization_test(self, episodes=120):
        """运行高级优化测试"""
        print("🚀 开始高级优化测试...")
        print(f"目标: 在{episodes}个回合内实现真实性能改进")
        
        # 创建优化的AD-PPO智能体
        agent = self._create_enhanced_agent()
        
        # 优化前的基线测试
        print("\n📊 基线性能测试...")
        baseline_metrics = self._evaluate_comprehensive_performance(agent, 15, "基线")
        
        # 开始优化训练
        print(f"\n🎯 开始{episodes}回合高级优化训练...")
        optimized_metrics = self._run_enhanced_training(agent, episodes)
        
        # 最终性能测试
        print("\n📊 优化后性能测试...")
        final_metrics = self._evaluate_comprehensive_performance(agent, 15, "优化后")
        
        # 生成详细对比报告
        self._generate_advanced_report(baseline_metrics, final_metrics, optimized_metrics)
        
        return baseline_metrics, final_metrics, optimized_metrics
    
    def _create_enhanced_agent(self):
        """创建增强的AD-PPO智能体"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        agent = ADPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,      # 增加网络容量
            lr=3e-4,             # 优化学习率
            gamma=0.99,          # 较高折扣因子
            gae_lambda=0.95,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.015,  # 增加探索
            max_grad_norm=0.5,
            device='cpu'
        )
        
        return agent
    
    def _evaluate_comprehensive_performance(self, agent, num_episodes, phase_name):
        """评估综合性能"""
        print(f"  评估{phase_name}性能 ({num_episodes}回合)...")
        
        all_metrics = []
        
        for episode in range(num_episodes):
            episode_data = self._run_comprehensive_episode(agent, evaluation=True)
            
            # 使用增强性能计算器
            metrics = self.performance_calculator.calculate_comprehensive_metrics(
                self.env, episode_data
            )
            
            all_metrics.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        # 打印关键指标
        print(f"    平均奖励: {avg_metrics['average_reward']:.2f}")
        print(f"    成功率: {avg_metrics['success_rate']:.2%}")
        print(f"    干扰率: {avg_metrics['jamming_ratio']:.2%}")
        print(f"    侦察完成度: {avg_metrics['reconnaissance_completion']:.3f}")
        print(f"    安全区域时间: {avg_metrics['safe_zone_development_time']:.2f}")
        print(f"    侦察协作率: {avg_metrics['reconnaissance_cooperation_rate']:.1f}%")
        print(f"    干扰协作率: {avg_metrics['jamming_cooperation_rate']:.1f}%")
        
        return avg_metrics
    
    def _run_comprehensive_episode(self, agent, evaluation=False):
        """运行综合评估回合"""
        state = self.env.reset()
        total_reward = 0
        step = 0
        
        # 记录详细轨迹
        uav_trajectory = []
        jamming_attempts = 0
        successful_jams = 0
        
        while step < self.env.max_steps:
            # 选择动作
            if evaluation:
                action, _, _ = agent.select_action(state, deterministic=True)
            else:
                action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录轨迹
            uav_positions = [uav.position.copy() for uav in self.env.uavs if uav.is_alive]
            uav_trajectory.append(uav_positions)
            
            # 实时干扰评估
            radar_positions = [radar.position for radar in self.env.radars]
            jamming_results = self.jamming_system.evaluate_cooperative_jamming(
                uav_positions, radar_positions
            )
            
            # 更新环境中的雷达状态
            for radar_idx, radar in enumerate(self.env.radars):
                if radar_idx < len(jamming_results['jamming_details']):
                    jamming_data = jamming_results['jamming_details'][radar_idx]
                    radar.is_jammed = jamming_data['is_jammed']
                    if jamming_data['is_jammed']:
                        successful_jams += 1
            
            jamming_attempts += len(uav_positions)
            
            # 训练模式下存储经验
            if not evaluation:
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
        
        # 训练模式下更新模型
        if not evaluation and len(agent.buffer.states) > 0:
            _, _, last_value = agent.select_action(state)
            agent.buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            rollout = agent.buffer.get()
            agent.update(rollout)
            agent.buffer.clear()
        
        return {
            'total_reward': total_reward,
            'steps': step,
            'uav_trajectory': uav_trajectory,
            'jamming_attempts': jamming_attempts,
            'successful_jams': successful_jams
        }
    
    def _run_enhanced_training(self, agent, episodes):
        """运行增强训练"""
        training_metrics = []
        
        for episode in range(episodes):
            # 每15回合评估一次性能并调整奖励
            if episode % 15 == 0 and episode > 0:
                current_metrics = self._evaluate_comprehensive_performance(
                    agent, 8, f"第{episode}回合"
                )
                
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
            self._run_comprehensive_episode(agent, evaluation=False)
            
            # 打印进度
            if episode % 24 == 0:
                print(f"  训练进度: {episode}/{episodes} ({episode/episodes*100:.1f}%)")
        
        return training_metrics
    
    def _generate_advanced_report(self, baseline, final, training_history):
        """生成高级对比报告"""
        print("\n" + "="*80)
        print("📈 高级优化效果报告")
        print("="*80)
        
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
        
        # 详细指标对比
        print(f"{'指标':<30} {'基线值':<15} {'优化后':<15} {'改进幅度':<15} {'论文目标':<15}")
        print("-" * 90)
        
        metrics_info = {
            'average_reward': ('平均奖励', 800.0),
            'success_rate': ('成功率 (%)', 60.0),
            'jamming_ratio': ('干扰率 (%)', 70.0),
            'reconnaissance_completion': ('侦察完成度', 0.97),
            'safe_zone_development_time': ('安全区域时间', 2.1),
            'reconnaissance_cooperation_rate': ('侦察协作率 (%)', 37.0),
            'jamming_cooperation_rate': ('干扰协作率 (%)', 34.0),
            'jamming_failure_rate': ('干扰失效率 (%)', 23.3),
            'overall_effectiveness': ('整体有效性', 0.8)
        }
        
        for key, (display_name, target) in metrics_info.items():
            if key in baseline and key in final:
                baseline_val = baseline[key]
                final_val = final[key]
                improvement = improvements.get(key, 0)
                
                if 'rate' in key or 'ratio' in key or key == 'success_rate':
                    if key == 'jamming_failure_rate':
                        # 失效率越低越好
                        status = "✅" if final_val <= target else "❌"
                    else:
                        status = "✅" if final_val >= target/100 else "❌"
                    print(f"{display_name:<30} {baseline_val:.1%} {'':>6} {final_val:.1%} {'':>6} {improvement:+.1f}% {'':>8} {target:.1f}% {status}")
                else:
                    status = "✅" if final_val >= target else "❌"
                    print(f"{display_name:<30} {baseline_val:.3f} {'':>8} {final_val:.3f} {'':>8} {improvement:+.1f}% {'':>8} {target:.1f} {status}")
        
        print("\n" + "="*80)
        
        # 关键成就
        print("🏆 关键成就:")
        achievements = []
        
        if final['jamming_ratio'] > 0.3:
            achievements.append(f"• 实现 {final['jamming_ratio']:.1%} 干扰率 (基线: {baseline['jamming_ratio']:.1%})")
        
        if final['success_rate'] > 0.2:
            achievements.append(f"• 达到 {final['success_rate']:.1%} 成功率 (基线: {baseline['success_rate']:.1%})")
        
        if final['safe_zone_development_time'] > 0.5:
            achievements.append(f"• 建立 {final['safe_zone_development_time']:.1f}s 安全区域 (基线: {baseline['safe_zone_development_time']:.1f}s)")
        
        if final['jamming_cooperation_rate'] > baseline['jamming_cooperation_rate']:
            achievements.append(f"• 干扰协作率提升至 {final['jamming_cooperation_rate']:.1f}%")
        
        avg_reward_improvement = improvements.get('average_reward', 0)
        if avg_reward_improvement > 50:
            achievements.append(f"• 平均奖励提升 {avg_reward_improvement:.1f}%")
        
        for achievement in achievements:
            print(achievement)
        
        if not achievements:
            print("• 尚未实现显著突破，需要进一步优化")
        
        # 距离论文目标的差距分析
        print("\n📊 与论文目标差距分析:")
        print("-" * 50)
        
        paper_targets = {
            'reconnaissance_completion': 0.97,
            'safe_zone_development_time': 2.1,
            'reconnaissance_cooperation_rate': 37.0,
            'jamming_cooperation_rate': 34.0,
            'jamming_failure_rate': 23.3
        }
        
        total_gap = 0
        target_count = 0
        
        for key, target in paper_targets.items():
            if key in final:
                current = final[key]
                if key == 'jamming_failure_rate':
                    gap = max(0, current - target) / target * 100
                elif key in ['reconnaissance_cooperation_rate', 'jamming_cooperation_rate']:
                    gap = abs(current - target) / target * 100
                else:
                    gap = abs(current - target) / target * 100
                
                total_gap += gap
                target_count += 1
                
                status = "✅" if gap < 30 else "⚠️" if gap < 60 else "❌"
                print(f"  {key}: 差距 {gap:.1f}% {status}")
        
        avg_gap = total_gap / max(1, target_count)
        print(f"\n平均差距: {avg_gap:.1f}%")
        
        if avg_gap < 25:
            print("🎉 非常接近论文目标！")
        elif avg_gap < 50:
            print("👍 较好地接近论文目标")
        else:
            print("🔧 需要进一步优化以接近论文目标")
        
        # 优化建议
        print("\n💡 下一步优化建议:")
        self._generate_specific_recommendations(final, paper_targets)
        
        # 保存结果
        self._save_advanced_results(baseline, final, improvements, training_history)
    
    def _generate_specific_recommendations(self, current_metrics, targets):
        """生成具体优化建议"""
        recommendations = []
        
        if current_metrics['jamming_ratio'] < 0.5:
            recommendations.append("• 增强干扰机制: 提高UAV干扰功率，优化干扰距离计算")
        
        if current_metrics['success_rate'] < 0.4:
            recommendations.append("• 改进任务成功判定: 降低成功阈值，增加渐进式奖励")
        
        if current_metrics['safe_zone_development_time'] < 1.0:
            recommendations.append("• 强化安全区域建立: 增加持续干扰奖励，优化多UAV协作")
        
        if current_metrics['jamming_cooperation_rate'] < 25:
            recommendations.append("• 提升协作效率: 实现智能任务分配，优化UAV间通信")
        
        if current_metrics['reconnaissance_cooperation_rate'] < 30:
            recommendations.append("• 优化侦察协作: 改进编队控制，增加协作探索奖励")
        
        if not recommendations:
            recommendations.append("• 继续精细调优现有机制，保持性能稳定性")
        
        for rec in recommendations:
            print(rec)
    
    def _save_advanced_results(self, baseline, final, improvements, training_history):
        """保存高级结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"experiments/advanced_optimization/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 转换numpy类型
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
        
        results_data = {
            'baseline': convert_numpy_types(baseline),
            'final': convert_numpy_types(final),
            'improvements': convert_numpy_types(improvements),
            'training_history': convert_numpy_types(training_history),
            'test_timestamp': timestamp
        }
        
        import json
        with open(os.path.join(save_dir, 'advanced_optimization_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {save_dir}")

def main():
    """主函数"""
    tester = AdvancedOptimizationTester()
    
    # 运行高级优化测试
    print("🎯 高级优化测试 - 实现真实性能改进")
    print("目标: 获得真实有效的干扰率、成功率等关键指标")
    
    baseline, final, history = tester.run_advanced_optimization_test(episodes=120)
    
    print("\n✅ 高级优化测试完成!")
    print("📝 这个测试实现了:")
    print("  1. 真实有效的干扰机制和性能指标计算")
    print("  2. 智能奖励设计器的自适应调整")
    print("  3. 综合性能评估和详细改进分析")
    print("  4. 与论文目标的精确对比和优化建议")

if __name__ == "__main__":
    main() 