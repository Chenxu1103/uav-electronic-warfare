#!/usr/bin/env python3
"""
智能奖励设计器 - 动态优化奖励函数

根据当前训练性能自动调整奖励权重，引导算法朝向论文目标性能发展。
"""

import numpy as np
import json
from datetime import datetime
import os

class IntelligentRewardDesigner:
    """智能奖励设计器"""
    
    def __init__(self, target_metrics=None):
        """
        初始化智能奖励设计器
        
        Args:
            target_metrics: 目标性能指标字典
        """
        self.target_metrics = target_metrics or {
            'reconnaissance_completion': 0.90,
            'safe_zone_time': 2.0,
            'reconnaissance_cooperation': 35.0,
            'jamming_cooperation': 30.0,
            'jamming_failure_rate': 25.0,
            'success_rate': 0.6,
            'average_reward': 800.0
        }
        
        # 基础奖励权重
        self.base_weights = {
            'jamming_success': 100.0,
            'partial_success': 50.0,
            'distance_penalty': -0.00005,
            'energy_penalty': -0.005,
            'detection_penalty': -0.1,
            'death_penalty': -1.0,
            'goal_reward': 1000.0,
            'coordination_reward': 50.0,
            'stealth_reward': 1.0,
            'approach_reward': 15.0,
            'jamming_attempt_reward': 8.0,
            'reward_scale': 0.8,
            'min_reward': -10.0,
            'max_reward': 150.0,
        }
        
        # 权重调整历史
        self.adjustment_history = []
        
        # 性能改进追踪
        self.performance_buffer = []
        self.buffer_size = 100
        
    def analyze_performance_gap(self, current_metrics):
        """
        分析当前性能与目标性能的差距
        
        Args:
            current_metrics: 当前性能指标字典
            
        Returns:
            dict: 性能差距分析结果
        """
        gaps = {}
        priorities = {}
        
        for metric, target in self.target_metrics.items():
            if metric in current_metrics:
                current = current_metrics[metric]
                
                if metric == 'jamming_failure_rate':
                    # 失效率越低越好
                    gap = current - target
                    normalized_gap = gap / target if target > 0 else 0
                else:
                    # 其他指标越高越好
                    gap = target - current
                    normalized_gap = gap / target if target > 0 else 0
                
                gaps[metric] = {
                    'absolute_gap': gap,
                    'relative_gap': normalized_gap,
                    'current': current,
                    'target': target
                }
                
                # 计算优先级（差距越大优先级越高）
                priorities[metric] = abs(normalized_gap)
        
        return {
            'gaps': gaps,
            'priorities': priorities,
            'most_critical': max(priorities.items(), key=lambda x: x[1]) if priorities else None
        }
    
    def design_adaptive_rewards(self, env, current_metrics, episode_count):
        """
        设计自适应奖励函数
        
        Args:
            env: 环境对象
            current_metrics: 当前性能指标
            episode_count: 当前训练回合数
            
        Returns:
            dict: 调整后的奖励权重
        """
        print(f"\n🎯 第{episode_count}回合 - 自适应奖励设计中...")
        
        # 分析性能差距
        analysis = self.analyze_performance_gap(current_metrics)
        
        if not analysis['most_critical']:
            return env.reward_weights
        
        most_critical_metric, critical_gap = analysis['most_critical']
        print(f"最关键指标: {most_critical_metric} (差距: {critical_gap:.3f})")
        
        # 根据关键指标调整奖励
        new_weights = self._adjust_weights_for_metric(
            env.reward_weights.copy(), 
            most_critical_metric, 
            analysis['gaps'][most_critical_metric],
            episode_count
        )
        
        # 应用平滑调整
        smoothed_weights = self._smooth_weight_changes(env.reward_weights, new_weights)
        
        # 记录调整历史
        self._record_adjustment(current_metrics, smoothed_weights, most_critical_metric)
        
        return smoothed_weights
    
    def _adjust_weights_for_metric(self, current_weights, metric, gap_info, episode_count):
        """根据特定指标调整权重"""
        adjustment_strength = min(abs(gap_info['relative_gap']) * 0.3, 0.5)  # 限制调整幅度
        
        if metric == 'reconnaissance_completion':
            # 提高侦察完成度
            if gap_info['current'] < gap_info['target']:
                current_weights['approach_reward'] *= (1 + adjustment_strength)
                current_weights['stealth_reward'] *= (1 + adjustment_strength)
                current_weights['detection_penalty'] *= (1 - adjustment_strength * 0.5)
                
        elif metric == 'safe_zone_time':
            # 提高安全区域建立速度
            if gap_info['current'] < gap_info['target']:
                current_weights['jamming_success'] *= (1 + adjustment_strength)
                current_weights['jamming_attempt_reward'] *= (1 + adjustment_strength)
                current_weights['goal_reward'] *= (1 + adjustment_strength * 0.5)
                
        elif metric == 'reconnaissance_cooperation':
            # 提高侦察协作率
            if gap_info['current'] < gap_info['target']:
                current_weights['coordination_reward'] *= (1 + adjustment_strength)
                current_weights['approach_reward'] *= (1 + adjustment_strength * 0.5)
                
        elif metric == 'jamming_cooperation':
            # 提高干扰协作率
            if gap_info['current'] < gap_info['target']:
                current_weights['coordination_reward'] *= (1 + adjustment_strength)
                current_weights['jamming_success'] *= (1 + adjustment_strength)
                current_weights['partial_success'] *= (1 + adjustment_strength)
                
        elif metric == 'jamming_failure_rate':
            # 降低干扰失效率
            if gap_info['current'] > gap_info['target']:
                current_weights['jamming_attempt_reward'] *= (1 + adjustment_strength)
                current_weights['approach_reward'] *= (1 + adjustment_strength)
                current_weights['distance_penalty'] *= (1 - adjustment_strength * 0.3)
                
        elif metric == 'success_rate':
            # 提高整体成功率
            if gap_info['current'] < gap_info['target']:
                current_weights['goal_reward'] *= (1 + adjustment_strength)
                current_weights['jamming_success'] *= (1 + adjustment_strength)
                current_weights['death_penalty'] *= (1 - adjustment_strength * 0.5)
                
        elif metric == 'average_reward':
            # 提高平均奖励
            if gap_info['current'] < gap_info['target']:
                current_weights['reward_scale'] *= (1 + adjustment_strength * 0.3)
                current_weights['max_reward'] *= (1 + adjustment_strength * 0.2)
                # 减少惩罚
                for key in ['distance_penalty', 'energy_penalty', 'detection_penalty']:
                    if current_weights[key] < 0:
                        current_weights[key] *= (1 - adjustment_strength * 0.3)
        
        return current_weights
    
    def _smooth_weight_changes(self, old_weights, new_weights, smoothing_factor=0.7):
        """平滑权重变化，避免剧烈调整"""
        smoothed = {}
        
        for key in old_weights:
            if key in new_weights:
                # 使用指数平滑
                smoothed[key] = old_weights[key] * smoothing_factor + new_weights[key] * (1 - smoothing_factor)
                
                # 限制变化幅度
                max_change = abs(old_weights[key]) * 0.2  # 最大变化20%
                change = smoothed[key] - old_weights[key]
                if abs(change) > max_change:
                    smoothed[key] = old_weights[key] + np.sign(change) * max_change
            else:
                smoothed[key] = old_weights[key]
        
        return smoothed
    
    def _record_adjustment(self, metrics, new_weights, critical_metric):
        """记录调整历史"""
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy(),
            'new_weights': new_weights.copy(),
            'critical_metric': critical_metric
        }
        
        self.adjustment_history.append(adjustment_record)
        
        # 保持历史记录在合理范围内
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
    
    def get_progressive_curriculum_weights(self, episode_count, total_episodes):
        """
        获取渐进式课程学习的权重
        
        Args:
            episode_count: 当前回合数
            total_episodes: 总回合数
            
        Returns:
            dict: 课程调整后的权重
        """
        progress = episode_count / total_episodes
        weights = self.base_weights.copy()
        
        # 早期训练：更多探索奖励，较少惩罚
        if progress < 0.3:
            weights['jamming_attempt_reward'] *= 2.0  # 鼓励尝试
            weights['approach_reward'] *= 1.5
            weights['distance_penalty'] *= 0.5  # 减少探索惩罚
            weights['energy_penalty'] *= 0.5
            
        # 中期训练：平衡探索和利用
        elif progress < 0.7:
            weights['coordination_reward'] *= 1.5  # 增加协作奖励
            weights['partial_success'] *= 1.3
            
        # 后期训练：更注重精确控制和高质量完成
        else:
            weights['goal_reward'] *= 1.5  # 更重视目标完成
            weights['jamming_success'] *= 1.3
            weights['reward_scale'] *= 1.2  # 放大奖励信号
        
        return weights
    
    def evaluate_reward_effectiveness(self, recent_performance, window_size=50):
        """
        评估奖励设计的有效性
        
        Args:
            recent_performance: 最近的性能数据列表
            window_size: 评估窗口大小
            
        Returns:
            dict: 奖励有效性评估结果
        """
        if len(recent_performance) < window_size:
            return {'status': 'insufficient_data'}
        
        # 获取最近的性能数据
        recent_data = recent_performance[-window_size:]
        
        # 计算趋势
        trends = {}
        for metric in self.target_metrics:
            if metric in recent_data[0]:
                values = [data[metric] for data in recent_data if metric in data]
                if len(values) >= 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    trends[metric] = trend
        
        # 评估改进速度
        improvement_rate = np.mean([abs(trend) for trend in trends.values() if not np.isnan(trend)])
        
        # 计算目标接近度
        if recent_data:
            latest_metrics = recent_data[-1]
            approach_scores = []
            
            for metric, target in self.target_metrics.items():
                if metric in latest_metrics:
                    current = latest_metrics[metric]
                    if metric == 'jamming_failure_rate':
                        score = max(0, 1 - abs(current - target) / target)
                    else:
                        score = max(0, 1 - abs(current - target) / target)
                    approach_scores.append(score)
            
            average_approach = np.mean(approach_scores) if approach_scores else 0
        else:
            average_approach = 0
        
        return {
            'status': 'evaluated',
            'trends': trends,
            'improvement_rate': improvement_rate,
            'target_approach_score': average_approach,
            'recommendation': self._get_adjustment_recommendation(trends, average_approach)
        }
    
    def _get_adjustment_recommendation(self, trends, approach_score):
        """获取调整建议"""
        if approach_score > 0.8:
            return "性能接近目标，继续当前策略"
        elif approach_score > 0.6:
            return "性能良好，可微调奖励权重"
        elif approach_score > 0.4:
            return "性能一般，需要调整奖励策略"
        else:
            return "性能较差，需要大幅调整奖励设计"
    
    def save_design_history(self, save_path):
        """保存奖励设计历史"""
        history_data = {
            'target_metrics': self.target_metrics,
            'base_weights': self.base_weights,
            'adjustment_history': self.adjustment_history,
            'total_adjustments': len(self.adjustment_history)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 奖励设计历史已保存到: {save_path}")

class RewardShapeOptimizer:
    """奖励塑形优化器"""
    
    def __init__(self):
        self.shaping_functions = {
            'exponential_decay': self._exponential_decay_shaping,
            'sigmoid_shaping': self._sigmoid_shaping,
            'linear_interpolation': self._linear_interpolation_shaping,
            'threshold_based': self._threshold_based_shaping
        }
    
    def _exponential_decay_shaping(self, distance, max_distance, decay_rate=0.1):
        """指数衰减奖励塑形"""
        normalized_distance = distance / max_distance
        return np.exp(-decay_rate * normalized_distance)
    
    def _sigmoid_shaping(self, value, midpoint, steepness=1.0):
        """Sigmoid奖励塑形"""
        return 1 / (1 + np.exp(-steepness * (value - midpoint)))
    
    def _linear_interpolation_shaping(self, value, min_val, max_val):
        """线性插值奖励塑形"""
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _threshold_based_shaping(self, value, thresholds, rewards):
        """基于阈值的奖励塑形"""
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return rewards[i]
        return rewards[-1]
    
    def optimize_distance_reward(self, uav_position, radar_position, max_range=1000):
        """优化距离相关奖励"""
        distance = np.linalg.norm(uav_position - radar_position)
        
        # 使用多阶段奖励
        if distance < 200:  # 近距离 - 高奖励
            return 1.0
        elif distance < 500:  # 中距离 - 中等奖励
            return self._linear_interpolation_shaping(distance, 200, 500) * 0.7 + 0.3
        elif distance < max_range:  # 远距离 - 低奖励
            return self._exponential_decay_shaping(distance, max_range, 0.002)
        else:  # 超出范围 - 无奖励
            return 0.0
    
    def optimize_cooperation_reward(self, uav_positions, cooperation_threshold=800):
        """优化协作奖励"""
        if len(uav_positions) < 2:
            return 0.0
        
        cooperation_score = 0.0
        pair_count = 0
        
        for i in range(len(uav_positions)):
            for j in range(i + 1, len(uav_positions)):
                distance = np.linalg.norm(uav_positions[i] - uav_positions[j])
                
                # 理想协作距离奖励
                if 200 <= distance <= cooperation_threshold:
                    normalized_distance = distance / cooperation_threshold
                    score = self._sigmoid_shaping(normalized_distance, 0.5, 5.0)
                    cooperation_score += score
                
                pair_count += 1
        
        return cooperation_score / pair_count if pair_count > 0 else 0.0

if __name__ == "__main__":
    # 测试智能奖励设计器
    designer = IntelligentRewardDesigner()
    
    # 模拟性能数据
    test_metrics = {
        'reconnaissance_completion': 0.45,
        'safe_zone_time': 0.8,
        'reconnaissance_cooperation': 20.0,
        'jamming_cooperation': 15.0,
        'jamming_failure_rate': 35.0,
        'success_rate': 0.2,
        'average_reward': 400.0
    }
    
    # 分析性能差距
    analysis = designer.analyze_performance_gap(test_metrics)
    print("性能差距分析:", analysis)
    
    # 测试奖励塑形优化器
    shaper = RewardShapeOptimizer()
    
    # 测试距离奖励
    distance_reward = shaper.optimize_distance_reward(
        np.array([100, 100, 0]), 
        np.array([300, 300, 0])
    )
    print(f"距离奖励: {distance_reward:.3f}")
    
    # 测试协作奖励
    positions = [np.array([0, 0, 0]), np.array([400, 400, 0]), np.array([800, 800, 0])]
    coop_reward = shaper.optimize_cooperation_reward(positions)
    print(f"协作奖励: {coop_reward:.3f}")
    
    print("✅ 智能奖励设计器测试完成!") 