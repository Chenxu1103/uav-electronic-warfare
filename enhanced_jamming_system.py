#!/usr/bin/env python3
"""
增强干扰系统 - 实现真实有效的雷达干扰机制

解决干扰率为0%的核心问题，确保UAV能够有效干扰雷达。
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional

class EnhancedJammingSystem:
    """增强的干扰系统"""
    
    def __init__(self):
        # 干扰参数配置
        self.jamming_config = {
            'max_range': 800.0,              # 最大干扰距离
            'optimal_range': 300.0,          # 最佳干扰距离
            'min_range': 50.0,               # 最小安全距离
            'power_threshold': 0.6,          # 干扰功率阈值
            'cooperation_bonus': 0.3,        # 协作干扰加成
            'angle_factor': 0.8,             # 角度影响因子
            'duration_threshold': 3,         # 持续干扰时间阈值
        }
        
        # 干扰状态追踪
        self.jamming_history = {}
        self.radar_status = {}
        self.cooperation_records = []
        
    def calculate_jamming_effectiveness(self, uav_position: np.ndarray, 
                                      radar_position: np.ndarray,
                                      uav_power: float = 1.0,
                                      radar_power: float = 1.0) -> Dict:
        """
        计算干扰有效性
        
        Args:
            uav_position: UAV位置
            radar_position: 雷达位置
            uav_power: UAV发射功率
            radar_power: 雷达功率
            
        Returns:
            dict: 干扰效果信息
        """
        # 计算距离
        distance = np.linalg.norm(uav_position - radar_position)
        
        # 距离效应计算
        distance_factor = self._calculate_distance_factor(distance)
        
        # 功率比计算
        power_ratio = uav_power / (radar_power + 0.1)  # 避免除零
        
        # 角度效应（简化，假设最优）
        angle_factor = self.jamming_config['angle_factor']
        
        # 综合干扰效果
        jamming_power = distance_factor * power_ratio * angle_factor
        
        # 判断是否成功干扰
        is_effective = jamming_power >= self.jamming_config['power_threshold']
        
        return {
            'distance': distance,
            'distance_factor': distance_factor,
            'power_ratio': power_ratio,
            'jamming_power': jamming_power,
            'is_effective': is_effective,
            'effectiveness_score': min(jamming_power, 1.0)
        }
    
    def _calculate_distance_factor(self, distance: float) -> float:
        """计算距离因子"""
        max_range = self.jamming_config['max_range']
        optimal_range = self.jamming_config['optimal_range']
        min_range = self.jamming_config['min_range']
        
        if distance <= min_range:
            return 0.3  # 太近反而效果不好
        elif distance <= optimal_range:
            # 在最佳范围内
            return 1.0
        elif distance <= max_range:
            # 距离衰减
            decay = (max_range - distance) / (max_range - optimal_range)
            return max(0.1, decay)
        else:
            return 0.0  # 超出范围
    
    def evaluate_cooperative_jamming(self, uav_positions: List[np.ndarray], 
                                   radar_positions: List[np.ndarray]) -> Dict:
        """
        评估协作干扰效果
        
        Args:
            uav_positions: UAV位置列表
            radar_positions: 雷达位置列表
            
        Returns:
            dict: 协作干扰评估结果
        """
        results = {
            'total_jammers': len(uav_positions),
            'total_radars': len(radar_positions),
            'jammed_radars': 0,
            'jamming_details': [],
            'cooperation_score': 0.0,
            'overall_effectiveness': 0.0
        }
        
        radar_jamming_status = {}
        
        # 对每个雷达评估干扰效果
        for radar_idx, radar_pos in enumerate(radar_positions):
            radar_jammers = []
            total_jamming_power = 0.0
            
            # 计算所有UAV对该雷达的干扰效果
            for uav_idx, uav_pos in enumerate(uav_positions):
                jamming_result = self.calculate_jamming_effectiveness(
                    uav_pos, radar_pos
                )
                
                if jamming_result['is_effective']:
                    radar_jammers.append({
                        'uav_idx': uav_idx,
                        'effectiveness': jamming_result['effectiveness_score']
                    })
                    total_jamming_power += jamming_result['jamming_power']
            
            # 协作加成
            if len(radar_jammers) > 1:
                cooperation_bonus = self.jamming_config['cooperation_bonus']
                total_jamming_power *= (1 + cooperation_bonus)
            
            # 判断雷达是否被成功干扰
            is_jammed = total_jamming_power >= self.jamming_config['power_threshold']
            
            radar_jamming_status[radar_idx] = {
                'is_jammed': is_jammed,
                'jamming_power': total_jamming_power,
                'jammers': radar_jammers,
                'cooperation_count': len(radar_jammers)
            }
            
            if is_jammed:
                results['jammed_radars'] += 1
            
            results['jamming_details'].append(radar_jamming_status[radar_idx])
        
        # 计算整体指标
        results['jamming_ratio'] = results['jammed_radars'] / max(1, results['total_radars'])
        results['cooperation_score'] = self._calculate_cooperation_score(radar_jamming_status)
        results['overall_effectiveness'] = self._calculate_overall_effectiveness(radar_jamming_status)
        
        return results
    
    def _calculate_cooperation_score(self, radar_status: Dict) -> float:
        """计算协作分数"""
        cooperation_instances = 0
        total_instances = 0
        
        for radar_data in radar_status.values():
            total_instances += 1
            if radar_data['cooperation_count'] > 1:
                cooperation_instances += 1
        
        return cooperation_instances / max(1, total_instances) * 100
    
    def _calculate_overall_effectiveness(self, radar_status: Dict) -> float:
        """计算整体有效性"""
        total_power = 0.0
        radar_count = len(radar_status)
        
        for radar_data in radar_status.values():
            total_power += min(radar_data['jamming_power'], 1.0)
        
        return total_power / max(1, radar_count)
    
    def update_jamming_history(self, episode: int, jamming_results: Dict):
        """更新干扰历史记录"""
        self.jamming_history[episode] = jamming_results
    
    def get_jamming_statistics(self, window_size: int = 50) -> Dict:
        """获取干扰统计信息"""
        if not self.jamming_history:
            return {'status': 'no_data'}
        
        recent_episodes = list(self.jamming_history.keys())[-window_size:]
        
        jamming_ratios = []
        cooperation_scores = []
        effectiveness_scores = []
        
        for episode in recent_episodes:
            data = self.jamming_history[episode]
            jamming_ratios.append(data.get('jamming_ratio', 0))
            cooperation_scores.append(data.get('cooperation_score', 0))
            effectiveness_scores.append(data.get('overall_effectiveness', 0))
        
        return {
            'avg_jamming_ratio': np.mean(jamming_ratios),
            'avg_cooperation_score': np.mean(cooperation_scores),
            'avg_effectiveness': np.mean(effectiveness_scores),
            'improvement_trend': self._calculate_trend(jamming_ratios),
            'total_episodes': len(recent_episodes)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

class EnhancedRadarModel:
    """增强的雷达模型"""
    
    def __init__(self, position: np.ndarray, detection_range: float = 1000.0):
        self.position = position
        self.detection_range = detection_range
        self.power = 1.0
        self.is_jammed = False
        self.jamming_level = 0.0
        self.jamming_duration = 0
        
    def update_jamming_status(self, jamming_power: float):
        """更新干扰状态"""
        threshold = 0.6
        
        if jamming_power >= threshold:
            self.is_jammed = True
            self.jamming_level = min(jamming_power, 1.0)
            self.jamming_duration += 1
        else:
            self.is_jammed = False
            self.jamming_level = 0.0
            self.jamming_duration = max(0, self.jamming_duration - 1)
    
    def get_detection_capability(self) -> float:
        """获取当前检测能力"""
        if self.is_jammed:
            return max(0.1, 1.0 - self.jamming_level)
        return 1.0

class RealTimePerformanceCalculator:
    """实时性能计算器"""
    
    def __init__(self):
        self.jamming_system = EnhancedJammingSystem()
        
    def calculate_comprehensive_metrics(self, env, episode_data: Dict) -> Dict:
        """计算综合性能指标"""
        # 基础数据
        uav_positions = [uav.position for uav in env.uavs if uav.is_alive]
        radar_positions = [radar.position for radar in env.radars]
        
        # 干扰效果评估
        jamming_results = self.jamming_system.evaluate_cooperative_jamming(
            uav_positions, radar_positions
        )
        
        # 计算各项指标
        metrics = {
            # 1. 侦察任务完成度
            'reconnaissance_completion': self._calculate_reconnaissance_completion(
                env, episode_data
            ),
            
            # 2. 安全区域开辟时间
            'safe_zone_development_time': self._calculate_safe_zone_time(
                jamming_results, episode_data
            ),
            
            # 3. 侦察协作率
            'reconnaissance_cooperation_rate': self._calculate_reconnaissance_cooperation(
                uav_positions
            ),
            
            # 4. 干扰协作率
            'jamming_cooperation_rate': jamming_results['cooperation_score'],
            
            # 5. 干扰动作失效率
            'jamming_failure_rate': self._calculate_jamming_failure_rate(
                jamming_results
            ),
            
            # 附加指标
            'success_rate': jamming_results['jamming_ratio'],
            'jamming_ratio': jamming_results['jamming_ratio'],
            'average_reward': episode_data.get('total_reward', 0),
            'overall_effectiveness': jamming_results['overall_effectiveness']
        }
        
        return metrics
    
    def _calculate_reconnaissance_completion(self, env, episode_data: Dict) -> float:
        """计算侦察任务完成度"""
        max_steps = env.max_steps
        actual_steps = episode_data.get('steps', max_steps)
        
        # 基于步数和存活UAV的完成度
        step_completion = min(actual_steps / max_steps, 1.0)
        
        # 存活率因子
        alive_uavs = sum(1 for uav in env.uavs if uav.is_alive)
        survival_factor = alive_uavs / len(env.uavs)
        
        # 目标接近度
        proximity_score = self._calculate_proximity_score(env)
        
        # 综合完成度
        completion = 0.4 * step_completion + 0.3 * survival_factor + 0.3 * proximity_score
        
        return min(completion, 1.0)
    
    def _calculate_proximity_score(self, env) -> float:
        """计算目标接近度分数"""
        if not env.uavs or not env.radars:
            return 0.0
        
        total_score = 0.0
        valid_pairs = 0
        
        for uav in env.uavs:
            if not uav.is_alive:
                continue
                
            min_distance = float('inf')
            for radar in env.radars:
                distance = np.linalg.norm(uav.position - radar.position)
                min_distance = min(min_distance, distance)
            
            # 距离越近分数越高
            if min_distance < 1000:  # 有效范围内
                score = max(0, 1 - min_distance / 1000)
                total_score += score
                valid_pairs += 1
        
        return total_score / max(1, valid_pairs)
    
    def _calculate_safe_zone_time(self, jamming_results: Dict, episode_data: Dict) -> float:
        """计算安全区域开辟时间"""
        jamming_ratio = jamming_results['jamming_ratio']
        
        if jamming_ratio >= 0.5:  # 至少50%雷达被干扰
            # 根据干扰效果计算安全区域时间
            effectiveness = jamming_results['overall_effectiveness']
            base_time = 1.0
            
            # 效果越好，安全区域建立时间越长
            safe_zone_time = base_time * (1 + effectiveness * 2)
            return min(safe_zone_time, 3.0)  # 最大3.0
        
        return 0.0
    
    def _calculate_reconnaissance_cooperation(self, uav_positions: List[np.ndarray]) -> float:
        """计算侦察协作率"""
        if len(uav_positions) < 2:
            return 0.0
        
        cooperation_instances = 0
        total_pairs = 0
        
        # 理想协作距离范围
        ideal_min = 200
        ideal_max = 600
        
        for i in range(len(uav_positions)):
            for j in range(i + 1, len(uav_positions)):
                distance = np.linalg.norm(uav_positions[i] - uav_positions[j])
                total_pairs += 1
                
                if ideal_min <= distance <= ideal_max:
                    cooperation_instances += 1
        
        return (cooperation_instances / max(1, total_pairs)) * 100
    
    def _calculate_jamming_failure_rate(self, jamming_results: Dict) -> float:
        """计算干扰动作失效率"""
        total_attempts = 0
        failed_attempts = 0
        
        for radar_data in jamming_results['jamming_details']:
            jammers = radar_data['jammers']
            total_attempts += len(jammers)
            
            # 统计失效的干扰尝试
            for jammer in jammers:
                if jammer['effectiveness'] < 0.7:  # 效果不佳视为失效
                    failed_attempts += 1
        
        if total_attempts == 0:
            return 100.0  # 没有尝试干扰，失效率100%
        
        return (failed_attempts / total_attempts) * 100

if __name__ == "__main__":
    # 测试增强干扰系统
    jamming_system = EnhancedJammingSystem()
    
    # 模拟UAV和雷达位置
    uav_positions = [
        np.array([100, 100, 50]),
        np.array([300, 200, 50]),
        np.array([500, 300, 50])
    ]
    
    radar_positions = [
        np.array([250, 150, 0]),
        np.array([400, 250, 0])
    ]
    
    # 评估协作干扰
    results = jamming_system.evaluate_cooperative_jamming(uav_positions, radar_positions)
    
    print("🎯 增强干扰系统测试结果:")
    print(f"干扰率: {results['jamming_ratio']:.2%}")
    print(f"协作分数: {results['cooperation_score']:.1f}%")
    print(f"整体有效性: {results['overall_effectiveness']:.3f}")
    print(f"被干扰雷达数: {results['jammed_radars']}/{results['total_radars']}")
    
    # 测试性能计算器
    calculator = RealTimePerformanceCalculator()
    
    # 模拟环境数据
    class MockEnv:
        def __init__(self):
            self.max_steps = 200
            self.uavs = [MockUAV() for _ in range(3)]
            self.radars = [MockRadar(pos) for pos in radar_positions]
    
    class MockUAV:
        def __init__(self):
            self.position = np.random.uniform([0, 0, 0], [500, 500, 100])
            self.is_alive = True
    
    class MockRadar:
        def __init__(self, position):
            self.position = position
    
    env = MockEnv()
    episode_data = {'steps': 180, 'total_reward': 800}
    
    metrics = calculator.calculate_comprehensive_metrics(env, episode_data)
    
    print("\n📊 综合性能指标:")
    for key, value in metrics.items():
        if 'rate' in key or 'ratio' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.3f}")
    
    print("\n✅ 增强干扰系统测试完成!") 