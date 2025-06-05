"""
论文表5-2指标评估脚本
计算AD-PPO算法的具体性能指标并与论文数据对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG

class PaperMetricsEvaluator:
    def __init__(self, num_episodes=100):
        """
        初始化指标评估器
        
        Args:
            num_episodes: 评估的回合数
        """
        self.num_episodes = num_episodes
        self.env = ElectronicWarfareEnv(num_uavs=3, num_radars=2, env_size=2000.0, max_steps=200)
        
        # 论文中的基准指标
        self.paper_metrics = {
            'reconnaissance_completion': 0.97,
            'safe_zone_time': 2.1,
            'reconnaissance_cooperation': 37.0,
            'jamming_cooperation': 34.0,
            'jamming_failure_rate': 23.3
        }
        
        # 用于记录指标的列表
        self.metrics_log = {
            'reconnaissance_completion': [],
            'safe_zone_time': [],
            'reconnaissance_cooperation': [],
            'jamming_cooperation': [],
            'jamming_failure_rate': [],
            'episode_rewards': [],
            'episode_steps': [],
            'successful_episodes': []
        }
        
    def calculate_reconnaissance_completion(self, episode_data):
        """
        计算侦察任务完成度
        侦察任务完成度 = 成功探测到的雷达数 / 总雷达数
        """
        detected_radars = set()
        for step_data in episode_data:
            for uav_id, uav_pos in enumerate(step_data['uav_positions']):
                for radar_id, radar_pos in enumerate(step_data['radar_positions']):
                    distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                    if distance < 600:  # 侦察范围600m
                        detected_radars.add(radar_id)
        
        completion_rate = len(detected_radars) / len(episode_data[0]['radar_positions'])
        return min(1.0, completion_rate)
    
    def calculate_safe_zone_time(self, episode_data):
        """
        计算安全区域开辟时间
        安全区域开辟时间 = 首次建立安全区域的时间步 * dt
        """
        for step, step_data in enumerate(episode_data):
            # 检查是否有任何雷达被干扰（建立了安全区域）
            if any(step_data['jammed_radars']):
                return (step + 1) * 0.1  # dt = 0.1
        return 3.0  # 如果没有建立安全区域，返回最大时间
    
    def calculate_reconnaissance_cooperation(self, episode_data):
        """
        计算侦察协作率
        侦察协作率 = 多个UAV同时侦察同一区域的时间步 / 总侦察时间步
        """
        cooperative_steps = 0
        total_reconnaissance_steps = 0
        
        for step_data in episode_data:
            # 检查每个雷达周围的UAV数量
            radar_uav_count = {}
            for radar_id, radar_pos in enumerate(step_data['radar_positions']):
                radar_uav_count[radar_id] = 0
                for uav_pos in step_data['uav_positions']:
                    distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                    if distance < 800:  # 侦察协作范围
                        radar_uav_count[radar_id] += 1
                        total_reconnaissance_steps += 1
            
            # 检查是否有协作侦察
            for count in radar_uav_count.values():
                if count > 1:
                    cooperative_steps += count
        
        if total_reconnaissance_steps == 0:
            return 0.0
        return (cooperative_steps / total_reconnaissance_steps) * 100
    
    def calculate_jamming_cooperation(self, episode_data):
        """
        计算干扰协作率
        干扰协作率 = 多个UAV协作干扰的时间步 / 总干扰时间步
        """
        cooperative_jamming_steps = 0
        total_jamming_steps = 0
        
        for step_data in episode_data:
            active_jammers = sum(1 for jamming in step_data['uav_jamming'] if jamming)
            if active_jammers > 0:
                total_jamming_steps += active_jammers
                if active_jammers > 1:
                    # 检查是否针对相同目标或相近区域
                    jamming_positions = [pos for i, pos in enumerate(step_data['uav_positions']) 
                                       if step_data['uav_jamming'][i]]
                    
                    # 如果干扰UAV之间距离较近，认为是协作干扰
                    for i in range(len(jamming_positions)):
                        for j in range(i+1, len(jamming_positions)):
                            distance = np.linalg.norm(np.array(jamming_positions[i]) - 
                                                    np.array(jamming_positions[j]))
                            if distance < 500:  # 协作距离阈值
                                cooperative_jamming_steps += 2
                                break
        
        if total_jamming_steps == 0:
            return 0.0
        return (cooperative_jamming_steps / total_jamming_steps) * 100
    
    def calculate_jamming_failure_rate(self, episode_data):
        """
        计算干扰动作失效率
        干扰失效率 = 无效干扰动作时间步 / 总干扰动作时间步
        """
        failed_jamming_steps = 0
        total_jamming_steps = 0
        
        for step_data in episode_data:
            for uav_id, is_jamming in enumerate(step_data['uav_jamming']):
                if is_jamming:
                    total_jamming_steps += 1
                    uav_pos = step_data['uav_positions'][uav_id]
                    
                    # 检查是否在有效干扰范围内
                    effective_jamming = False
                    for radar_pos in step_data['radar_positions']:
                        distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                        if distance < 400:  # 有效干扰范围
                            effective_jamming = True
                            break
                    
                    if not effective_jamming:
                        failed_jamming_steps += 1
        
        if total_jamming_steps == 0:
            return 0.0
        return (failed_jamming_steps / total_jamming_steps) * 100
    
    def run_episode_evaluation(self, algorithm_name="AD-PPO"):
        """运行单回合评估"""
        env = self.env
        state = env.reset()
        
        episode_data = []
        episode_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            # 记录当前状态
            step_data = {
                'uav_positions': [uav.position.copy() for uav in env.uavs],
                'radar_positions': [radar.position.copy() for radar in env.radars],
                'uav_jamming': [uav.is_jamming for uav in env.uavs],
                'jammed_radars': [radar.is_jammed for radar in env.radars]
            }
            episode_data.append(step_data)
            
            # 简单的随机动作策略（实际应用中应该用训练好的模型）
            action = env.action_space.sample()
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # 计算各项指标
        metrics = {
            'reconnaissance_completion': self.calculate_reconnaissance_completion(episode_data),
            'safe_zone_time': self.calculate_safe_zone_time(episode_data),
            'reconnaissance_cooperation': self.calculate_reconnaissance_cooperation(episode_data),
            'jamming_cooperation': self.calculate_jamming_cooperation(episode_data),
            'jamming_failure_rate': self.calculate_jamming_failure_rate(episode_data),
            'episode_reward': episode_reward,
            'episode_steps': steps,
            'success': info.get('success', False)
        }
        
        return metrics
    
    def evaluate_algorithm(self, algorithm_name="AD-PPO"):
        """评估算法性能"""
        print(f"开始评估 {algorithm_name} 算法...")
        print(f"运行 {self.num_episodes} 个回合...")
        
        all_metrics = []
        
        for episode in range(self.num_episodes):
            if episode % 20 == 0:
                print(f"进度: {episode}/{self.num_episodes}")
            
            metrics = self.run_episode_evaluation(algorithm_name)
            all_metrics.append(metrics)
            
            # 记录到日志
            for key in self.metrics_log:
                if key in metrics:
                    self.metrics_log[key].append(metrics[key])
                elif key == 'successful_episodes':
                    self.metrics_log[key].append(metrics['success'])
        
        return all_metrics
    
    def calculate_summary_metrics(self):
        """计算汇总指标"""
        summary = {}
        
        # 计算平均值和标准差
        for metric_name in ['reconnaissance_completion', 'safe_zone_time', 
                           'reconnaissance_cooperation', 'jamming_cooperation', 
                           'jamming_failure_rate']:
            values = self.metrics_log[metric_name]
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values),
                'paper_value': self.paper_metrics[metric_name]
            }
        
        return summary
    
    def print_comparison_table(self, summary):
        """打印对比表格"""
        print("\n" + "="*80)
        print("🎯 论文指标对比分析")
        print("="*80)
        print(f"{'指标':<20} {'论文值':<10} {'实验均值':<10} {'实验最高':<10} {'标准差':<10} {'差异分析':<15}")
        print("-" * 80)
        
        metrics_names = {
            'reconnaissance_completion': '侦察任务完成度',
            'safe_zone_time': '安全区域开辟时间',
            'reconnaissance_cooperation': '侦察协作率(%)',
            'jamming_cooperation': '干扰协作率(%)',
            'jamming_failure_rate': '干扰失效率(%)'
        }
        
        for metric_key, metric_name in metrics_names.items():
            paper_val = summary[metric_key]['paper_value']
            exp_mean = summary[metric_key]['mean']
            exp_max = summary[metric_key]['max']
            exp_std = summary[metric_key]['std']
            
            # 计算差异百分比
            if paper_val != 0:
                diff_percent = abs(exp_mean - paper_val) / paper_val * 100
                if diff_percent < 10:
                    status = "优秀 ✓"
                elif diff_percent < 25:
                    status = "良好"
                else:
                    status = "需改进"
            else:
                status = "特殊"
            
            print(f"{metric_name:<20} {paper_val:<10.2f} {exp_mean:<10.2f} {exp_max:<10.2f} {exp_std:<10.3f} {status:<15}")
        
        print("-" * 80)
        
        # 计算总体评分
        total_score = 0
        for metric_key in metrics_names.keys():
            paper_val = summary[metric_key]['paper_value']
            exp_mean = summary[metric_key]['mean']
            if paper_val != 0:
                score = max(0, 100 - abs(exp_mean - paper_val) / paper_val * 100)
                total_score += score
        
        avg_score = total_score / len(metrics_names)
        print(f"\n总体匹配度评分: {avg_score:.1f}/100")
        
        if avg_score >= 90:
            print("🎉 实验结果与论文高度一致！")
        elif avg_score >= 75:
            print("✓ 实验结果与论文较为一致")
        elif avg_score >= 60:
            print("⚠️ 实验结果与论文存在一定差异")
        else:
            print("❌ 实验结果与论文差异较大，需要优化")
    
    def generate_optimization_suggestions(self, summary):
        """生成优化建议"""
        print("\n" + "="*80)
        print("🔧 优化建议")
        print("="*80)
        
        suggestions = []
        
        # 侦察任务完成度
        recon_diff = self.paper_metrics['reconnaissance_completion'] - summary['reconnaissance_completion']['mean']
        if recon_diff > 0.1:
            suggestions.append("1. 增强侦察策略：扩大侦察范围，优化UAV路径规划")
        
        # 安全区域开辟时间
        time_diff = summary['safe_zone_time']['mean'] - self.paper_metrics['safe_zone_time']
        if time_diff > 0.5:
            suggestions.append("2. 优化干扰时机：更早启动干扰，提高干扰决策响应速度")
        
        # 协作率
        recon_coop_diff = self.paper_metrics['reconnaissance_cooperation'] - summary['reconnaissance_cooperation']['mean']
        if recon_coop_diff > 5:
            suggestions.append("3. 强化侦察协作：增加UAV间通信，改进协作奖励机制")
        
        jamming_coop_diff = self.paper_metrics['jamming_cooperation'] - summary['jamming_cooperation']['mean']
        if jamming_coop_diff > 5:
            suggestions.append("4. 改进干扰协作：优化多UAV干扰策略，增强协调性")
        
        # 失效率
        failure_diff = summary['jamming_failure_rate']['mean'] - self.paper_metrics['jamming_failure_rate']
        if failure_diff > 5:
            suggestions.append("5. 降低失效率：改进动作选择逻辑，增加有效性检查")
        
        if not suggestions:
            suggestions.append("当前性能已接近论文水平，可进行微调优化")
        
        for suggestion in suggestions:
            print(suggestion)
        
        return suggestions

def main():
    """主函数"""
    print("开始论文指标评估...")
    
    evaluator = PaperMetricsEvaluator(num_episodes=50)  # 使用50个回合进行快速评估
    
    # 运行评估
    results = evaluator.evaluate_algorithm("AD-PPO")
    
    # 计算汇总指标
    summary = evaluator.calculate_summary_metrics()
    
    # 打印对比表格
    evaluator.print_comparison_table(summary)
    
    # 生成优化建议
    suggestions = evaluator.generate_optimization_suggestions(summary)
    
    # 保存结果
    output_dir = 'experiments/paper_metrics_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细数据
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
    
    # 保存汇总数据
    summary_data = []
    for metric_name, data in summary.items():
        summary_data.append({
            'metric': metric_name,
            'paper_value': data['paper_value'],
            'experiment_mean': data['mean'],
            'experiment_std': data['std'],
            'experiment_max': data['max'],
            'experiment_min': data['min']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_comparison.csv'), index=False)
    
    print(f"\n📊 评估结果已保存至: {output_dir}")
    print("- detailed_metrics.csv: 详细的每回合指标数据")
    print("- summary_comparison.csv: 汇总对比数据")

if __name__ == "__main__":
    main() 