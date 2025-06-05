"""
快速改进脚本 - 实现分析报告建议
"""

import numpy as np
import pandas as pd
import os
from src.environment.electronic_warfare_env import ElectronicWarfareEnv

class QuickImprover:
    def __init__(self, num_episodes=30):
        self.num_episodes = num_episodes
        self.paper_metrics = {
            'reconnaissance_completion': 0.97,
            'safe_zone_time': 2.1,
            'reconnaissance_cooperation': 37.0,
            'jamming_cooperation': 34.0,
            'jamming_failure_rate': 23.3
        }
        
        # 改进参数
        self.improved_params = {
            'reconnaissance_range': 800,
            'jamming_range': 600,
            'cooperation_distance': 500,
            'early_jamming_step': 30
        }
        
        self.metrics_log = {
            'reconnaissance_completion': [],
            'safe_zone_time': [],
            'reconnaissance_cooperation': [],
            'jamming_cooperation': [],
            'jamming_failure_rate': []
        }
    
    def create_improved_env(self):
        env = ElectronicWarfareEnv(num_uavs=3, num_radars=2, env_size=2000.0, max_steps=210)
        
        # 改进奖励权重
        env.reward_weights.update({
            'jamming_success': 150.0,
            'coordination_reward': 80.0,
            'approach_reward': 25.0,
            'jamming_attempt_reward': 15.0,
            'stealth_reward': 2.0,
            'distance_penalty': -0.00003,
            'energy_penalty': -0.003,
            'detection_penalty': -0.05
        })
        
        return env
    
    def improved_strategy(self, env, step):
        actions = []
        
        for i, uav in enumerate(env.uavs):
            if not uav.is_alive:
                actions.extend([0, 0, 0, 0, 0, 0])
                continue
            
            # 找最近雷达
            min_distance = float('inf')
            target_radar = None
            for radar in env.radars:
                distance = np.linalg.norm(uav.position - radar.position)
                if distance < min_distance:
                    min_distance = distance
                    target_radar = radar
            
            if target_radar is not None:
                direction = target_radar.position - uav.position
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    direction = direction / direction_norm
                    
                    # 角色策略
                    if i == 0:  # 侦察
                        if min_distance > 600:
                            vx, vy, vz = direction[0] * 0.9, direction[1] * 0.9, -0.3
                            should_jam = False
                        else:
                            # 螺旋侦察
                            angle = step * 0.12
                            vx = direction[0] * 0.4 + np.cos(angle) * 0.4
                            vy = direction[1] * 0.4 + np.sin(angle) * 0.4
                            vz = -0.15
                            should_jam = step > 30
                    
                    elif i == 1:  # 干扰
                        if min_distance > 480:
                            vx, vy, vz = direction[0] * 0.8, direction[1] * 0.8, -0.25
                            should_jam = step > 25
                        else:
                            vx, vy, vz = direction[0] * 0.1, direction[1] * 0.1, 0.0
                            should_jam = True
                    
                    else:  # 协作
                        if min_distance > 500:
                            vx, vy, vz = direction[0] * 0.7, direction[1] * 0.7, -0.2
                            should_jam = False
                        else:
                            vx, vy, vz = direction[0] * 0.3, direction[1] * 0.3, 0.0
                            should_jam = step > 40
                    
                    # 添加随机性
                    vx += np.random.normal(0, 0.06)
                    vy += np.random.normal(0, 0.06)
                    
                    # 限制范围
                    vx = np.clip(vx, -1.0, 1.0)
                    vy = np.clip(vy, -1.0, 1.0)
                    vz = np.clip(vz, -1.0, 1.0)
                    
                    # 干扰参数
                    if should_jam and min_distance < 600:
                        jam_dir_x = direction[0] * 0.95
                        jam_dir_y = direction[1] * 0.95
                        jam_power = 0.98
                    else:
                        jam_dir_x = 0.0
                        jam_dir_y = 0.0
                        jam_power = 0.0
                    
                    actions.extend([vx, vy, vz, jam_dir_x, jam_dir_y, jam_power])
                else:
                    actions.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                actions.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(actions, dtype=np.float32)
    
    def calculate_metrics(self, episode_data):
        # 侦察完成度
        detected_radars = set()
        detection_scores = []
        
        for step_data in episode_data:
            for radar_id, radar_pos in enumerate(step_data['radar_positions']):
                for uav_pos in step_data['uav_positions']:
                    distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                    if distance < 800:
                        detected_radars.add(radar_id)
                        quality = max(0, 1 - distance/800)
                        detection_scores.append(quality)
        
        base_completion = len(detected_radars) / len(episode_data[0]['radar_positions'])
        if detection_scores:
            avg_quality = np.mean(detection_scores)
            if len(detection_scores) >= 20:  # 持续侦察奖励
                reconnaissance_completion = min(1.0, base_completion * avg_quality * 1.2)
            else:
                reconnaissance_completion = base_completion * avg_quality
        else:
            reconnaissance_completion = 0.0
        
        # 安全区域时间
        safe_zone_time = 3.0
        for step, step_data in enumerate(episode_data):
            if sum(step_data['jammed_radars']) >= 1:
                safe_zone_time = (step + 1) * 0.1
                break
        
        # 侦察协作率
        cooperative_recon = 0
        total_recon = 0
        
        for step_data in episode_data:
            for radar_id, radar_pos in enumerate(step_data['radar_positions']):
                uavs_count = 0
                for uav_pos in step_data['uav_positions']:
                    distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                    if distance < 800:
                        uavs_count += 1
                
                if uavs_count > 0:
                    total_recon += 1
                    if uavs_count > 1:
                        cooperative_recon += 1
        
        reconnaissance_cooperation = 0.0
        if total_recon > 0:
            reconnaissance_cooperation = (cooperative_recon / total_recon) * 100
        
        # 干扰协作率
        cooperative_jam = 0
        total_jam = 0
        
        for step_data in episode_data:
            jamming_uavs = []
            for uav_id, is_jamming in enumerate(step_data['uav_jamming']):
                if is_jamming:
                    jamming_uavs.append(step_data['uav_positions'][uav_id])
            
            if len(jamming_uavs) > 0:
                total_jam += 1
                if len(jamming_uavs) > 1:
                    for i in range(len(jamming_uavs)):
                        for j in range(i+1, len(jamming_uavs)):
                            distance = np.linalg.norm(np.array(jamming_uavs[i]) - np.array(jamming_uavs[j]))
                            if 100 < distance < 600:
                                cooperative_jam += 1
                                break
                        else:
                            continue
                        break
        
        jamming_cooperation = 0.0
        if total_jam > 0:
            jamming_cooperation = (cooperative_jam / total_jam) * 100
        
        # 干扰失效率
        failed_jamming = 0
        total_jamming = 0
        
        for step_data in episode_data:
            for uav_id, is_jamming in enumerate(step_data['uav_jamming']):
                if is_jamming:
                    total_jamming += 1
                    uav_pos = step_data['uav_positions'][uav_id]
                    
                    effective = False
                    for radar_pos in step_data['radar_positions']:
                        distance = np.linalg.norm(np.array(uav_pos) - np.array(radar_pos))
                        if distance < 600:
                            effective = True
                            break
                    
                    if not effective:
                        failed_jamming += 1
        
        jamming_failure_rate = 0.0
        if total_jamming > 0:
            jamming_failure_rate = (failed_jamming / total_jamming) * 100
        
        return {
            'reconnaissance_completion': reconnaissance_completion,
            'safe_zone_time': safe_zone_time,
            'reconnaissance_cooperation': reconnaissance_cooperation,
            'jamming_cooperation': jamming_cooperation,
            'jamming_failure_rate': jamming_failure_rate
        }
    
    def run_episode(self):
        env = self.create_improved_env()
        state = env.reset()
        
        episode_data = []
        
        for step in range(env.max_steps):
            step_data = {
                'uav_positions': [uav.position.copy() for uav in env.uavs],
                'radar_positions': [radar.position.copy() for radar in env.radars],
                'uav_jamming': [uav.is_jamming for uav in env.uavs],
                'jammed_radars': [radar.is_jammed for radar in env.radars]
            }
            episode_data.append(step_data)
            
            action = self.improved_strategy(env, step)
            state, reward, done, info = env.step(action)
            
            if done:
                break
        
        return self.calculate_metrics(episode_data)
    
    def evaluate(self):
        print("🚀 快速改进评估开始...")
        
        for episode in range(self.num_episodes):
            if episode % 10 == 0:
                print(f"进度: {episode}/{self.num_episodes}")
            
            metrics = self.run_episode()
            
            for key in self.metrics_log:
                self.metrics_log[key].append(metrics[key])
        
        # 计算汇总
        summary = {}
        for metric_name in self.metrics_log:
            values = self.metrics_log[metric_name]
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'paper_value': self.paper_metrics[metric_name]
            }
        
        # 打印结果
        print("\n" + "="*85)
        print("🎯 快速改进结果")
        print("="*85)
        print(f"{'指标':<20} {'论文值':<10} {'改进均值':<10} {'改进最高':<10} {'匹配度':<15}")
        print("-" * 85)
        
        metrics_names = {
            'reconnaissance_completion': '侦察任务完成度',
            'safe_zone_time': '安全区域开辟时间',
            'reconnaissance_cooperation': '侦察协作率(%)',
            'jamming_cooperation': '干扰协作率(%)',
            'jamming_failure_rate': '干扰失效率(%)'
        }
        
        total_score = 0
        
        for metric_key, metric_name in metrics_names.items():
            paper_val = summary[metric_key]['paper_value']
            exp_mean = summary[metric_key]['mean']
            exp_max = summary[metric_key]['max']
            
            if paper_val != 0:
                match_percent = max(0, 100 - abs(exp_mean - paper_val) / paper_val * 100)
                total_score += match_percent
                
                if match_percent >= 75:
                    status = "优秀 ✓"
                elif match_percent >= 60:
                    status = "良好"
                elif match_percent >= 40:
                    status = "一般"
                else:
                    status = "待改进"
            else:
                status = "特殊"
                match_percent = 50
            
            print(f"{metric_name:<20} {paper_val:<10.2f} {exp_mean:<10.2f} {exp_max:<10.2f} {status:<15}")
        
        avg_score = total_score / len(metrics_names)
        print(f"\n🎯 总体匹配度: {avg_score:.1f}/100")
        
        if avg_score >= 65:
            print("🎉 快速改进取得良好效果！")
        elif avg_score >= 45:
            print("📈 改进有效果，可继续优化")
        else:
            print("⚠️ 需要更深层优化")
        
        # 保存结果
        output_dir = 'experiments/quick_improvements'
        os.makedirs(output_dir, exist_ok=True)
        
        summary_data = []
        for metric_name, data in summary.items():
            summary_data.append({
                'metric': metric_name,
                'paper_value': data['paper_value'],
                'improved_mean': data['mean'],
                'improved_std': data['std'],
                'improved_max': data['max']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'quick_improvements.csv'), index=False)
        
        print(f"\n📊 结果已保存至: {output_dir}")
        
        return summary

def main():
    improver = QuickImprover(num_episodes=30)
    summary = improver.evaluate()

if __name__ == "__main__":
    main() 