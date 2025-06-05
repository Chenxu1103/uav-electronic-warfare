"""
按照论文布局的可视化演示脚本
模拟论文中的任务流程布局
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from src.utils.visualization import VisualizationTool

def generate_paper_layout_data():
    """生成按照论文图片布局的数据"""
    
    # 1. 无人机从左侧起始，模拟从载机出发
    trajectories = []
    
    # 无人机初始位置（左侧）
    uav_start_positions = [
        [-700, -100, 400],   # UAV1
        [-720, 0, 450],      # UAV2  
        [-710, 100, 420],    # UAV3
        [-690, -50, 380]     # UAV4
    ]
    
    # 雷达网位置（右侧纵向排列）
    radar_positions = [
        [450, -200, 0],      # 雷达1
        [470, -100, 0],      # 雷达2
        [440, 0, 0],         # 雷达3
        [460, 100, 0],       # 雷达4  
        [450, 200, 0]        # 雷达5
    ]
    
    # 生成轨迹：从左侧向右侧雷达网接近
    for i in range(4):
        trajectory = {
            'uav_id': i,
            'positions': []
        }
        
        start_x, start_y, start_z = uav_start_positions[i]
        
        # 25步轨迹，逐步接近雷达网
        for step in range(25):
            progress = step / 24.0
            
            # 主要向右移动，朝向雷达网
            x = start_x + progress * 800 + np.random.normal(0, 30)
            y = start_y + progress * (i-1.5) * 50 + np.random.normal(0, 25)
            z = start_z - progress * 200 + np.random.normal(0, 20)
            z = max(z, 100)  # 确保不会太低
            
            trajectory['positions'].append([x, y, z])
        
        trajectories.append(trajectory)
    
    # 雷达数据
    radars = []
    for i, pos in enumerate(radar_positions):
        radars.append({
            'id': i,
            'position': pos,
            'jammed': i % 2 == 1  # 交替干扰状态
        })
    
    # 当前UAV状态
    uavs = []
    for i, traj in enumerate(trajectories):
        last_pos = traj['positions'][-1]
        uavs.append({
            'id': i,
            'position': last_pos,
            'jamming': i % 2 == 0
        })
    
    # 算法比较数据
    episodes = list(range(12))
    ad_ppo_rewards = [350 + i*30 + np.random.normal(0, 15) for i in episodes]
    maddpg_rewards = [300 + i*25 + np.random.normal(0, 20) for i in episodes]
    
    ad_ppo_success_rates = [min(100, max(0, 25 + i*5 + np.random.normal(0, 4))) for i in episodes]
    maddpg_success_rates = [min(100, max(0, 20 + i*4 + np.random.normal(0, 5))) for i in episodes]
    
    comparison_data = pd.DataFrame({
        'Episode': episodes + episodes,
        'Algorithm': ['AD-PPO'] * len(episodes) + ['MADDPG'] * len(episodes),
        'Average Reward': ad_ppo_rewards + maddpg_rewards,
        'Success Rate (%)': ad_ppo_success_rates + maddpg_success_rates
    })
    
    return {
        'trajectories': trajectories,
        'radars': radars,
        'uavs': uavs,
        'comparison_data': comparison_data
    }

def main():
    """主函数"""
    try:
        print("开始生成按照论文布局的可视化...")
        
        output_dir = 'experiments/paper_layout'
        os.makedirs(output_dir, exist_ok=True)
        
        viz_tool = VisualizationTool(save_dir=output_dir)
        
        print("正在生成论文布局数据...")
        data = generate_paper_layout_data()
        
        # 1. 态势图
        print("正在生成态势图...")
        viz_tool.plot_situation_map_2d(
            uavs=data['uavs'],
            radars=data['radars'],
            title='Electronic Warfare Situation Map (Paper Layout)',
            show_jamming=True,
            save_path=os.path.join(output_dir, 'paper_situation_map.png')
        )
        
        # 2. 轨迹图
        print("正在生成轨迹图...")
        viz_tool.plot_trajectories(
            trajectories=data['trajectories'],
            radars=data['radars'],
            title='UAV Trajectories - Approaching Radar Network (Paper Layout)',
            save_path=os.path.join(output_dir, 'paper_trajectories.png')
        )
        
        # 3. 算法比较
        print("正在生成算法比较图...")
        viz_tool.plot_algorithm_comparison(
            data=data['comparison_data'],
            metrics=['Average Reward', 'Success Rate (%)'],
            title='AD-PPO vs MADDPG Performance Comparison',
            save_path=os.path.join(output_dir, 'paper_algorithm_comparison.png')
        )
        
        print(f"论文布局可视化已完成，保存至: {output_dir}")
        
    except Exception as e:
        print(f"生成过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 