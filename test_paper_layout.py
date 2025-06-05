"""
快速测试脚本 - 验证环境布局是否符合论文
"""

import numpy as np
from src.environment.electronic_warfare_env import ElectronicWarfareEnv

def test_layout():
    """测试环境布局是否符合论文"""
    print("测试环境布局是否符合论文要求...")
    
    # 创建环境
    env = ElectronicWarfareEnv(num_uavs=4, num_radars=3, env_size=2000.0)
    
    # 重置环境获取初始状态
    env.reset()
    
    print(f"环境大小: {env.env_size}m x {env.env_size}m")
    print(f"无人机数量: {len(env.uavs)}")
    print(f"雷达数量: {len(env.radars)}")
    print()
    
    print("=== 无人机初始位置分析 ===")
    for i, uav in enumerate(env.uavs):
        x, y, z = uav.position
        print(f"UAV {i}: 位置 ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # 检查是否在左侧
        if x < -200:  # 左侧区域
            side = "左侧 ✓"
        else:
            side = "非左侧 ✗"
        print(f"       位于环境{side}")
        
        # 检查速度方向
        vx, vy, vz = uav.velocity
        if vx > 0:  # 向右飞行
            direction = "向右 ✓"
        else:
            direction = "非向右 ✗"
        print(f"       速度方向{direction} ({vx:.1f}, {vy:.1f}, {vz:.1f})")
    
    print()
    print("=== 雷达位置分析 ===")
    for i, radar in enumerate(env.radars):
        x, y, z = radar.position
        print(f"雷达 {i}: 位置 ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # 检查是否在右侧
        if x > 200:  # 右侧区域
            side = "右侧 ✓"
        else:
            side = "非右侧 ✗"
        print(f"        位于环境{side}")
    
    # 计算无人机与雷达的相对位置
    print()
    print("=== 布局验证 ===")
    uav_x_avg = np.mean([uav.position[0] for uav in env.uavs])
    radar_x_avg = np.mean([radar.position[0] for radar in env.radars])
    
    print(f"无人机平均X位置: {uav_x_avg:.1f}")
    print(f"雷达平均X位置: {radar_x_avg:.1f}")
    
    if uav_x_avg < radar_x_avg:
        print("布局验证: 无人机在雷达西侧 ✓ (符合论文)")
        layout_correct = True
    else:
        print("布局验证: 无人机不在雷达西侧 ✗ (不符合论文)")
        layout_correct = False
    
    distance = radar_x_avg - uav_x_avg
    print(f"平均距离: {distance:.1f}m")
    
    print()
    if layout_correct:
        print("🎉 环境布局符合论文要求！")
    else:
        print("❌ 环境布局需要进一步调整")
    
    return layout_correct

if __name__ == "__main__":
    test_layout() 