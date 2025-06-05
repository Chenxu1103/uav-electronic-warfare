#!/usr/bin/env python3
"""
快速论文复现测试 - 验证精确复现系统

使用较少回合验证新系统是否能够快速接近论文指标
"""

import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from paper_exact_reproduction_system import PaperExactReproductionSystem

def quick_test_paper_reproduction():
    """快速测试论文复现"""
    print("🚀 快速论文复现测试")
    print("目标: 用200回合验证新系统的有效性")
    
    # 创建系统
    system = PaperExactReproductionSystem()
    
    # 快速训练
    try:
        agent, final_metrics = system.run_paper_exact_reproduction(total_episodes=200)
        
        print("\n📊 快速测试结果:")
        print(f"侦察完成度: {final_metrics['reconnaissance_completion']:.3f} (目标: 0.97)")
        print(f"安全区域时间: {final_metrics['safe_zone_development_time']:.2f} (目标: 2.1)")
        print(f"侦察协作率: {final_metrics['reconnaissance_cooperation_rate']:.1f}% (目标: 37%)")
        print(f"干扰协作率: {final_metrics['jamming_cooperation_rate']:.1f}% (目标: 34%)")
        
        # 计算总体改进度
        targets = [0.97, 2.1, 37.0, 34.0]
        results = [
            final_metrics['reconnaissance_completion'],
            final_metrics['safe_zone_development_time'],
            final_metrics['reconnaissance_cooperation_rate'],
            final_metrics['jamming_cooperation_rate']
        ]
        
        improvements = []
        for i, (result, target) in enumerate(zip(results, targets)):
            if i == 1:  # 安全区域时间
                improvement = min(100, result / target * 100)
            else:
                improvement = min(100, result / target * 100)
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        print(f"\n总体接近度: {avg_improvement:.1f}%")
        
        if avg_improvement > 80:
            print("✅ 优秀! 新系统显示出巨大潜力!")
        elif avg_improvement > 60:
            print("👍 良好! 新系统效果明显!")
        else:
            print("⚠️ 需要更多训练时间")
            
        return True, final_metrics
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False, None

if __name__ == "__main__":
    success, metrics = quick_test_paper_reproduction()
    
    if success:
        print("\n✅ 快速验证完成! 新系统已准备好进行完整复现!")
    else:
        print("\n❌ 需要检查系统配置") 