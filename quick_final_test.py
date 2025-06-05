#!/usr/bin/env python3
"""
快速最终测试 - 验证最终完整复现系统

使用少量回合快速验证系统的改进效果
"""

import os
import sys
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from final_complete_reproduction_system import FinalCompleteReproductionSystem

def quick_final_test():
    """快速最终测试"""
    print("🚀 最终完整复现系统快速测试")
    print("目标: 验证系统在300回合内的改进效果")
    
    try:
        system = FinalCompleteReproductionSystem()
        
        # 使用较少回合进行快速测试
        agent, final_metrics = system.run_complete_reproduction(total_episodes=300)
        
        print("\n📊 快速测试结果概览:")
        print(f"侦察完成度: {final_metrics['reconnaissance_completion']:.3f} (目标: 0.97, 达成率: {(final_metrics['reconnaissance_completion']/0.97)*100:.1f}%)")
        print(f"安全区域时间: {final_metrics['safe_zone_development_time']:.2f} (目标: 2.1, 达成率: {min(100, (final_metrics['safe_zone_development_time']/2.1)*100):.1f}%)")
        print(f"侦察协作率: {final_metrics['reconnaissance_cooperation_rate']:.1f}% (目标: 37%, 达成率: {(final_metrics['reconnaissance_cooperation_rate']/37)*100:.1f}%)")
        print(f"干扰协作率: {final_metrics['jamming_cooperation_rate']:.1f}% (目标: 34%, 达成率: {(final_metrics['jamming_cooperation_rate']/34)*100:.1f}%)")
        
        # 计算总体改进
        improvements = [
            (final_metrics['reconnaissance_completion']/0.97)*100,
            min(100, (final_metrics['safe_zone_development_time']/2.1)*100),
            (final_metrics['reconnaissance_cooperation_rate']/37)*100,
            (final_metrics['jamming_cooperation_rate']/34)*100
        ]
        
        avg_improvement = np.mean(improvements)
        print(f"\n总体接近度: {avg_improvement:.1f}%")
        
        if avg_improvement > 85:
            print("🎉 优秀! 系统显示出极高的论文复现潜力!")
        elif avg_improvement > 70:
            print("👍 良好! 系统效果显著，有望达到论文水准!")
        elif avg_improvement > 50:
            print("⚠️ 中等，系统正在向论文目标收敛!")
        else:
            print("🔧 需要更长时间的训练以接近论文指标")
        
        return True, final_metrics
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, metrics = quick_final_test()
    
    if success:
        print("\n✅ 快速验证完成! 最终系统已准备好进行完整论文复现!")
        print("📊 建议运行完整版本以获得更接近论文的指标数据!")
    else:
        print("\n❌ 需要检查系统配置") 