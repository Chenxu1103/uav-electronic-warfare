#!/usr/bin/env python3
"""
快速修复验证 - 测试维度问题是否解决

使用50回合快速验证系统修复效果
"""

import os
import sys
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from final_complete_reproduction_system import FinalCompleteReproductionSystem

def quick_fix_verification():
    """快速修复验证"""
    print("🔧 快速修复验证测试")
    print("目标: 验证维度问题已解决，系统可正常运行")
    print("="*60)
    
    try:
        system = FinalCompleteReproductionSystem()
        
        # 使用极少回合进行快速验证
        print("开始验证训练 (50回合)...")
        agent, final_metrics = system.run_complete_reproduction(total_episodes=50)
        
        print("\n✅ 维度问题已解决!")
        print("📊 快速验证结果:")
        
        # 显示关键指标
        key_metrics = [
            'reconnaissance_completion',
            'safe_zone_development_time', 
            'reconnaissance_cooperation_rate',
            'jamming_cooperation_rate'
        ]
        
        for metric in key_metrics:
            if metric in final_metrics:
                value = final_metrics[metric]
                print(f"  {metric}: {value:.3f}")
        
        print("\n🎯 系统状态: 正常运行")
        print("💡 建议: 现在可以安全运行完整训练")
        
        return True, final_metrics
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, metrics = quick_fix_verification()
    
    if success:
        print("\n✅ 修复验证成功!")
        print("🚀 现在可以运行完整系统了!")
        print("\n推荐运行命令:")
        print("python enhanced_paper_reproduction_test.py  # 800回合增强测试")
        print("python final_complete_reproduction_system.py  # 1600回合完整训练") 
    else:
        print("\n❌ 仍有问题需要解决") 