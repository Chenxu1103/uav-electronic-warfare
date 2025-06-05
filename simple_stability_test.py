#!/usr/bin/env python3
"""
简单稳定性测试 - 50回合快速验证

专门验证关键改进：
1. 干扰失效率是否显著降低
2. 基本训练稳定性是否改善
"""

import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from stability_enhanced_system import StabilityEnhancedSystem

def simple_stability_test():
    """简单稳定性测试"""
    print("🔬 简单稳定性测试")
    print("目标: 50回合快速验证核心改进")
    print("="*50)
    
    # 创建稳定性增强系统
    system = StabilityEnhancedSystem()
    
    # 极简配置 - 只测试一个阶段
    system.training_stages = [
        {
            'name': '核心功能验证',
            'episodes': 50,
            'env_config': {'num_uavs': 3, 'num_radars': 2, 'env_size': 1000.0, 'max_steps': 80},
            'focus': 'stability',
            'learning_rate': 3e-4
        }
    ]
    
    print("开始核心功能验证 (50回合)...")
    
    try:
        # 运行训练
        agent, metrics = system.run_stability_enhanced_training(total_episodes=50)
        
        # 分析结果
        print("\n🎯 核心功能验证结果:")
        print("="*50)
        
        avg = metrics['average']
        
        # 关键指标检查
        print("🔍 关键指标验证:")
        print(f"  干扰失效率: {avg['jamming_failure_rate']:.1f}%")
        if avg['jamming_failure_rate'] < 70:
            print("  ✅ 干扰失效率显著改善 (从91.5%下降)")
        else:
            print("  ⚠️ 干扰失效率仍需优化")
        
        print(f"  干扰协作率: {avg['jamming_cooperation_rate']:.1f}%")
        if avg['jamming_cooperation_rate'] > 1:
            print("  ✅ 干扰协作率有所改善 (从0.5%提升)")
        else:
            print("  ⚠️ 干扰协作率需要继续改善")
        
        print(f"  侦察完成度: {avg['reconnaissance_completion']:.3f}")
        if avg['reconnaissance_completion'] > 0.7:
            print("  ✅ 侦察完成度保持良好水平")
        else:
            print("  ⚠️ 侦察完成度需要提升")
        
        # 训练稳定性评估
        std = metrics['std']
        cv_jamming = std['jamming_cooperation_rate'] / max(avg['jamming_cooperation_rate'], 1.0)
        cv_recon = std['reconnaissance_completion'] / max(avg['reconnaissance_completion'], 0.1)
        
        print(f"\n🧠 训练稳定性:")
        print(f"  协作率变异系数: {cv_jamming:.3f}")
        print(f"  完成度变异系数: {cv_recon:.3f}")
        
        if cv_jamming < 1.0 and cv_recon < 0.3:
            print("  ✅ 训练稳定性良好")
            stability_status = "good"
        else:
            print("  ⚠️ 训练稳定性需要改善")
            stability_status = "needs_improvement"
        
        # 总体评估
        print(f"\n🏆 总体评估:")
        improvement_count = 0
        
        if avg['jamming_failure_rate'] < 70:
            improvement_count += 1
        if avg['jamming_cooperation_rate'] > 1:
            improvement_count += 1
        if avg['reconnaissance_completion'] > 0.7:
            improvement_count += 1
        if stability_status == "good":
            improvement_count += 1
        
        print(f"  改善项目: {improvement_count}/4")
        
        if improvement_count >= 3:
            print("  🎉 核心功能验证成功！建议运行完整测试")
            recommendation = "full_test"
        elif improvement_count >= 2:
            print("  🔥 部分改善明显，可以继续优化")
            recommendation = "continue_optimization"
        else:
            print("  ⚠️ 需要检查基础配置")
            recommendation = "check_config"
        
        return {
            'improvement_count': improvement_count,
            'recommendation': recommendation,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("💡 建议检查环境配置和依赖")
        return {
            'improvement_count': 0,
            'recommendation': 'fix_error',
            'error': str(e)
        }

if __name__ == "__main__":
    print("🔬 开始简单稳定性测试...")
    
    result = simple_stability_test()
    
    print(f"\n🚀 后续建议:")
    if result['recommendation'] == 'full_test':
        print("python stability_quick_test.py  # 200回合完整验证")
    elif result['recommendation'] == 'continue_optimization':
        print("python stability_enhanced_system.py  # 1700回合深度训练") 
    elif result['recommendation'] == 'check_config':
        print("检查配置，调整参数后重试")
    else:
        print("修复错误后重试")
    
    print("\n✅ 简单稳定性测试完成!") 