#!/usr/bin/env python3
"""
稳定性增强快速测试

专门测试新的稳定性增强系统能否解决：
1. 干扰失效率91.5% -> 目标25%以下
2. 训练不稳定问题
3. 协作行为持续性问题

快速验证200回合的改进效果
"""

import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from stability_enhanced_system import StabilityEnhancedSystem

def quick_stability_test():
    """快速稳定性测试"""
    print("🚀 稳定性增强快速测试")
    print("目标: 验证优化系统能否解决关键问题")
    print("="*70)
    
    # 与之前系统对比
    print("📈 与之前系统对比:")
    print("之前系统 (ultra_advanced_quick_test.py 结果):")
    print("  干扰失效率: 91.5% (严重问题)")
    print("  干扰协作率: 0.5% 平均, 50.0% 最高 (不稳定)")
    print("  安全区域时间: 0.27s 平均, 2.45s 最高 (不稳定)")
    print("  侦察完成度: 0.765 平均, 0.890 最高")
    print()
    
    print("稳定性增强系统预期改进:")
    print("  🎯 干扰失效率: 目标降低到25%以下")
    print("  🎯 协作稳定性: 提升平均值，缩小方差")
    print("  🎯 训练稳定性: 更稳定的学习过程")
    print("  🎯 整体复现率: 从23.4%提升到60%+")
    
    # 创建稳定性增强系统
    system = StabilityEnhancedSystem()
    
    # 修改为快速测试配置
    system.training_stages = [
        {
            'name': '稳定性验证',
            'episodes': 150,
            'env_config': {'num_uavs': 3, 'num_radars': 2, 'env_size': 1200.0, 'max_steps': 120},
            'focus': 'stability',
            'learning_rate': 3e-4
        },
        {
            'name': '干扰优化验证',
            'episodes': 50,
            'env_config': {'num_uavs': 3, 'num_radars': 2, 'env_size': 1500.0, 'max_steps': 150},
            'focus': 'jamming',
            'learning_rate': 2e-4
        }
    ]
    
    print("\n开始稳定性增强训练 (200回合)...")
    
    # 运行训练
    agent, metrics = system.run_stability_enhanced_training(total_episodes=200)
    
    # 详细结果分析
    print("\n🎯 稳定性增强快速测试结果:")
    print("="*70)
    
    avg = metrics['average']
    std = metrics['std']
    max_vals = metrics['max']
    
    print("\n📊 关键问题解决效果分析:")
    print("-"*60)
    
    # 1. 干扰失效率改善
    jamming_failure_old = 91.5
    jamming_failure_new = avg['jamming_failure_rate']
    improvement_failure = jamming_failure_old - jamming_failure_new
    
    print("干扰失效率:")
    print(f"  之前值: {jamming_failure_old}%")
    print(f"  当前值: {jamming_failure_new:.1f}% ± {std['jamming_failure_rate']:.1f}")
    print(f"  改善幅度: {improvement_failure:.1f}%")
    if jamming_failure_new < 30:
        print("  状态: 🎉 显著改善")
    elif jamming_failure_new < 50:
        print("  状态: 🔥 明显改善")
    else:
        print("  状态: ⚠️ 需继续优化")
    
    # 2. 协作稳定性改善
    cooperation_old_avg = 0.5
    cooperation_old_max = 50.0
    cooperation_new_avg = avg['jamming_cooperation_rate']
    cooperation_new_max = max_vals['jamming_cooperation_rate']
    cooperation_stability = cooperation_new_avg / max(cooperation_new_max, 1.0)
    
    print("\n干扰协作率:")
    print(f"  之前: {cooperation_old_avg}% 平均, {cooperation_old_max}% 最高 (稳定性: {cooperation_old_avg/cooperation_old_max:.3f})")
    print(f"  当前: {cooperation_new_avg:.1f}% 平均, {cooperation_new_max:.1f}% 最高 (稳定性: {cooperation_stability:.3f})")
    if cooperation_stability > 0.5:
        print("  状态: 🎉 稳定性显著提升")
    elif cooperation_stability > 0.3:
        print("  状态: 🔥 稳定性改善")
    else:
        print("  状态: ⚠️ 仍需改善稳定性")
    
    # 3. 安全区域时间稳定性
    safe_zone_old_avg = 0.27
    safe_zone_old_max = 2.45
    safe_zone_new_avg = avg['safe_zone_development_time']
    safe_zone_new_max = max_vals['safe_zone_development_time']
    safe_zone_stability = safe_zone_new_avg / max(safe_zone_new_max, 0.1)
    
    print("\n安全区域时间:")
    print(f"  之前: {safe_zone_old_avg}s 平均, {safe_zone_old_max}s 最高")
    print(f"  当前: {safe_zone_new_avg:.2f}s 平均, {safe_zone_new_max:.2f}s 最高")
    if safe_zone_new_avg > 1.0:
        print("  状态: 🎉 显著提升")
    elif safe_zone_new_avg > 0.5:
        print("  状态: 🔥 有所改善")
    else:
        print("  状态: ⚠️ 需继续改善")
    
    # 4. 侦察完成度
    recon_old_avg = 0.765
    recon_new_avg = avg['reconnaissance_completion']
    recon_improvement = recon_new_avg - recon_old_avg
    
    print("\n侦察完成度:")
    print(f"  之前: {recon_old_avg:.3f}")
    print(f"  当前: {recon_new_avg:.3f} ± {std['reconnaissance_completion']:.3f}")
    print(f"  改善: {recon_improvement:+.3f}")
    if recon_new_avg > 0.85:
        print("  状态: 🎉 优秀水平")
    elif recon_new_avg > 0.80:
        print("  状态: 🔥 良好水平")
    else:
        print("  状态: ⚠️ 需继续提升")
    
    # 总体复现成功率对比
    paper_targets = system.paper_targets
    
    # 计算总体达成率
    total_achievement = np.mean([
        min(100, avg['reconnaissance_completion'] / paper_targets['reconnaissance_completion'] * 100),
        min(100, avg['safe_zone_development_time'] / paper_targets['safe_zone_development_time'] * 100),
        min(100, avg['reconnaissance_cooperation_rate'] / paper_targets['reconnaissance_cooperation_rate'] * 100),
        min(100, avg['jamming_cooperation_rate'] / paper_targets['jamming_cooperation_rate'] * 100),
        max(0, (paper_targets['jamming_failure_rate'] - avg['jamming_failure_rate']) / paper_targets['jamming_failure_rate'] * 100)
    ])
    
    old_achievement = 23.4  # 之前系统的成功率
    achievement_improvement = total_achievement - old_achievement
    
    print("\n" + "="*70)
    print("🏆 总体复现成功率对比:")
    print(f"  之前系统: {old_achievement}%")
    print(f"  稳定系统: {total_achievement:.1f}%")
    print(f"  提升幅度: {achievement_improvement:+.1f}%")
    
    if total_achievement > 60:
        print("  状态: 🎉 显著改善，已达到良好水平")
    elif total_achievement > 40:
        print("  状态: 🔥 明显改善，继续优化可达到优秀水平")
    else:
        print("  状态: ⚠️ 有所改善，需要更多优化")
    
    # 训练稳定性分析
    print("\n🧠 训练稳定性分析:")
    
    # 计算变异系数（标准差/均值）来衡量稳定性
    cv_jamming_coop = std['jamming_cooperation_rate'] / max(avg['jamming_cooperation_rate'], 1.0)
    cv_safe_zone = std['safe_zone_development_time'] / max(avg['safe_zone_development_time'], 0.1)
    cv_recon = std['reconnaissance_completion'] / max(avg['reconnaissance_completion'], 0.1)
    
    print(f"  干扰协作率变异系数: {cv_jamming_coop:.3f} {'(稳定)' if cv_jamming_coop < 0.5 else '(不稳定)'}")
    print(f"  安全区域时间变异系数: {cv_safe_zone:.3f} {'(稳定)' if cv_safe_zone < 1.0 else '(不稳定)'}")
    print(f"  侦察完成度变异系数: {cv_recon:.3f} {'(稳定)' if cv_recon < 0.2 else '(不稳定)'}")
    
    overall_stability = (cv_jamming_coop < 0.5) + (cv_safe_zone < 1.0) + (cv_recon < 0.2)
    if overall_stability >= 2:
        print("  整体稳定性: 🎉 良好")
    elif overall_stability >= 1:
        print("  整体稳定性: 🔥 中等")
    else:
        print("  整体稳定性: ⚠️ 需改善")
    
    # 关键突破点总结
    print("\n🚀 关键突破点总结:")
    breakthrough_count = 0
    
    if jamming_failure_new < 40:
        print("  🎉 干扰失效率重大突破")
        breakthrough_count += 1
    
    if cooperation_new_avg > 5:
        print("  🎉 干扰协作率重大突破")
        breakthrough_count += 1
    
    if safe_zone_new_avg > 0.5:
        print("  🎉 安全区域时间重大突破")
        breakthrough_count += 1
    
    if recon_new_avg > 0.85:
        print("  🎉 侦察完成度重大突破")
        breakthrough_count += 1
    
    if cv_jamming_coop < 0.5 and cv_recon < 0.2:
        print("  🎉 训练稳定性重大突破")
        breakthrough_count += 1
    
    print(f"\n🏆 总突破点数: {breakthrough_count}/5")
    
    # 后续建议
    print("\n💡 后续训练建议:")
    if total_achievement > 60:
        print("  ✅ 建议运行完整版ultra_advanced_reproduction_system.py (1700回合)")
        print("  ✅ 当前系统已具备良好基础，可以进行深度优化")
    elif total_achievement > 40:
        print("  🔧 建议先运行更多稳定性训练")
        print("  🔧 可以适当增加协作权重")
        print("  ✅ 然后运行完整版训练")
    else:
        print("  🔧 需要进一步调整网络架构或奖励机制")
        print("  🔧 建议分析训练日志，找出不稳定原因")
    
    print("\n✅ 稳定性增强快速测试完成!")
    
    return {
        'total_achievement': total_achievement,
        'improvement': achievement_improvement,
        'breakthrough_count': breakthrough_count,
        'metrics': metrics
    }

if __name__ == "__main__":
    results = quick_stability_test()
    
    print(f"\n🚀 建议运行命令:")
    if results['total_achievement'] > 60:
        print("python ultra_advanced_reproduction_system.py  # 1700回合完整训练")
    else:
        print("python stability_enhanced_system.py  # 1700回合稳定性训练")
        print("# 或继续调试优化参数") 