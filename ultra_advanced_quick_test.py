#!/usr/bin/env python3
"""
超级高级系统快速测试

使用200回合快速验证：
1. 1024维超深度网络效果
2. 协作训练模块是否生效
3. 干扰协作率是否能突破0%
4. 安全区域时间是否有改善
"""

import os
import sys
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultra_advanced_reproduction_system import UltraAdvancedReproductionSystem

def ultra_advanced_quick_test():
    """超级高级快速测试"""
    print("🚀 超级高级系统快速测试")
    print("目标: 验证1024维网络+协作模块能否突破性能瓶颈")
    print("="*70)
    
    try:
        system = UltraAdvancedReproductionSystem()
        
        # 使用200回合进行快速验证
        print("开始超级高级训练 (200回合)...")
        agent, final_metrics = system.run_ultra_advanced_reproduction(total_episodes=200)
        
        print("\n🎯 超级高级快速测试结果:")
        print("="*70)
        
        # 重点关注关键指标的突破
        key_metrics = {
            'reconnaissance_completion': ('侦察完成度', 0.97),
            'safe_zone_development_time': ('安全区域时间', 2.1),
            'reconnaissance_cooperation_rate': ('侦察协作率', 37.0),
            'jamming_cooperation_rate': ('干扰协作率', 34.0)
        }
        
        print("\n📊 关键性能突破分析:")
        print("-" * 60)
        
        breakthroughs = []
        
        for metric, (name, target) in key_metrics.items():
            if metric in final_metrics:
                value = final_metrics[metric]
                max_value = final_metrics.get(f'{metric}_max', value)
                achievement = min(100, value / target * 100) if target > 0 else 0
                max_achievement = min(100, max_value / target * 100) if target > 0 else 0
                
                print(f"{name}:")
                print(f"  平均值: {value:.3f} (达成率: {achievement:.1f}%)")
                print(f"  最高值: {max_value:.3f} (最高达成率: {max_achievement:.1f}%)")
                
                # 检查突破
                if metric == 'jamming_cooperation_rate':
                    if max_value > 5:
                        breakthroughs.append(f"🎉 干扰协作率重大突破: {max_value:.1f}%")
                    elif value > 1:
                        breakthroughs.append(f"✅ 干扰协作率有改善: {value:.1f}%")
                
                if metric == 'safe_zone_development_time':
                    if max_value > 1.0:
                        breakthroughs.append(f"🎉 安全区域时间重大突破: {max_value:.2f}s")
                    elif value > 0.5:
                        breakthroughs.append(f"✅ 安全区域时间有改善: {value:.2f}s")
                
                if metric == 'reconnaissance_completion':
                    if value > 0.85:
                        breakthroughs.append(f"🎉 侦察完成度接近论文: {value:.3f}")
                
                print()
        
        print("\n🚀 系统突破总结:")
        if breakthroughs:
            for breakthrough in breakthroughs:
                print(f"  {breakthrough}")
        else:
            print("  ⚠️ 未发现显著突破，需要更长训练时间")
        
        # 网络架构优势分析
        print(f"\n🧠 超级网络架构优势:")
        print(f"  1024维隐藏层 + 8层深度 + 双重注意力")
        print(f"  专门的协作编码器和干扰编码器")
        print(f"  分离动作输出优化协作行为")
        
        # 训练建议
        print(f"\n💡 训练建议:")
        jamming_coop = final_metrics.get('jamming_cooperation_rate', 0)
        if jamming_coop < 2:
            print("  🔧 建议增加协作权重，延长协作基础训练阶段")
        elif jamming_coop < 10:
            print("  📈 系统正在收敛，建议运行完整1700回合训练")
        else:
            print("  🎉 系统表现优秀，可以期待更高性能")
        
        return True, final_metrics
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def compare_with_previous_results():
    """与之前结果对比"""
    print("\n📈 与之前系统对比:")
    print("之前系统 (quick_fix_test.py 结果):")
    print("  侦察完成度: 0.805 (80.5%)")
    print("  干扰协作率: 0.0% (需要突破)")
    print("  安全区域时间: 0.203s (9.6%)")
    print("  任务成功率: 25.0%")
    print()
    print("超级高级系统预期改进:")
    print("  🎯 干扰协作率: 目标突破5-15%")
    print("  🎯 安全区域时间: 目标突破1.0s+")
    print("  🎯 侦察完成度: 保持85%+水平")
    print("  🎯 整体协作能力: 显著提升")

if __name__ == "__main__":
    compare_with_previous_results()
    
    success, metrics = ultra_advanced_quick_test()
    
    if success:
        print("\n✅ 超级高级系统测试完成!")
        print("🚀 建议运行命令:")
        print("python ultra_advanced_reproduction_system.py  # 1700回合完整训练") 
    else:
        print("\n❌ 系统需要进一步调试") 