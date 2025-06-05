#!/usr/bin/env python3
"""
查看超级高级系统训练结果

用于查看训练完成后的突破成果
"""

import os
import json
import glob
from datetime import datetime

def check_ultra_results():
    """检查超级高级系统结果"""
    print("🔍 查看超级高级系统训练结果")
    print("="*60)
    
    # 查找最新的超级高级结果
    results_pattern = "experiments/ultra_advanced/*/ultra_advanced_results.json"
    result_files = glob.glob(results_pattern)
    
    if not result_files:
        print("❌ 没有找到超级高级训练结果")
        print("💡 请确保训练已完成")
        return
    
    # 获取最新结果
    latest_file = max(result_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        final_metrics = data.get('final_metrics', {})
        achievement_rate = data.get('achievement_rate', 0)
        timestamp = data.get('timestamp', 'unknown')
        
        print(f"📅 训练时间: {timestamp}")
        print(f"🎯 总体达成率: {achievement_rate:.1f}%")
        print()
        
        # 关键突破分析
        print("🚀 关键突破分析:")
        print("-" * 50)
        
        # 干扰协作率突破
        jamming_coop = final_metrics.get('jamming_cooperation_rate', 0)
        jamming_coop_max = final_metrics.get('jamming_cooperation_rate_max', 0)
        
        print(f"干扰协作率:")
        print(f"  平均: {jamming_coop:.1f}% (目标: 5-15%)")
        print(f"  最高: {jamming_coop_max:.1f}%")
        
        if jamming_coop_max > 5:
            print(f"  🎉 重大突破！成功突破0%瓶颈！")
        elif jamming_coop > 1:
            print(f"  ✅ 有改善，继续训练可能更好")
        else:
            print(f"  ⚠️ 仍需进一步优化")
        
        print()
        
        # 安全区域时间突破
        safe_zone = final_metrics.get('safe_zone_development_time', 0)
        safe_zone_max = final_metrics.get('safe_zone_development_time_max', 0)
        
        print(f"安全区域时间:")
        print(f"  平均: {safe_zone:.2f}s (目标: 1.0s+)")
        print(f"  最高: {safe_zone_max:.2f}s")
        
        if safe_zone_max > 1.0:
            print(f"  🎉 重大突破！超过1秒目标！")
        elif safe_zone > 0.5:
            print(f"  ✅ 显著改善")
        else:
            print(f"  ⚠️ 仍需进一步优化")
        
        print()
        
        # 侦察完成度
        recon_comp = final_metrics.get('reconnaissance_completion', 0)
        print(f"侦察完成度:")
        print(f"  当前: {recon_comp:.3f} (目标: 0.97)")
        print(f"  达成率: {min(100, recon_comp/0.97*100):.1f}%")
        
        print()
        
        # 总体评估
        print("🏆 总体评估:")
        if achievement_rate >= 50:
            print(f"  🔥 超级高级系统表现优秀！")
            print(f"  💡 建议运行完整1700回合训练以达到更高水平")
        elif achievement_rate >= 30:
            print(f"  ✅ 超级高级系统运行良好")
            print(f"  💡 建议运行完整训练或调整参数")
        else:
            print(f"  ⚠️ 需要进一步优化")
            print(f"  💡 建议检查系统配置")
        
        # 下一步建议
        print(f"\n🚀 下一步建议:")
        if jamming_coop_max > 5 or safe_zone_max > 1.0:
            print(f"  🎯 已有重大突破！建议运行:")
            print(f"     python ultra_advanced_reproduction_system.py")
            print(f"  🎯 进行1700回合完整训练")
        else:
            print(f"  🔧 建议先优化系统或增加训练回合数")
        
    except Exception as e:
        print(f"❌ 读取结果失败: {e}")

def check_training_progress():
    """检查训练是否还在进行"""
    print(f"\n📊 训练状态检查:")
    
    # 简单的进程检查
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'ultra_advanced'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(f"  ✅ 训练正在进行中...")
            print(f"  ⏱️ 预计还需10-15分钟")
        else:
            print(f"  ✅ 训练可能已完成")
    except:
        print(f"  💡 无法检测训练状态")

if __name__ == "__main__":
    check_ultra_results()
    check_training_progress()
    
    print(f"\n💡 使用提示:")
    print(f"  定期运行此脚本查看最新结果")
    print(f"  python check_ultra_results.py") 