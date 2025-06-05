#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体测试脚本 - 检查系统上可用的中文字体
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def check_chinese_fonts():
    """检查系统上可用的中文字体"""
    print("正在检查系统字体...")
    
    # 获取所有可用字体
    fonts = [font.name for font in fm.fontManager.ttflist]
    
    # 常见的中文字体名称
    chinese_fonts = [
        'PingFang SC',
        'Arial Unicode MS', 
        'Heiti SC',
        'STSong',
        'SimHei',
        'Microsoft YaHei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Source Han Sans SC'
    ]
    
    print("\n=== 系统中可用的中文字体 ===")
    available_chinese_fonts = []
    for font in chinese_fonts:
        if font in fonts:
            print(f"✅ {font}")
            available_chinese_fonts.append(font)
        else:
            print(f"❌ {font}")
    
    return available_chinese_fonts

def test_chinese_display(available_fonts):
    """测试中文字符显示"""
    if not available_fonts:
        print("\n警告：未找到可用的中文字体，将使用默认字体")
        test_font = 'DejaVu Sans'
    else:
        test_font = available_fonts[0]
        print(f"\n使用字体进行测试: {test_font}")
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [test_font]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 测试1：简单的中文标题和标签
    ax1 = axes[0, 0]
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax1.plot(x, y)
    ax1.set_title('算法性能测试')
    ax1.set_xlabel('时间步数')
    ax1.set_ylabel('奖励值')
    
    # 测试2：柱状图
    ax2 = axes[0, 1]
    algorithms = ['AD-PPO', 'MADDPG']
    rewards = [100, 5958]
    bars = ax2.bar(algorithms, rewards)
    ax2.set_title('算法对比')
    ax2.set_ylabel('最终平均奖励')
    
    # 添加数值标签
    for bar, value in zip(bars, rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{value}', ha='center', va='bottom')
    
    # 测试3：多指标对比
    ax3 = axes[1, 0]
    metrics = ['成功率(%)', '干扰率(%)', '训练效率']
    ad_ppo_values = [100, 100, 85]
    maddpg_values = [100, 50, 70]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x_pos - width/2, ad_ppo_values, width, label='AD-PPO', alpha=0.8)
    ax3.bar(x_pos + width/2, maddpg_values, width, label='MADDPG', alpha=0.8)
    
    ax3.set_title('多指标性能对比')
    ax3.set_xlabel('评估指标')
    ax3.set_ylabel('得分')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    
    # 测试4：文本显示
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.8, '项目名称：多无人机电子对抗决策算法', 
             transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.6, '算法类型：AD-PPO vs MADDPG', 
             transform=ax4.transAxes, ha='center', fontsize=11)
    ax4.text(0.5, 0.4, '测试环境：电子对抗仿真环境', 
             transform=ax4.transAxes, ha='center', fontsize=11)
    ax4.text(0.5, 0.2, f'当前字体：{test_font}', 
             transform=ax4.transAxes, ha='center', fontsize=10, style='italic')
    ax4.set_title('中文字符显示测试')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # 保存测试图像
    save_path = 'font_test_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n测试图像已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🎨 中文字体支持测试")
    print("=" * 50)
    
    # 检查可用字体
    available_fonts = check_chinese_fonts()
    
    # 测试中文显示
    test_chinese_display(available_fonts)
    
    print("\n📋 字体配置建议:")
    if available_fonts:
        print(f"✅ 推荐使用字体: {available_fonts[0]}")
        print("✅ 中文字符应该能正常显示")
    else:
        print("⚠️  系统中未找到常见的中文字体")
        print("🔧 建议安装以下字体之一:")
        print("   - 对于 macOS: PingFang SC (通常已预装)")
        print("   - 对于 Windows: Microsoft YaHei")
        print("   - 对于 Linux: WenQuanYi Micro Hei 或 Noto Sans CJK SC")
    
    print("\n🚀 字体配置已应用到项目中，重新运行evaluation.py应该能正常显示中文")

if __name__ == "__main__":
    main() 