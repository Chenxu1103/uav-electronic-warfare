#!/usr/bin/env python3
"""
高级性能监控器

实时监控和分析训练过程中的关键指标：
1. 实时性能曲线绘制
2. 性能瓶颈诊断
3. 训练建议生成
4. 多维度指标分析
5. 论文指标达成预测
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class AdvancedPerformanceMonitor:
    """高级性能监控器"""
    
    def __init__(self):
        self.paper_targets = {
            'reconnaissance_completion': 0.97,
            'safe_zone_development_time': 2.1,
            'reconnaissance_cooperation_rate': 37.0,
            'jamming_cooperation_rate': 34.0,
            'jamming_failure_rate': 23.3
        }
        
        # 性能基准
        self.performance_benchmarks = {
            'excellent': 90,    # 90%+ 论文达成率
            'good': 75,         # 75%+ 论文达成率
            'acceptable': 60,   # 60%+ 论文达成率
            'poor': 60          # 60%以下
        }
        
    def load_experiment_results(self, experiment_dir="experiments"):
        """加载实验结果"""
        print("📊 加载实验结果...")
        
        results = []
        
        # 搜索所有实验目录
        search_patterns = [
            f"{experiment_dir}/*/final_reproduction_results.json",
            f"{experiment_dir}/*/ultra_advanced_results.json",
            f"{experiment_dir}/*/enhanced_reproduction_results.json"
        ]
        
        for pattern in search_patterns:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 提取系统信息
                    system_type = data.get('system_type', 'unknown')
                    timestamp = data.get('timestamp', 'unknown')
                    achievement_rate = data.get('achievement_rate', 0)
                    final_metrics = data.get('final_metrics', {})
                    
                    results.append({
                        'system_type': system_type,
                        'timestamp': timestamp,
                        'achievement_rate': achievement_rate,
                        'metrics': final_metrics,
                        'file_path': file_path
                    })
                    
                except Exception as e:
                    print(f"⚠️ 无法加载 {file_path}: {e}")
        
        print(f"✅ 已加载 {len(results)} 个实验结果")
        return results
    
    def analyze_performance_trends(self, results):
        """分析性能趋势"""
        print("\n📈 性能趋势分析")
        print("="*80)
        
        if not results:
            print("❌ 没有找到实验结果")
            return
        
        # 按时间排序
        sorted_results = sorted(results, key=lambda x: x['timestamp'])
        
        # 分析各系统类型的性能
        system_performance = {}
        for result in sorted_results:
            system_type = result['system_type']
            if system_type not in system_performance:
                system_performance[system_type] = []
            system_performance[system_type].append(result)
        
        print("🎯 各系统性能对比:")
        print("-" * 60)
        
        for system_type, results_list in system_performance.items():
            if results_list:
                latest_result = results_list[-1]
                avg_achievement = np.mean([r['achievement_rate'] for r in results_list])
                max_achievement = max([r['achievement_rate'] for r in results_list])
                
                print(f"{system_type}:")
                print(f"  实验次数: {len(results_list)}")
                print(f"  平均达成率: {avg_achievement:.1f}%")
                print(f"  最高达成率: {max_achievement:.1f}%")
                print(f"  最新达成率: {latest_result['achievement_rate']:.1f}%")
                
                # 性能等级评估
                if max_achievement >= self.performance_benchmarks['excellent']:
                    level = "🔥 优秀"
                elif max_achievement >= self.performance_benchmarks['good']:
                    level = "🎉 良好"
                elif max_achievement >= self.performance_benchmarks['acceptable']:
                    level = "✅ 可接受"
                else:
                    level = "⚠️ 需要改进"
                
                print(f"  性能等级: {level}")
                print()
    
    def analyze_key_metrics_breakdown(self, results):
        """分析关键指标细分"""
        print("\n🔍 关键指标细分分析")
        print("="*80)
        
        if not results:
            return
        
        # 找到最佳结果
        best_result = max(results, key=lambda x: x['achievement_rate'])
        metrics = best_result['metrics']
        
        print(f"📊 最佳性能系统: {best_result['system_type']}")
        print(f"📅 时间: {best_result['timestamp']}")
        print(f"🎯 总体达成率: {best_result['achievement_rate']:.1f}%")
        print()
        
        print("📈 各指标详细分析:")
        print("-" * 100)
        print(f"{'指标':<25} {'论文目标':<12} {'实现值':<12} {'最高值':<12} {'达成率':<10} {'状态':<10} {'分析':<15}")
        print("-" * 100)
        
        target_mapping = {
            'reconnaissance_completion': ('侦察任务完成度', 0.97),
            'safe_zone_development_time': ('安全区域开辟时间', 2.1),
            'reconnaissance_cooperation_rate': ('侦察协作率(%)', 37.0),
            'jamming_cooperation_rate': ('干扰协作率(%)', 34.0),
            'jamming_failure_rate': ('干扰失效率(%)', 23.3),
        }
        
        for key, (name, target) in target_mapping.items():
            if key in metrics:
                value = metrics[key]
                max_value = metrics.get(f'{key}_max', value)
                
                if key == 'jamming_failure_rate':
                    achievement = max(0, 100 - abs(value - target) / target * 100)
                else:
                    achievement = min(100, value / target * 100)
                
                # 状态分析
                if achievement >= 90:
                    status = "🔥 完美"
                    analysis = "已达到论文水准"
                elif achievement >= 75:
                    status = "🎉 优秀"
                    analysis = "接近论文水准"
                elif achievement >= 60:
                    status = "✅ 良好"
                    analysis = "有待提升"
                else:
                    status = "⚠️ 不足"
                    analysis = "需要重点优化"
                
                print(f"{name:<25} {target:<12.3f} {value:<12.3f} {max_value:<12.3f} {achievement:<10.1f} {status:<10} {analysis:<15}")
        
        print("-" * 100)
    
    def diagnose_performance_bottlenecks(self, results):
        """诊断性能瓶颈"""
        print("\n🔧 性能瓶颈诊断")
        print("="*80)
        
        if not results:
            return
        
        # 找到最佳结果进行分析
        best_result = max(results, key=lambda x: x['achievement_rate'])
        metrics = best_result['metrics']
        
        bottlenecks = []
        
        # 检查各项指标
        jamming_coop = metrics.get('jamming_cooperation_rate', 0)
        if jamming_coop < 10:
            bottlenecks.append({
                'type': '干扰协作率',
                'severity': 'critical' if jamming_coop < 2 else 'high',
                'current': jamming_coop,
                'target': 34.0,
                'suggestions': [
                    '增加协作训练模块权重',
                    '延长协作专训阶段时间',
                    '优化干扰机制设计',
                    '增加联合干扰奖励'
                ]
            })
        
        safe_zone_time = metrics.get('safe_zone_development_time', 0)
        if safe_zone_time < 1.0:
            bottlenecks.append({
                'type': '安全区域开辟时间',
                'severity': 'high' if safe_zone_time < 0.5 else 'medium',
                'current': safe_zone_time,
                'target': 2.1,
                'suggestions': [
                    '优化任务完成奖励机制',
                    '增加持续性任务设计',
                    '改进环境状态持久化',
                    '调整时间步长设置'
                ]
            })
        
        recon_coop = metrics.get('reconnaissance_cooperation_rate', 0)
        if recon_coop < 25:
            bottlenecks.append({
                'type': '侦察协作率',
                'severity': 'medium',
                'current': recon_coop,
                'target': 37.0,
                'suggestions': [
                    '强化多智能体协调机制',
                    '优化信息共享奖励',
                    '改进集体决策算法',
                    '增加协作成功检测'
                ]
            })
        
        # 输出瓶颈分析
        if bottlenecks:
            print("🚨 发现性能瓶颈:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                severity_icon = "🔴" if bottleneck['severity'] == 'critical' else "🟡" if bottleneck['severity'] == 'high' else "🟢"
                
                print(f"\n{severity_icon} 瓶颈 {i}: {bottleneck['type']}")
                print(f"   当前值: {bottleneck['current']:.2f}")
                print(f"   目标值: {bottleneck['target']:.2f}")
                print(f"   严重程度: {bottleneck['severity']}")
                print(f"   优化建议:")
                for suggestion in bottleneck['suggestions']:
                    print(f"     • {suggestion}")
        else:
            print("🎉 未发现显著性能瓶颈！系统整体表现良好！")
    
    def generate_optimization_roadmap(self, results):
        """生成优化路线图"""
        print("\n🗺️ 性能优化路线图")
        print("="*80)
        
        if not results:
            return
        
        best_result = max(results, key=lambda x: x['achievement_rate'])
        current_achievement = best_result['achievement_rate']
        
        print(f"🎯 当前总体达成率: {current_achievement:.1f}%")
        print()
        
        # 根据当前性能水平提供路线图
        if current_achievement < 30:
            print("📍 当前阶段: 初级优化")
            print("🎯 短期目标: 达到50%达成率")
            print("📋 优化重点:")
            print("  1. 基础网络架构优化")
            print("  2. 奖励机制调整")
            print("  3. 环境参数配置")
            print("  4. 基础训练稳定性")
            
        elif current_achievement < 60:
            print("📍 当前阶段: 中级优化")
            print("🎯 短期目标: 达到75%达成率")
            print("📋 优化重点:")
            print("  1. 协作机制强化")
            print("  2. 高级网络架构")
            print("  3. 课程学习策略")
            print("  4. 性能监控系统")
            
        elif current_achievement < 80:
            print("📍 当前阶段: 高级优化")
            print("🎯 短期目标: 达到90%达成率")
            print("📋 优化重点:")
            print("  1. 超参数精细调优")
            print("  2. 高级协作算法")
            print("  3. 专业化训练模块")
            print("  4. 性能瓶颈突破")
            
        else:
            print("📍 当前阶段: 顶级优化")
            print("🎯 目标: 达到论文完美水准")
            print("📋 优化重点:")
            print("  1. 极致性能调优")
            print("  2. 稳定性保证")
            print("  3. 泛化能力提升")
            print("  4. 论文级别收敛")
        
        # 推荐下一步行动
        print(f"\n🚀 推荐下一步行动:")
        if current_achievement < 50:
            print("  📁 运行: enhanced_paper_reproduction_test.py")
            print("  🎯 重点: 基础性能稳定")
        elif current_achievement < 75:
            print("  📁 运行: ultra_advanced_reproduction_system.py")
            print("  🎯 重点: 协作能力突破")
        else:
            print("  📁 运行: final_complete_reproduction_system.py")
            print("  🎯 重点: 论文级别收敛")
    
    def plot_performance_curves(self, results):
        """绘制性能曲线"""
        print("\n📊 生成性能可视化图表...")
        
        if not results:
            print("❌ 没有数据可绘制")
            return
        
        # 按系统类型分组
        system_types = {}
        for result in results:
            system_type = result['system_type']
            if system_type not in system_types:
                system_types[system_type] = []
            system_types[system_type].append(result)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('论文复现系统性能分析', fontsize=16, fontweight='bold')
        
        # 1. 总体达成率对比
        ax1 = axes[0, 0]
        system_names = list(system_types.keys())
        avg_achievements = [np.mean([r['achievement_rate'] for r in system_types[name]]) for name in system_names]
        max_achievements = [max([r['achievement_rate'] for r in system_types[name]]) for name in system_names]
        
        x = np.arange(len(system_names))
        width = 0.35
        
        ax1.bar(x - width/2, avg_achievements, width, label='平均达成率', alpha=0.8)
        ax1.bar(x + width/2, max_achievements, width, label='最高达成率', alpha=0.8)
        ax1.set_xlabel('系统类型')
        ax1.set_ylabel('达成率 (%)')
        ax1.set_title('总体达成率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(system_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 关键指标雷达图
        ax2 = axes[0, 1]
        best_result = max(results, key=lambda x: x['achievement_rate'])
        metrics = best_result['metrics']
        
        # 雷达图数据
        labels = ['侦察完成度', '安全区域时间', '侦察协作率', '干扰协作率']
        values = [
            min(100, metrics.get('reconnaissance_completion', 0) / 0.97 * 100),
            min(100, metrics.get('safe_zone_development_time', 0) / 2.1 * 100),
            min(100, metrics.get('reconnaissance_cooperation_rate', 0) / 37.0 * 100),
            min(100, metrics.get('jamming_cooperation_rate', 0) / 34.0 * 100)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, label='当前性能')
        ax2.fill(angles, values, alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, 100)
        ax2.set_title('关键指标达成率')
        ax2.grid(True)
        
        # 3. 性能趋势
        ax3 = axes[1, 0]
        for system_type, results_list in system_types.items():
            if len(results_list) > 1:
                sorted_results = sorted(results_list, key=lambda x: x['timestamp'])
                achievements = [r['achievement_rate'] for r in sorted_results]
                ax3.plot(range(len(achievements)), achievements, 'o-', label=system_type, linewidth=2)
        
        ax3.set_xlabel('实验次数')
        ax3.set_ylabel('达成率 (%)')
        ax3.set_title('性能趋势')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 指标分布
        ax4 = axes[1, 1]
        metrics_data = []
        labels = []
        
        for key, name in [('reconnaissance_completion', '侦察完成度'), 
                         ('jamming_cooperation_rate', '干扰协作率'),
                         ('reconnaissance_cooperation_rate', '侦察协作率')]:
            values = [r['metrics'].get(key, 0) for r in results if key in r['metrics']]
            if values:
                metrics_data.append(values)
                labels.append(name)
        
        if metrics_data:
            ax4.boxplot(metrics_data, labels=labels)
            ax4.set_title('指标分布')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"performance_analysis_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能图表已保存: {save_path}")
        
        # 显示图表
        plt.show()
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 启动高级性能监控分析")
        print("="*80)
        
        # 加载实验结果
        results = self.load_experiment_results()
        
        if not results:
            print("❌ 没有找到实验结果，请先运行一些实验")
            print("💡 建议运行:")
            print("  python quick_fix_test.py")
            print("  python ultra_advanced_quick_test.py")
            return
        
        # 执行各项分析
        self.analyze_performance_trends(results)
        self.analyze_key_metrics_breakdown(results)
        self.diagnose_performance_bottlenecks(results)
        self.generate_optimization_roadmap(results)
        
        # 生成可视化
        try:
            self.plot_performance_curves(results)
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
        
        print("\n✅ 性能分析完成!")
        
        # 提供下一步建议
        best_result = max(results, key=lambda x: x['achievement_rate'])
        print(f"\n🎯 基于当前最佳成果 ({best_result['achievement_rate']:.1f}%) 的建议:")
        
        if best_result['achievement_rate'] < 50:
            print("📁 立即运行: python enhanced_paper_reproduction_test.py")
        elif best_result['achievement_rate'] < 80:
            print("📁 立即运行: python ultra_advanced_quick_test.py")
        else:
            print("📁 立即运行: python ultra_advanced_reproduction_system.py")

def main():
    """主函数"""
    monitor = AdvancedPerformanceMonitor()
    monitor.run_complete_analysis()

if __name__ == "__main__":
    main() 